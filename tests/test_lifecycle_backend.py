from __future__ import annotations

import importlib.util
import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from instavar_voice_lab.lineage import build_dataset_lineage


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("f5_lifecycle", ROOT / "scripts" / "instavar_voice_lifecycle.py")
assert SPEC and SPEC.loader
LIFECYCLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LIFECYCLE)


class LifecycleBackendTests(unittest.TestCase):
    def test_backend_routes_all_stages_and_binds_lora(self) -> None:
        spec = json.loads((ROOT / "instavar-voice-backend.json").read_text())
        self.assertEqual(spec["schema_version"], "1.2.0")
        self.assertEqual(spec["capability_binding"]["adaptation"], "lora")
        required = {item["name"] for item in spec["required_environment"]}
        self.assertIn("PERSISTED_PACKAGE_ROOT", required)
        self.assertIn("package/persisted-package.json", spec["expected_artifacts"]["package"])
        for stage in ("preflight", "train", "infer", "evaluate", "package"):
            self.assertEqual(spec["commands"][stage][-1], stage)

    def test_selected_adapter_is_one_safe_child(self) -> None:
        self.assertEqual(LIFECYCLE._safe_name("lora_1250"), "lora_1250")
        for unsafe in ("", ".", "..", "../lora_last", "nested/lora_last", "/lora_last"):
            with self.assertRaises(ValueError):
                LIFECYCLE._safe_name(unsafe)

    def test_trainer_accepts_explicit_checkpoint_path(self) -> None:
        source = (ROOT / "src" / "f5_tts" / "train" / "finetune_cli.py").read_text()
        self.assertIn('"--checkpoint_path"', source)
        self.assertIn("args.checkpoint_path or", source)

    def test_archive_rejects_empty_or_symlinked_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter = root / "adapter"
            adapter.mkdir()
            with self.assertRaises(ValueError):
                LIFECYCLE._archive(adapter, root / "empty.tar", arcname="adapter")
            target = root / "outside"
            target.write_bytes(b"outside")
            (adapter / "adapter_model.safetensors").symlink_to(target)
            with self.assertRaises(ValueError):
                LIFECYCLE._archive(adapter, root / "linked.tar", arcname="adapter")

    def test_extract_rejects_empty_and_special_archives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            empty = root / "empty.tar"
            with tarfile.open(empty, "w"):
                pass
            with self.assertRaises(ValueError):
                LIFECYCLE._extract(empty, root / "empty-output")

            special = root / "special.tar"
            with tarfile.open(special, "w") as archive:
                member = tarfile.TarInfo("adapter/device")
                member.type = tarfile.CHRTYPE
                archive.addfile(member, io.BytesIO())
            with self.assertRaises(ValueError):
                LIFECYCLE._extract(special, root / "special-output")

    def test_extract_rejects_sibling_traversal_and_duplicate_members(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases: dict[str, list[tuple[str, bytes]]] = {
                "sibling": [("adapter/model.safetensors", b"model"), ("notes.txt", b"extra")],
                "traversal": [("adapter/../escape.bin", b"escape")],
                "duplicate": [
                    ("adapter/model.safetensors", b"first"),
                    ("adapter/model.safetensors", b"second"),
                ],
            }
            for name, entries in cases.items():
                source = root / f"{name}.tar"
                with tarfile.open(source, "w") as archive:
                    for member_name, payload in entries:
                        member = tarfile.TarInfo(member_name)
                        member.size = len(payload)
                        archive.addfile(member, io.BytesIO(payload))
                with (
                    self.subTest(name=name),
                    self.assertRaisesRegex(ValueError, "unsafe adapter archive member"),
                ):
                    LIFECYCLE._extract(source, root / f"{name}-output")

    def test_persist_package_is_content_addressed_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "adapter-package.tar"
            source.write_bytes(b"immutable package")
            store = root / "store"
            store.mkdir()

            first = LIFECYCLE._persist_package(source, store)
            destination = Path(first["persisted_path"])
            self.assertEqual(first["adaptation_mode"], "lora")
            self.assertTrue(destination.name.startswith("f5-tts-lora-package-sha256-"))
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            self.assertFalse(first["reused_existing"])

            second = LIFECYCLE._persist_package(source, store)
            self.assertEqual(second["package_sha256"], first["package_sha256"])
            self.assertTrue(second["reused_existing"])

            destination.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                LIFECYCLE._persist_package(source, store)

    def test_persistent_package_root_rejects_work_and_checkout_trees(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            work.mkdir()
            checkout = root / "checkout"
            checkout.mkdir()
            (checkout / "packages").mkdir()
            environments = (
                ({"PERSISTED_PACKAGE_ROOT": str(work)}, "outside the lifecycle work directory"),
                ({"PERSISTED_PACKAGE_ROOT": str(checkout / "packages")}, "outside the repository checkout"),
            )
            for override, message in environments:
                with (
                    self.subTest(override=override),
                    patch.dict(
                        os.environ,
                        {"INSTAVAR_VOICE_WORK_DIR": str(work), **override},
                        clear=False,
                    ),
                    patch.object(LIFECYCLE, "REPO_ROOT", checkout),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    LIFECYCLE._persistent_package_root()

    def test_persistence_probe_leaves_no_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = LIFECYCLE._probe_persistent_package_root(root)
            identity = root.stat()
            self.assertTrue(result["writable"])
            self.assertTrue(result["atomic_hard_link"])
            self.assertEqual(result["device"], identity.st_dev)
            self.assertEqual(result["inode"], identity.st_ino)
            self.assertEqual(list(root.iterdir()), [])

    def test_persistence_probe_does_not_unlink_a_link_it_did_not_create(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch.object(LIFECYCLE.os, "link", side_effect=FileExistsError("collision")),
                patch.object(Path, "unlink", autospec=True) as unlink,
                self.assertRaisesRegex(ValueError, "cannot publish an atomic package"),
            ):
                LIFECYCLE._probe_persistent_package_root(root)
            unlinked = [call.args[0] for call in unlink.call_args_list]
            self.assertEqual(len(unlinked), 1)
            self.assertTrue(str(unlinked[0]).endswith(".partial"))

    def test_package_root_is_bound_to_preflight_path_device_and_inode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            store = root / "store"
            other = root / "other"
            for path in (work, store, other):
                path.mkdir()
            identity = store.stat()
            environment = {
                "INSTAVAR_VOICE_WORK_DIR": str(work),
                "PERSISTED_PACKAGE_ROOT": str(store),
            }
            preflight = {
                "persistent_package_root": str(store.resolve()),
                "persistence_probe": {"device": identity.st_dev, "inode": identity.st_ino},
            }
            with patch.dict(os.environ, environment, clear=False):
                self.assertEqual(LIFECYCLE._locked_persistent_package_root(preflight), store.resolve())
                changed_path = {**preflight, "persistent_package_root": str(other)}
                with self.assertRaisesRegex(ValueError, "changed after preflight"):
                    LIFECYCLE._locked_persistent_package_root(changed_path)
                changed_device = {
                    **preflight,
                    "persistence_probe": {"device": identity.st_dev + 1, "inode": identity.st_ino},
                }
                with self.assertRaisesRegex(ValueError, "changed after preflight"):
                    LIFECYCLE._locked_persistent_package_root(changed_device)
                changed_inode = {
                    **preflight,
                    "persistence_probe": {"device": identity.st_dev, "inode": identity.st_ino + 1},
                }
                with self.assertRaisesRegex(ValueError, "changed after preflight"):
                    LIFECYCLE._locked_persistent_package_root(changed_inode)

                store.rename(root / "retired-store")
                store.mkdir()
                self.assertEqual(store.stat().st_dev, identity.st_dev)
                with self.assertRaisesRegex(ValueError, "changed after preflight"):
                    LIFECYCLE._locked_persistent_package_root(preflight)

    def test_dataset_lineage_binds_raw_splits_to_implicit_f5_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw: dict[str, Path] = {}
            for split in ("train", "validation", "test"):
                audio = root / f"{split}.wav"
                audio.write_bytes(b"audio")
                manifest = root / f"{split}.jsonl"
                manifest.write_text(json.dumps({"audio": str(audio), "text": split}) + "\n")
                raw[split] = manifest
            prepared = root / "data" / "fixture_pinyin"
            (prepared / "raw").mkdir(parents=True)
            (prepared / "raw" / "data.arrow").write_bytes(b"arrow")
            (prepared / "duration.json").write_text("{}\n")
            receipt = root / "dataset-lineage.json"
            receipt.write_text(
                json.dumps(
                    build_dataset_lineage(
                        lineage_id="f5-fixture-v1",
                        producer_repository="instavar/f5-tts-lora-finetuning",
                        producer_revision="a" * 40,
                        inputs={
                            "raw_train": (raw["train"], "file"),
                            "raw_validation": (raw["validation"], "file"),
                            "raw_test": (raw["test"], "file"),
                        },
                        outputs={"prepared_dataset": (prepared, "tree")},
                    )
                )
            )
            environment = {
                "DATASET_NAME": "fixture",
                "DATASET_LINEAGE": str(receipt),
                "RAW_TRAIN_JSONL": str(raw["train"]),
                "RAW_VALIDATION_JSONL": str(raw["validation"]),
                "RAW_TEST_JSONL": str(raw["test"]),
            }
            with (
                patch.dict(os.environ, environment, clear=False),
                patch.object(LIFECYCLE, "REPO_ROOT", root),
                patch.object(LIFECYCLE, "_git_head", return_value="a" * 40),
            ):
                report = LIFECYCLE._verify_dataset_lineage()
            self.assertEqual(report["lineage_id"], "f5-fixture-v1")
            (prepared / "raw" / "data.arrow").write_bytes(b"changed")
            with (
                patch.dict(os.environ, environment, clear=False),
                patch.object(LIFECYCLE, "REPO_ROOT", root),
                patch.object(LIFECYCLE, "_git_head", return_value="a" * 40),
                self.assertRaisesRegex(ValueError, "prepared_dataset"),
            ):
                LIFECYCLE._verify_dataset_lineage()


if __name__ == "__main__":
    unittest.main()
