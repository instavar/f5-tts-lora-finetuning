from __future__ import annotations

import importlib.util
import io
import json
import os
import tarfile
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from instavar_voice_lab.lineage import build_dataset_lineage


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("f5_lifecycle", ROOT / "scripts" / "instavar_voice_lifecycle.py")
assert SPEC and SPEC.loader
LIFECYCLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LIFECYCLE)


class LifecycleBackendTests(unittest.TestCase):
    def _write_wav(self, path: Path) -> None:
        with wave.open(str(path), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(24000)
            audio.writeframes(b"\x00\x00" * 240)

    def _write_vocoder(self, path: Path) -> None:
        path.mkdir()
        (path / "config.yaml").write_text("feature_extractor: fixture\n", encoding="utf-8")
        (path / "pytorch_model.bin").write_bytes(b"vocoder")

    def _build_package(
        self,
        root: Path,
        *,
        base: Path,
        reference_audio: Path,
        vocoder: Path,
        vocabulary: Path | None = None,
        add_unbound_file: bool = False,
        corrupt_adapter: bool = False,
    ) -> Path:
        package = root / "package-source"
        package.mkdir()
        adapter = root / "adapter-source"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
        if corrupt_adapter:
            (package / "selected-adapter.tar").write_bytes(b"not a tar")
        else:
            LIFECYCLE._archive(adapter, package / "selected-adapter.tar", arcname="adapter")
        for name in LIFECYCLE.PACKAGE_MEMBER_NAMES - {"selected-adapter.tar"}:
            (package / name).write_bytes(f"fixture:{name}\n".encode())
        if add_unbound_file:
            (package / "unbound.txt").write_text("not in manifest\n", encoding="utf-8")
        files = [
            {"path": path.name, "sha256": LIFECYCLE._sha256(path), "bytes": path.stat().st_size}
            for path in sorted(package.iterdir())
            if path.name in LIFECYCLE.PACKAGE_MEMBER_NAMES
        ]
        LIFECYCLE._write_json(
            package / "package-manifest.json",
            {
                "schema_version": LIFECYCLE.PACKAGE_SCHEMA_VERSION,
                "backend_id": "f5-tts-lora-pytorch",
                "model": "F5TTS_v1_Base",
                "companion_revision": "a" * 40,
                "external_dependencies": {
                    "base_checkpoint": LIFECYCLE._external_file_identity(base),
                    "reference_audio": LIFECYCLE._external_file_identity(reference_audio),
                    "vocabulary": LIFECYCLE._external_file_identity(vocabulary) if vocabulary else None,
                    "vocoder": LIFECYCLE._external_tree_identity(vocoder),
                },
                "inference_contract": {
                    "reference_text_sha256": LIFECYCLE._text_sha256("Reference transcript."),
                    "seed": 42,
                    "smoke_text_sha256": LIFECYCLE._text_sha256(LIFECYCLE.DEFAULT_SMOKE_TEXT),
                },
                "files": files,
                "evidence_boundary": "fixture",
            },
        )
        archive = root / "package.tar"
        LIFECYCLE._archive(package, archive, arcname="package")
        return archive

    def test_backend_routes_all_stages_and_binds_lora(self) -> None:
        spec = json.loads((ROOT / "instavar-voice-backend.json").read_text())
        self.assertEqual(spec["schema_version"], "1.2.0")
        self.assertEqual(spec["capability_binding"]["adaptation"], "lora")
        required = {item["name"] for item in spec["required_environment"]}
        self.assertIn("PERSISTED_PACKAGE_ROOT", required)
        self.assertIn("VOCODER_DIR", required)
        self.assertIn("package/persisted-package.json", spec["expected_artifacts"]["package"])
        for stage in ("preflight", "train", "infer", "evaluate", "package"):
            self.assertEqual(spec["commands"][stage][-1], stage)

    def test_stage_result_is_required_before_restore_mutates_state(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(LIFECYCLE, "_restore") as restore,
            self.assertRaisesRegex(ValueError, "INSTAVAR_VOICE_STAGE_RESULT"),
        ):
            LIFECYCLE.run("restore")
        restore.assert_not_called()

    def test_preflight_binding_rejects_file_and_training_control_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            (work / "preflight").mkdir(parents=True)
            paths = {}
            for name in (
                "base",
                "lineage",
                "experiment",
                "plan",
                "train",
                "validation",
                "test",
                "reference",
            ):
                path = root / name
                path.write_bytes(f"{name}\n".encode())
                paths[name] = path
            vocoder = root / "vocoder"
            self._write_vocoder(vocoder)
            environment = {
                "INSTAVAR_VOICE_WORK_DIR": str(work),
                "BASE_MODEL_CHECKPOINT": str(paths["base"]),
                "DATASET_LINEAGE": str(paths["lineage"]),
                "INSTAVAR_VOICE_EXPERIMENT_MANIFEST": str(paths["experiment"]),
                "GENERATION_PLAN": str(paths["plan"]),
                "RAW_TRAIN_JSONL": str(paths["train"]),
                "RAW_VALIDATION_JSONL": str(paths["validation"]),
                "RAW_TEST_JSONL": str(paths["test"]),
                "REFERENCE_AUDIO": str(paths["reference"]),
                "VOCODER_DIR": str(vocoder),
                "REFERENCE_TEXT": "Reference transcript.",
                "CANDIDATE_ID": "f5-fixture",
                "DATASET_NAME": "fixture",
                "SELECTED_ADAPTER_NAME": "lora_1",
            }
            with (
                patch.dict(os.environ, environment, clear=True),
                patch.object(LIFECYCLE, "_git_head", return_value="a" * 40),
            ):
                bound = LIFECYCLE._bound_lifecycle_inputs()
                LIFECYCLE._write_json(
                    work / "preflight" / "preflight.json",
                    {
                        "schema_version": LIFECYCLE.PREFLIGHT_SCHEMA_VERSION,
                        "status": "passed",
                        "companion_revision": "a" * 40,
                        "bound_inputs": bound,
                    },
                )
                self.assertEqual(LIFECYCLE._verified_preflight()["bound_inputs"], bound)
                paths["plan"].write_bytes(b"changed plan\n")
                with self.assertRaisesRegex(ValueError, "inputs or controls changed"):
                    LIFECYCLE._verified_preflight()
                paths["plan"].write_bytes(b"plan\n")
                os.environ["EPOCHS"] = "21"
                with self.assertRaisesRegex(ValueError, "inputs or controls changed"):
                    LIFECYCLE._verified_preflight()
                os.environ["EPOCHS"] = "20"
                os.environ["SMOKE_SEED"] = "43"
                with self.assertRaisesRegex(ValueError, "inputs or controls changed"):
                    LIFECYCLE._verified_preflight()
                os.environ["SMOKE_SEED"] = "42"
                os.environ["TRAINING_SEED"] = "667"
                with self.assertRaisesRegex(ValueError, "inputs or controls changed"):
                    LIFECYCLE._verified_preflight()

    def test_selected_adapter_is_one_safe_child(self) -> None:
        self.assertEqual(LIFECYCLE._safe_name("lora_1250"), "lora_1250")
        for unsafe in ("", ".", "..", "../lora_last", "nested/lora_last", "/lora_last"):
            with self.assertRaises(ValueError):
                LIFECYCLE._safe_name(unsafe)

    def test_lifecycle_resume_requires_preflight_authority_and_exact_output_child(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            checkpoint = output / "lora_1"
            checkpoint.mkdir()
            preflight = {"bound_inputs": {"values": {"allow_train_resume": True}}}
            with patch.dict(os.environ, {"TRAIN_RESUME_FROM": str(checkpoint)}, clear=True):
                self.assertEqual(LIFECYCLE._training_resume_checkpoint(output, preflight), checkpoint.resolve())
                denied = {"bound_inputs": {"values": {"allow_train_resume": False}}}
                with self.assertRaisesRegex(ValueError, "ALLOW_TRAIN_RESUME=1"):
                    LIFECYCLE._training_resume_checkpoint(output, denied)
                outside = root / "lora_1"
                outside.mkdir()
                os.environ["TRAIN_RESUME_FROM"] = str(outside)
                with self.assertRaisesRegex(ValueError, "lifecycle output"):
                    LIFECYCLE._training_resume_checkpoint(output, preflight)

    def test_allow_train_resume_is_one_strict_preflight_bit(self) -> None:
        for value, expected in (("0", False), ("1", True)):
            with patch.dict(os.environ, {"ALLOW_TRAIN_RESUME": value}, clear=True):
                self.assertIs(LIFECYCLE._allow_train_resume(), expected)
        for value in ("true", "yes", "2", ""):
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {"ALLOW_TRAIN_RESUME": value}, clear=True),
                self.assertRaisesRegex(ValueError, "must be 0 or 1"),
            ):
                LIFECYCLE._allow_train_resume()

    def test_trainer_accepts_explicit_checkpoint_path(self) -> None:
        source = (ROOT / "src" / "f5_tts" / "train" / "finetune_cli.py").read_text()
        self.assertIn('"--checkpoint_path"', source)
        self.assertIn("args.checkpoint_path or", source)

    def test_lifecycle_launches_accelerate_through_current_python(self) -> None:
        source = (ROOT / "scripts" / "instavar_voice_lifecycle.py").read_text(encoding="utf-8")
        self.assertIn('"accelerate.commands.launch"', source)
        self.assertNotIn('[\n        "accelerate",\n        "launch",', source)

    def test_inference_cli_accepts_only_explicit_local_vocoder_mode(self) -> None:
        source = (ROOT / "src" / "f5_tts" / "infer" / "infer_cli.py").read_text(encoding="utf-8")
        self.assertIn('"--vocoder_local_path"', source)
        self.assertIn("requires --load_vocoder_from_local", source)
        self.assertIn('"--seed"', source)
        self.assertIn("seed_everything(seed)", source)

    def test_smoke_seed_rejects_ambiguous_or_out_of_range_values(self) -> None:
        self.assertEqual(LIFECYCLE._seed(42, label="fixture"), 42)
        self.assertEqual(LIFECYCLE._seed("42", label="fixture"), 42)
        for value in (True, "", "01", "1.0", -1, 2**63):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "integer from 0"):
                LIFECYCLE._seed(value, label="fixture")

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

    def test_package_extract_rejects_traversal_links_and_duplicate_members(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archives: dict[str, Path] = {}
            traversal = root / "traversal.tar"
            with tarfile.open(traversal, "w") as archive:
                member = tarfile.TarInfo("package/../escape")
                member.size = 1
                archive.addfile(member, io.BytesIO(b"x"))
            archives["traversal"] = traversal
            linked = root / "linked.tar"
            with tarfile.open(linked, "w") as archive:
                member = tarfile.TarInfo("package/link")
                member.type = tarfile.SYMTYPE
                member.linkname = "/tmp/target"
                archive.addfile(member)
            archives["linked"] = linked
            duplicate = root / "duplicate.tar"
            with tarfile.open(duplicate, "w") as archive:
                for payload in (b"first", b"second"):
                    member = tarfile.TarInfo("package/file")
                    member.size = len(payload)
                    archive.addfile(member, io.BytesIO(payload))
            archives["duplicate"] = duplicate
            for name, source in archives.items():
                with (
                    self.subTest(name=name),
                    self.assertRaisesRegex(ValueError, "unsafe lifecycle package member"),
                ):
                    LIFECYCLE._extract_package(source, root / f"{name}-output")

    def test_restore_verifies_package_and_publishes_only_after_fresh_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base.safetensors"
            base.write_bytes(b"base")
            reference = root / "reference.wav"
            self._write_wav(reference)
            vocoder = root / "vocoder"
            self._write_vocoder(vocoder)
            package = self._build_package(root, base=base, reference_audio=reference, vocoder=vocoder)
            destination = root / "restored"
            commands: list[list[str]] = []

            def fake_run(command: list[str], *, capture: bool = False) -> str:
                commands.append(command)
                output = Path(command[command.index("--output_dir") + 1]) / command[command.index("--output_file") + 1]
                self._write_wav(output)
                return ""

            environment = {
                "PERSISTED_PACKAGE_PATH": str(package),
                "EXPECTED_PACKAGE_SHA256": LIFECYCLE._sha256(package),
                "BASE_MODEL_CHECKPOINT": str(base),
                "REFERENCE_AUDIO": str(reference),
                "VOCODER_DIR": str(vocoder),
                "REFERENCE_TEXT": "Reference transcript.",
                "RESTORE_OUTPUT_DIR": str(destination),
            }
            with (
                patch.dict(os.environ, environment, clear=False),
                patch.object(LIFECYCLE, "_run", side_effect=fake_run),
            ):
                LIFECYCLE._restore()

            self.assertEqual(len(commands), 1)
            self.assertIn("--load_vocoder_from_local", commands[0])
            self.assertEqual(commands[0][commands[0].index("--vocoder_local_path") + 1], str(vocoder.resolve()))
            self.assertEqual(commands[0][commands[0].index("--seed") + 1], "42")
            self.assertTrue((destination / "restored-adapter" / "adapter" / "adapter_model.safetensors").is_file())
            receipt = json.loads((destination / "restore-receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "passed")
            self.assertEqual(receipt["package_sha256"], LIFECYCLE._sha256(package))
            self.assertEqual(receipt["restored_wav"]["sample_rate_hz"], 24000)
            self.assertFalse(any(path.name.startswith(".restored.partial.") for path in root.iterdir()))

    def test_restore_fails_closed_on_package_base_vocab_and_inference_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base.safetensors"
            base.write_bytes(b"base")
            wrong_base = root / "wrong-base.safetensors"
            wrong_base.write_bytes(b"wrong")
            wrong_reference = root / "wrong-reference.wav"
            self._write_wav(wrong_reference)
            with wrong_reference.open("ab") as audio:
                audio.write(b"different")
            vocabulary = root / "vocab.txt"
            vocabulary.write_text("token\n", encoding="utf-8")
            reference = root / "reference.wav"
            self._write_wav(reference)
            vocoder = root / "vocoder"
            self._write_vocoder(vocoder)
            wrong_vocoder = root / "wrong-vocoder"
            self._write_vocoder(wrong_vocoder)
            (wrong_vocoder / "pytorch_model.bin").write_bytes(b"wrong vocoder")
            package = self._build_package(
                root,
                base=base,
                reference_audio=reference,
                vocoder=vocoder,
                vocabulary=vocabulary,
            )
            common = {
                "PERSISTED_PACKAGE_PATH": str(package),
                "EXPECTED_PACKAGE_SHA256": LIFECYCLE._sha256(package),
                "BASE_MODEL_CHECKPOINT": str(base),
                "REFERENCE_AUDIO": str(reference),
                "VOCODER_DIR": str(vocoder),
                "REFERENCE_TEXT": "Reference transcript.",
            }
            cases = (
                ({"EXPECTED_PACKAGE_SHA256": "0" * 64, "VOCAB_FILE": str(vocabulary)}, "EXPECTED_PACKAGE_SHA256"),
                ({"BASE_MODEL_CHECKPOINT": str(wrong_base), "VOCAB_FILE": str(vocabulary)}, "base checkpoint"),
                ({"VOCAB_FILE": str(vocabulary), "VOCODER_DIR": str(wrong_vocoder)}, "vocoder directory"),
                ({"VOCAB_FILE": str(vocabulary), "REFERENCE_AUDIO": str(wrong_reference)}, "reference audio"),
                ({"VOCAB_FILE": str(vocabulary), "REFERENCE_TEXT": "Different transcript."}, "REFERENCE_TEXT"),
                ({"VOCAB_FILE": str(vocabulary), "SMOKE_TEXT": "Different smoke text."}, "SMOKE_TEXT"),
                ({"VOCAB_FILE": str(vocabulary), "SMOKE_SEED": "43"}, "SMOKE_SEED"),
                ({}, "requires an external vocabulary"),
            )
            for index, (override, message) in enumerate(cases):
                destination = root / f"failed-{index}"
                environment = {**common, **override, "RESTORE_OUTPUT_DIR": str(destination)}
                with (
                    self.subTest(message=message),
                    patch.dict(os.environ, environment, clear=True),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    LIFECYCLE._restore()
                self.assertFalse(destination.exists())

            destination = root / "failed-inference"
            environment = {
                **common,
                "VOCAB_FILE": str(vocabulary),
                "RESTORE_OUTPUT_DIR": str(destination),
            }
            with (
                patch.dict(os.environ, environment, clear=True),
                patch.object(LIFECYCLE, "_run", side_effect=RuntimeError("inference failed")),
                self.assertRaisesRegex(RuntimeError, "inference failed"),
            ):
                LIFECYCLE._restore()
            self.assertFalse(destination.exists())
            self.assertFalse(any(path.name.startswith(".failed-inference.partial.") for path in root.iterdir()))

    def test_restore_rejects_unbound_files_and_corrupt_inner_adapter(self) -> None:
        for condition in ("unbound", "corrupt-adapter"):
            with self.subTest(condition=condition), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                base = root / "base.safetensors"
                base.write_bytes(b"base")
                reference = root / "reference.wav"
                self._write_wav(reference)
                vocoder = root / "vocoder"
                self._write_vocoder(vocoder)
                package = self._build_package(
                    root,
                    base=base,
                    reference_audio=reference,
                    vocoder=vocoder,
                    add_unbound_file=condition == "unbound",
                    corrupt_adapter=condition == "corrupt-adapter",
                )
                destination = root / "restored"
                environment = {
                    "PERSISTED_PACKAGE_PATH": str(package),
                    "EXPECTED_PACKAGE_SHA256": LIFECYCLE._sha256(package),
                    "BASE_MODEL_CHECKPOINT": str(base),
                    "REFERENCE_AUDIO": str(reference),
                    "VOCODER_DIR": str(vocoder),
                    "REFERENCE_TEXT": "Reference transcript.",
                    "RESTORE_OUTPUT_DIR": str(destination),
                }
                with patch.dict(os.environ, environment, clear=True), self.assertRaises((ValueError, tarfile.TarError)):
                    LIFECYCLE._restore()
                self.assertFalse(destination.exists())

    def test_package_manifest_binds_vocabulary_and_round_trips_through_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            store = root / "store"
            for path in (
                work / "preflight",
                work / "train",
                work / "infer",
                work / "evaluate",
                store,
            ):
                path.mkdir(parents=True)
            base = root / "base.safetensors"
            base.write_bytes(b"base")
            vocabulary = root / "vocab.txt"
            vocabulary.write_text("token\n", encoding="utf-8")
            vocoder = root / "vocoder"
            self._write_vocoder(vocoder)
            reference = root / "reference.wav"
            self._write_wav(reference)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            LIFECYCLE._archive(adapter, work / "train" / "selected-adapter.tar", arcname="adapter")
            (work / "evaluate" / "evaluation-bundle.tar").write_bytes(b"evaluation")
            self._write_wav(work / "infer" / "candidate.wav")
            identity = store.stat()
            LIFECYCLE._write_json(
                work / "preflight" / "preflight.json",
                {"fixture": True},
            )
            preflight = {
                "schema_version": LIFECYCLE.PREFLIGHT_SCHEMA_VERSION,
                "status": "passed",
                "companion_revision": "a" * 40,
                "model": "F5TTS_v1_Base",
                "base_checkpoint_sha256": LIFECYCLE._sha256(base),
                "base_checkpoint_bytes": base.stat().st_size,
                "vocabulary": LIFECYCLE._external_file_identity(vocabulary),
                "bound_inputs": {
                    "files": {
                        "reference_audio": LIFECYCLE._external_file_identity(reference),
                        "vocoder": LIFECYCLE._external_tree_identity(vocoder),
                    },
                    "values": {
                        "reference_text_sha256": LIFECYCLE._text_sha256("Reference transcript."),
                        "smoke_seed": 42,
                        "smoke_text_sha256": LIFECYCLE._text_sha256(LIFECYCLE.DEFAULT_SMOKE_TEXT),
                    },
                },
                "persistent_package_root": str(store.resolve()),
                "persistence_probe": {"device": identity.st_dev, "inode": identity.st_ino},
            }
            documents = {}
            for name in ("experiment", "plan", "lineage"):
                path = root / f"{name}.json"
                path.write_text("{}\n", encoding="utf-8")
                documents[name] = path
            environment = {
                "INSTAVAR_VOICE_WORK_DIR": str(work),
                "PERSISTED_PACKAGE_ROOT": str(store),
                "INSTAVAR_VOICE_EXPERIMENT_MANIFEST": str(documents["experiment"]),
                "GENERATION_PLAN": str(documents["plan"]),
                "DATASET_LINEAGE": str(documents["lineage"]),
                "VOCAB_FILE": str(vocabulary),
                "VOCODER_DIR": str(vocoder),
                "MODEL": "F5TTS_v1_Base",
            }
            with (
                patch.dict(os.environ, environment, clear=False),
                patch.object(LIFECYCLE, "_verified_preflight", return_value=preflight),
            ):
                LIFECYCLE._package()
            package = work / "package" / "adapter-package.tar"
            unpacked = LIFECYCLE._extract_package(package, root / "unpacked")
            manifest = LIFECYCLE._verify_package_contents(
                unpacked,
                base_checkpoint=base,
                vocab_file=vocabulary,
                vocoder_dir=vocoder,
                reference_audio=reference,
            )
            self.assertEqual(manifest["schema_version"], LIFECYCLE.PACKAGE_SCHEMA_VERSION)
            self.assertEqual(manifest["external_dependencies"]["vocabulary"]["sha256"], LIFECYCLE._sha256(vocabulary))
            self.assertEqual(manifest["inference_contract"]["seed"], 42)
            receipt = json.loads((work / "package" / "persisted-package.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["package_sha256"], LIFECYCLE._sha256(package))

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
