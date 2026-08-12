from __future__ import annotations

import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
