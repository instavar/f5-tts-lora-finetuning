from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from f5_tts.train.lora_initial_adapter import (  # noqa: E402
    RECEIPT_NAME,
    InitialAdapterError,
    initial_adapter_identity,
    publish_initial_adapter,
    sha256_file,
    validate_initial_adapter_config,
)


class FakePeftModel:
    def save_pretrained(self, output: Path, *, safe_serialization: bool) -> None:
        if not safe_serialization:
            raise AssertionError("safe serialization must be enabled")
        (output / "adapter_config.json").write_text(
            json.dumps(
                {
                    "r": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.0,
                    "target_modules": ["to_k", "to_q"],
                    "bias": "none",
                }
            ),
            encoding="utf-8",
        )
        (output / "adapter_model.safetensors").write_bytes(b"initial-adapter")


class InitialAdapterContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.base = self.root / "base.safetensors"
        self.base.write_bytes(b"base")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def publish(self, name: str = "initial") -> Path:
        return publish_initial_adapter(
            FakePeftModel(),
            self.root / name,
            producer_revision="a" * 40,
            base_checkpoint=self.base,
            seed=42,
            rank=8,
            alpha=16,
            dropout=0.0,
            target_modules=["to_q", "to_k"],
        )

    def test_atomic_publication_round_trip_and_configuration(self) -> None:
        adapter = self.publish()
        identity = initial_adapter_identity(adapter)
        self.assertEqual(
            {entry["path"] for entry in identity["files"]},
            {"adapter_config.json", "adapter_model.safetensors", RECEIPT_NAME},
        )
        validate_initial_adapter_config(
            adapter,
            rank=8,
            alpha=16,
            dropout=0.0,
            target_modules=["to_k", "to_q"],
        )
        self.assertFalse(any(path.name.endswith(".partial") for path in self.root.iterdir()))

    def test_publication_rejects_overwrite_and_bad_revision(self) -> None:
        adapter = self.publish()
        with self.assertRaises(FileExistsError):
            publish_initial_adapter(
                FakePeftModel(),
                adapter,
                producer_revision="a" * 40,
                base_checkpoint=self.base,
                seed=42,
                rank=8,
                alpha=16,
                dropout=0.0,
                target_modules=["to_q", "to_k"],
            )
        with self.assertRaisesRegex(InitialAdapterError, "40-character"):
            publish_initial_adapter(
                FakePeftModel(),
                self.root / "bad-revision",
                producer_revision="main",
                base_checkpoint=self.base,
                seed=42,
                rank=8,
                alpha=16,
                dropout=0.0,
                target_modules=["to_q", "to_k"],
            )

    def test_live_file_or_receipt_drift_fails_closed(self) -> None:
        adapter = self.publish()
        (adapter / "adapter_model.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(InitialAdapterError, "live file tree"):
            initial_adapter_identity(adapter)

        other = self.publish("receipt-drift")
        receipt_path = other / RECEIPT_NAME
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["files"][0]["sha256"] = "0" * 64
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(InitialAdapterError, "live file tree"):
            initial_adapter_identity(other)

        metadata_drift = self.publish("metadata-drift")
        receipt_path = metadata_drift / RECEIPT_NAME
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["lora"]["rank"] = 16
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(InitialAdapterError, "configuration drift"):
            initial_adapter_identity(metadata_drift)

    def test_symlink_and_cross_file_hardlink_fail_closed(self) -> None:
        adapter = self.publish()
        (adapter / "alias").symlink_to(adapter / "adapter_config.json")
        with self.assertRaisesRegex(InitialAdapterError, "symlinks"):
            initial_adapter_identity(adapter)

        hardlinked = self.publish("hardlinked")
        receipt_path = hardlinked / RECEIPT_NAME
        receipt_path.unlink()
        os.link(hardlinked / "adapter_config.json", receipt_path)
        with self.assertRaisesRegex(InitialAdapterError, "hardlinks"):
            initial_adapter_identity(hardlinked)

    def test_lora_configuration_drift_is_rejected(self) -> None:
        adapter = self.publish()
        with self.assertRaisesRegex(InitialAdapterError, "configuration drift"):
            validate_initial_adapter_config(
                adapter,
                rank=16,
                alpha=16,
                dropout=0.0,
                target_modules=["to_k", "to_q"],
            )

    def test_receipt_hashes_the_pre_receipt_tree(self) -> None:
        adapter = self.publish()
        receipt = json.loads((adapter / RECEIPT_NAME).read_text(encoding="utf-8"))
        self.assertEqual(
            receipt["files"],
            [
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in sorted([adapter / "adapter_config.json", adapter / "adapter_model.safetensors"])
            ],
        )


class InitialAdapterSourceContractTest(unittest.TestCase):
    def test_cli_binds_and_explicitly_loads_initial_adapter(self) -> None:
        source = (ROOT / "src/f5_tts/train/finetune_cli.py").read_text(encoding="utf-8")
        self.assertIn('"--initial_adapter_dir"', source)
        self.assertIn('"--publish_initial_adapter"', source)
        self.assertIn("initial_adapter_dir=initial_adapter_dir", source)
        self.assertIn("PeftModel.from_pretrained(model, initial_adapter_dir, is_trainable=True)", source)
        self.assertLess(source.index("seed_everything(args.seed)"), source.index("model = CFM("))
        self.assertLess(source.index("build_contract("), source.index("PeftModel.from_pretrained"))


if __name__ == "__main__":
    unittest.main()
