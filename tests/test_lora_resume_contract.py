import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from f5_tts.train.lora_resume_contract import (  # noqa: E402
    RUNTIME_STATE_NAME,
    STATE_NAME,
    ResumeContractError,
    build_contract,
    prunable_checkpoints,
    require_fresh_output,
    validate_checkpoint,
    write_sidecar,
)


class LoRAResumeContractTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "output"
        self.output.mkdir()
        self.base = self.root / "base.safetensors"
        self.base.write_bytes(b"base")
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        (self.dataset / "data.arrow").write_bytes(b"dataset")
        self.source = self.root / "trainer.py"
        self.source.write_text("print('trainer')\n", encoding="utf-8")
        self.contract = build_contract(
            output_dir=self.output,
            base_checkpoint=self.base,
            dataset_root=self.dataset,
            optional_files={"tokenizer": None},
            source_files=[self.source],
            training_config={"epochs": 3, "batch_size_type": "frame"},
            runtime={"python": "3.test", "world_size": 1},
        )

    def tearDown(self):
        self.temporary.cleanup()

    def checkpoint(self, completed_updates=4, data_epoch=1):
        checkpoint = self.output / f"lora_{completed_updates}"
        checkpoint.mkdir()
        for name, payload in {
            "adapter_config.json": b"{}\n",
            "adapter_model.safetensors": b"adapter",
            "training_state.pt": b"optimizer",
            RUNTIME_STATE_NAME: b"runtime",
        }.items():
            (checkpoint / name).write_bytes(payload)
        (checkpoint / STATE_NAME).write_text(
            json.dumps(
                {
                    "completed_updates": completed_updates,
                    "data_epoch": data_epoch,
                    "batches_consumed_in_epoch": 2,
                }
            ),
            encoding="utf-8",
        )
        write_sidecar(
            checkpoint,
            contract=self.contract,
            completed_updates=completed_updates,
            required_files=[
                "adapter_config.json",
                "adapter_model.safetensors",
                "training_state.pt",
                STATE_NAME,
                RUNTIME_STATE_NAME,
            ],
        )
        return checkpoint

    def test_valid_checkpoint_round_trip(self):
        checkpoint = self.checkpoint()
        resolved, state, _ = validate_checkpoint(
            checkpoint,
            output_dir=self.output,
            expected_contract=self.contract,
            trust_resume_state=True,
            world_size=1,
        )
        self.assertEqual(resolved, checkpoint.resolve())
        self.assertEqual(state["completed_updates"], 4)

    def test_resume_requires_explicit_pickle_trust(self):
        with self.assertRaisesRegex(ResumeContractError, "trust_resume_state"):
            validate_checkpoint(
                self.checkpoint(),
                output_dir=self.output,
                expected_contract=self.contract,
                trust_resume_state=False,
                world_size=1,
            )

    def test_distributed_resume_fails_closed(self):
        with self.assertRaisesRegex(ResumeContractError, "world_size=1"):
            validate_checkpoint(
                self.checkpoint(),
                output_dir=self.output,
                expected_contract=self.contract,
                trust_resume_state=True,
                world_size=2,
            )

    def test_mutated_adapter_is_rejected(self):
        checkpoint = self.checkpoint()
        (checkpoint / "adapter_model.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(ResumeContractError, "drift"):
            validate_checkpoint(
                checkpoint,
                output_dir=self.output,
                expected_contract=self.contract,
                trust_resume_state=True,
                world_size=1,
            )

    def test_dataset_drift_changes_contract(self):
        checkpoint = self.checkpoint()
        (self.dataset / "data.arrow").write_bytes(b"changed")
        changed = build_contract(
            output_dir=self.output,
            base_checkpoint=self.base,
            dataset_root=self.dataset,
            optional_files={},
            source_files=[self.source],
            training_config={"epochs": 3, "batch_size_type": "frame"},
            runtime={"python": "3.test", "world_size": 1},
        )
        with self.assertRaisesRegex(ResumeContractError, "contract drift"):
            validate_checkpoint(
                checkpoint,
                output_dir=self.output,
                expected_contract=changed,
                trust_resume_state=True,
                world_size=1,
            )

    def test_latest_nested_symlink_and_partial_outputs_are_rejected(self):
        checkpoint = self.checkpoint()
        for unsafe in (self.output / "lora_last", self.output / "nested" / checkpoint.name):
            unsafe.parent.mkdir(parents=True, exist_ok=True)
            unsafe.mkdir(exist_ok=True)
            with self.assertRaises(ResumeContractError):
                validate_checkpoint(
                    unsafe,
                    output_dir=self.output,
                    expected_contract=self.contract,
                    trust_resume_state=True,
                    world_size=1,
                )
        alias = self.output / "lora_8"
        alias.symlink_to(checkpoint, target_is_directory=True)
        with self.assertRaisesRegex(ResumeContractError, "symlink"):
            validate_checkpoint(
                alias,
                output_dir=self.output,
                expected_contract=self.contract,
                trust_resume_state=True,
                world_size=1,
            )
        partial_output = self.root / "partial-output"
        partial_output.mkdir()
        (partial_output / ".lora_9.partial.1").mkdir()
        with self.assertRaisesRegex(ResumeContractError, "fresh directory"):
            require_fresh_output(partial_output)

    def test_completed_target_is_rejected(self):
        checkpoint = self.checkpoint(data_epoch=3)
        with self.assertRaisesRegex(ResumeContractError, "reached"):
            validate_checkpoint(
                checkpoint,
                output_dir=self.output,
                expected_contract=self.contract,
                trust_resume_state=True,
                world_size=1,
            )

    def test_pruning_uses_exact_direct_numeric_directories(self):
        for update in (2, 10, 5):
            checkpoint = self.output / f"lora_{update}"
            checkpoint.mkdir()
            (checkpoint / "resume-contract.json").write_text("{}\n", encoding="utf-8")
        (self.output / "lora_evil").mkdir()
        (self.output / "lora_last").mkdir()
        self.assertEqual(
            [path.name for path in prunable_checkpoints(self.output, keep_last=1)],
            ["lora_2", "lora_5"],
        )

    def test_pruning_rejects_symbolic_numeric_checkpoint(self):
        target = self.root / "external"
        target.mkdir()
        (self.output / "lora_2").symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ResumeContractError, "symbolic"):
            prunable_checkpoints(self.output, keep_last=1)

    def test_pruning_rejects_unowned_numeric_checkpoint(self):
        (self.output / "lora_2").mkdir()
        with self.assertRaisesRegex(ResumeContractError, "unowned"):
            prunable_checkpoints(self.output, keep_last=1)


class LoRATrainerSourceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trainer = (ROOT / "src/f5_tts/model/trainer.py").read_text(encoding="utf-8")
        cls.cli = (ROOT / "src/f5_tts/train/finetune_cli.py").read_text(encoding="utf-8")
        cls.dataset = (ROOT / "src/f5_tts/model/dataset.py").read_text(encoding="utf-8")

    def test_cli_requires_explicit_checkpoint_and_trust(self):
        self.assertIn('"--resume_from"', self.cli)
        self.assertIn('"--trust_resume_state"', self.cli)
        self.assertIn("trust_resume_state requires an explicit resume_from", self.cli)

    def test_cli_binds_one_seed_to_process_rng_data_order_and_resume_contract(self):
        self.assertIn('"--seed"', self.cli)
        self.assertIn("seed_everything(args.seed)", self.cli)
        self.assertIn('"seed": args.seed', self.cli)
        self.assertIn('"shuffle_seed": args.seed', self.cli)
        self.assertIn("resumable_with_seed=args.seed", self.cli)
        self.assertLess(self.cli.index("seed_everything(args.seed)"), self.cli.index("model = CFM("))

    def test_lora_loader_never_selects_last_or_highest_checkpoint(self):
        lora_start = self.trainer.index("if self.use_lora:", self.trainer.index("def load_checkpoint"))
        standard_start = self.trainer.index("# Standard (non-LoRA) checkpoint loading", lora_start)
        lora_loader = self.trainer[lora_start:standard_start]
        self.assertIn("validate_checkpoint(", lora_loader)
        self.assertNotIn("lora_last", lora_loader)
        self.assertNotIn("sorted(", lora_loader)

    def test_runtime_state_restores_after_skip_wrapper_is_created(self):
        skip_index = self.trainer.index("skip_first_batches(")
        restore_index = self.trainer.index("random.setstate(restore_runtime_state")
        self.assertLess(skip_index, restore_index)

    def test_sample_logging_happens_before_rng_checkpoint(self):
        loop_start = self.trainer.index("save_last = global_update")
        final_save = self.trainer.index("self.save_checkpoint(global_update, last=save_last)", loop_start)
        sample_block = self.trainer.index("if save_numbered and self.log_samples", loop_start)
        self.assertLess(sample_block, final_save)

    def test_sample_sampler_is_epoch_addressable(self):
        self.assertIn("class EpochRandomSampler", self.dataset)
        self.assertIn("self.random_seed + self.epoch", self.dataset)
        self.assertIn("sampler = EpochRandomSampler", self.trainer)

    def test_lora_checkpoint_is_atomic_and_immutable(self):
        self.assertIn("Refusing to reuse unowned immutable LoRA checkpoint", self.trainer)
        self.assertIn('adapter_config["target_modules"] = sorted', self.trainer)
        self.assertIn("os.replace(lora_dir, final_dir)", self.trainer)
        self.assertIn("os.symlink(os.path.basename(final_dir)", self.trainer)

    def test_shared_seed_helper_covers_numpy(self):
        utils = (ROOT / "src/f5_tts/model/utils.py").read_text(encoding="utf-8")
        self.assertIn("np.random.seed(seed % (2**32))", utils)


if __name__ == "__main__":
    unittest.main()
