# Serialized initial F5 LoRA adapter readiness

Date: 2026-08-14, Asia/Singapore

## Finding

The F5 companion can now publish one seeded initial PEFT adapter before paired
training conditions begin. Publication uses the same model construction and
base-checkpoint loader as training, writes safe serialized adapter weights to a
temporary sibling, records a byte-hashed receipt, fsyncs the files and
directory, and renames without overwrite.

Fresh and resumed training processes can load the exact directory through
`--initial_adapter_dir`. Before model loading, the companion verifies the live
file tree against the receipt, rejects unsafe links, validates the PEFT LoRA
configuration against the requested rank, alpha, dropout, and target modules,
and binds every file into the guarded resume contract. The executable lifecycle
also requires `TRAIN_INITIAL_ADAPTER_DIR`, hashes the tree at preflight, and
forwards it to the trainer.

This closes the repository implementation gap between a deterministic
initialization declaration and an explicitly serialized and loaded initial
adapter. It does not retroactively upgrade earlier evaluator evidence.

## OOD controls

Dependency-free tests cover:

- no-overwrite atomic publication;
- invalid producer revision and output path handling;
- missing or unsafe serialized weights;
- live file and receipt byte drift;
- receipt metadata and LoRA configuration drift;
- symlink and cross-file hardlink rejection;
- mismatched requested LoRA rank and target configuration;
- seed-before-model-construction ordering;
- explicit `PeftModel.from_pretrained(..., is_trainable=True)` loading;
- resume-contract inclusion of the initial adapter; and
- lifecycle preflight drift when initial adapter bytes change.

The complete dependency-free repository suite passed 57 tests. The capability,
historical-evidence, lifecycle-backend, Docker-source, compile, and formatting
checks also passed locally.

## Evidence boundary

No real model was initialized or trained for this implementation result. PEFT
loading, model compatibility, CUDA behavior, interruption handling, and final
artifact equality remain unverified for the new path.

The next qualifying experiment must create the initial adapter in one process,
condition both an uninterrupted and an interrupted-resumed process on those
same bytes, and preserve evaluator 0.45 schema 1.1 receipts. The interruption
must be observed before the target update, and all five independently stored
final artifact roles must be compared. A passing result would remain scoped to
that dataset, model, seed, dependency stack, update schedule, interruption
point, world size, and host. It would not prove perceptual quality, adaptation
benefit, arbitrary continuation, or distributed equivalence.
