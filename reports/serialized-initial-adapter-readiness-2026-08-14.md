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
adapter. A subsequent bounded GPU experiment exercised that exact path and is
recorded in
[`resume-serialized-initial-live-conditioned-gpu-2026-08-14.md`](resume-serialized-initial-live-conditioned-gpu-2026-08-14.md).
It does not retroactively upgrade earlier evaluator evidence.

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

## Live follow-up and evidence boundary

Companion revision `02eca0556027355077e20c3b0fbe26963d6b8b13` subsequently
published one real initial adapter and conditioned both an uninterrupted and an
interrupted-resumed two-update RTX 3090 Ti process on its exact four-file tree.
Evaluator 0.45 rehashed the tree, observed the interruption after update one,
and found byte-identical final model, optimizer, scheduler, trainer, and RNG
roles. The report passed at
`byte_exact_live_conditioned_artifact_set`.

This qualifies the serialized publication, validation, explicit load route,
and five-role artifact comparison for the exact tested configuration. The
evaluator correctly records `proves_training_semantics: false` and
`proves_model_quality: false`: it cannot independently inspect every trainer
operation or hidden state, and no audio was generated. The result remains
scoped to that dataset, model, seed, dependency stack, update schedule,
interruption point, world size, and host. It does not prove perceptual quality,
adaptation benefit, arbitrary continuation, or distributed equivalence.
