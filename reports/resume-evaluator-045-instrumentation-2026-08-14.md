# Evaluator 0.45 resume instrumentation

Date: 2026-08-14, Asia/Singapore

## Change

Future F5-TTS LoRA checkpoints now expose optimizer and scheduler state as
separate `optimizer-state.pt` and `scheduler-state.pt` files. The trainer still
writes the combined `training_state.pt` used by the guarded loader, so older
checkpoints remain compatible. The checkpoint sidecar hashes the two new files
before the partial directory is atomically published as `lora_N`.

The five evaluator 0.45 final-state roles now map as follows:

| Evaluator role | F5-TTS checkpoint member |
| --- | --- |
| `model_state` | `adapter_model.safetensors` |
| `optimizer_state` | `optimizer-state.pt` |
| `scheduler_state` | `scheduler-state.pt` |
| `trainer_state` | `training-state.json` |
| `rng_state` | `runtime-state.pt` |

`adapter_config.json` and the combined `training_state.pt` remain sidecar-bound
additional evidence. The runtime loader continues to read the combined file;
the decomposed copies are for cross-run evidence and inspection.

`evaluator_lora_artifact_paths(...)` now rehashes each mapped live file against
the published sidecar and rejects legacy incomplete evidence, terminal
symlinks, unsafe members, and cross-role hardlinks before returning paths to the
shared evaluator.

## OOD and compatibility controls

Dependency-free tests cover:

- a new checkpoint whose sidecar binds both decomposed files;
- mutation of `optimizer-state.pt` after publication;
- a legacy checkpoint without the decomposed files that still passes the
  original guarded loader contract; and
- source-level checks that the trainer writes both decomposed files before the
  atomic directory publication.
- exact five-role mapping plus rejection of legacy, mutated, and hardlinked
  evaluator inputs.

The focused guarded-resume suite passed 20 tests locally with Python 3.11.
After adding the fail-closed mapper, full dependency-free discovery passed 50
tests locally.
The public contract workflow pins evaluator 0.45 revision
`29c38cfd86b889abc8b79df063c817dd8f684903` and verifies that its live
conditioning receipt and comparison APIs are present.

## Evidence boundary

No new GPU training was run for this instrumentation change. It establishes
repository behavior and dependency-free contract coverage only. It does not
show that two newly serialized optimizer or scheduler files are byte-identical
across real runs, and it does not upgrade the retained 2026-08-14 GPU drill.

A fresh stronger comparison must create schema 1.1 receipts from live,
preregistered conditioning artifacts before inspecting the outcome. In
particular, the run needs a content-bound Base artifact, dataset-lineage
receipt, training-controls artifact, and initial-state artifact shared by the
uninterrupted and interrupted-resumed conditions. The evaluator must then
rehash those inputs and the five independently stored final-state roles.

The new files add storage because `training_state.pt` remains for compatibility.
That tradeoff avoids silently breaking existing resume authorities. A future
checkpoint schema may remove the combined duplication only through an explicit
migration with backward-compatible loading tests.
