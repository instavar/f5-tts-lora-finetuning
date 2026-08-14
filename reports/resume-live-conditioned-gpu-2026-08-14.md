# F5 evaluator 0.45 live-conditioned GPU resume evidence

Date: 2026-08-14, Asia/Singapore

## Result

A fresh uninterrupted two-update LoRA run and an independently interrupted and
resumed run produced byte-identical model, optimizer, scheduler, trainer, and
RNG role files on one RTX 3090 Ti. Instavar Voice evaluator 0.45 rehashed the
four conditioning inputs, both run receipts, the interruption receipt, and all
ten independently stored final-role files. The comparison passed at claim tier
`byte_exact_live_conditioned_artifact_set` with no mismatched roles.

This is stronger than the earlier retained F5 drill because the decomposed
optimizer and scheduler files existed during training and the comparison used
schema 1.1 live-conditioning receipts. It remains one tiny world-size-one,
sample-batched, two-update result. It does not establish training semantics,
runtime loading of declared bytes, quality, or general numerical equivalence.

## Bound revisions and runtime

- companion: `fbea10f5aae769e0f7f01dce5ff1aab69b6ea8bc`
- evaluator: `29c38cfd86b889abc8b79df063c817dd8f684903`
- GPU: NVIDIA GeForce RTX 3090 Ti, 24,564 MiB
- driver: `580.173.02`
- Python: `3.10.19`
- training seed: `20260814`
- epochs and total updates: 2 and 2
- dataset rows: 1
- batch type and size: sample and 1
- gradient accumulation: 1
- LoRA rank and alpha: 16 and 16
- learning rate: `1e-4`
- save interval: every update

The remote companion and evaluator were clean detached checkouts at those exact
revisions. The GPU was idle before and after the drill.

## Interruption evidence

The interrupted process ran in its own session. The harness observed immutable
`lora_1/resume-contract.json`, sent `SIGTERM` to the complete process group, and
waited for process exit before starting another process.

- process exit status: `143`
- checkpoint completed updates: 1
- `lora_2` absent after interrupted-process exit: true
- partial checkpoint absent after interrupted-process exit: true
- update 1 loss in uninterrupted and interrupted processes: `0.717`
- update 2 loss in uninterrupted and resumed processes: `0.544`
- interruption receipt checkpoint-sidecar SHA-256:
  `3a7b6b4639674917bfda0a7ce8d86901de0b2466cb6eceaea15fbea23778a751`

No matching training or Accelerate process remained after the run, and GPU
utilization returned to zero.

## Live conditioning

Both schema 1.1 run receipts fingerprinted the same four preregistered inputs:

- the live Base checkpoint file;
- a dataset-lineage JSON generated from the one-row dataset tree;
- an exact training-controls JSON; and
- an `initial-state.json` deterministic initialization contract binding the
  Base checkpoint, seed, LoRA rank and alpha, and producer revision.

The last item is deliberately limited. It is not a serialized live initial
adapter and does not prove either process loaded the declared bytes. The
comparison report therefore correctly records `proves_training_semantics:
false`. Persisting and explicitly loading one immutable initial adapter would
be required for that stronger conditioning boundary.

## Five-role comparison

| Role | Bytes | SHA-256 | Exact |
| --- | ---: | --- | --- |
| `model_state` | 11,559,832 | `833b47ca98abb3c30f21faedf1f51e14b76a31ee8cecbe527e817c25337a4c72` | yes |
| `optimizer_state` | 23,222,479 | `0e46a180617dbd59d8dca11e5bc084cded82ff7fd521ca52b80bad1a1af1c52e` | yes |
| `scheduler_state` | 1,629 | `b46f311639adca3b7851bd0c69cf55ca35326296186657a4feeac76cda534d25` | yes |
| `trainer_state` | 82 | `45847d3cb16e60b681fe27510dd7a3aad74308a54deb4fa665b9fa3f15e66213` | yes |
| `rng_state` | 14,869 | `0a1928935a6ac5d21e410e7ae2a7959ad0336766a2c57689ae074f32da739338` | yes |

The repository mapper revalidated each file against its checkpoint sidecar
before the evaluator inspected it. The evaluator also confirmed that the two
artifact sets use independent storage and that no compared role aliases a
conditioning input, receipt, or another role.

## Retained evidence

- complete remote checkpoint and evidence root:
  `/mnt/work/chee-wei-jie/voice-models/instavar-f5-resume-live-045-20260814`
- compact remote export:
  `/mnt/work/chee-wei-jie/voice-model-outputs/evaluation/f5-resume-live-045-20260814`
- hash-verified local export:
  `/Users/CheeWeiJie/Downloads/desktop-tailscale-tts/f5-resume-live-045-20260814`
- evaluator report internal SHA-256:
  `cf33833e6df03012ed4f6d160cee3ef04f6ab3bcfc0bfc852e20998ae4f84768`
- exported evaluator report file SHA-256:
  `779ff157c4a9a0e47a0e8e8c450f21b5927c677746f9c51acf2af8b674540618`
- exported run summary file SHA-256:
  `9b9475d278c024d67225d2fbf2336e5ee0f97e18455ad8451f4f4c2d0ee92b0a`

The two exported file hashes matched between the remote export and the local
copy. The compact export retains the receipts, comparison plan, comparison
report, run summary, logs, timing records, and exact harness without duplicating
the large checkpoint trees.

## Remaining gaps

- Persist and explicitly load one immutable initial adapter if a live initial
  model-state boundary is required.
- Exercise frame batching, gradient accumulation, sample logging, stochastic
  workers, mid-epoch interruption, longer runs, and distributed training.
- Repeat on another dependency and accelerator stack before generalizing
  determinism.
- Run matched Base-versus-adapted objective evaluation and blind listening for
  quality claims.
- Python 3.10 emitted a dependency end-of-support warning for October 2026; a
  Python 3.11 environment needs a separate compatibility qualification.
