# F5 serialized-initial live-conditioned GPU resume evidence

Date: 2026-08-14, Asia/Singapore

## Result

One process published a seeded PEFT initial adapter through the repository's
no-overwrite publication path. A fresh uninterrupted two-update LoRA process
and an independently interrupted and resumed process then used the same exact
four-file adapter tree. On one RTX 3090 Ti, Instavar Voice evaluator 0.45
rehashed all conditioning inputs, the interruption receipt, both run receipts,
and ten independently stored final-role files.

The comparison passed at claim tier
`byte_exact_live_conditioned_artifact_set`. Model, optimizer, scheduler,
trainer, and RNG roles were byte-identical with no mismatches. This is bounded
evidence for the repository's serialized initial-adapter route and the exact
tested resume configuration. It is not evidence of perceptual quality,
adaptation benefit, arbitrary continuation, or distributed equivalence.

## Bound revisions and runtime

- companion: `02eca0556027355077e20c3b0fbe26963d6b8b13`
- evaluator: `29c38cfd86b889abc8b79df063c817dd8f684903`
- GPU: NVIDIA GeForce RTX 3090 Ti, 24,564 MiB
- driver: `580.173.02`
- Python: `3.10.19`
- training seed: `20260814`
- total updates: 2
- dataset rows: 1
- batch type and size: sample and 1
- gradient accumulation: 1
- LoRA rank and alpha: 16 and 16
- learning rate: `1e-4`
- save interval: every update

The companion and evaluator were clean detached checkouts at the bound
revisions. The complete run used the same installed F5 training environment as
the prior evaluator 0.45 drill.

## Serialized initial adapter

The publication command used the real F5 model construction and base-checkpoint
loader, then exited before training. Its receipt bound the producer revision,
absolute Base checkpoint path, seed, LoRA configuration, and exact live files.

| Initial-adapter file | Bytes | SHA-256 |
| --- | ---: | --- |
| `README.md` | 5,074 | `514a54fd4623edc63a2156865a2f5c32eb8a32a22caf57c2d08cbc1b70fe4c90` |
| `adapter_config.json` | 1,045 | `8006de2111280d51f5e966328fa9ea46087a4e60b60fed8734eb92e05165c2c7` |
| `adapter_model.safetensors` | 11,559,832 | `b3c6e8df0b0a04aa272c6f8a8b5bffb17780b3f2b05f592b357d1857ee74ea55` |
| `initial-adapter-receipt.json` | 1,144 | `68467dd1ac5dbb872e3cbadd5e7b123e2abd5fd7f57a66793e2a2ce050e1c11d` |

Evaluator 0.45 represented that exact four-file tree as the `initial_state`
conditioning role with SHA-256
`fe4b162aa3a82cb7c471e5436bae2188b7e2eae3f5e76a5e181f146c0883b27d`.
Both run receipts declared the same conditioning identity, and each guarded
resume contract bound every file in the initial-adapter directory.

## Interruption evidence

The interrupted process ran in its own process group. The harness observed the
immutable update-one checkpoint, sent `SIGTERM` to the process group, waited for
exit status `143`, and verified that neither the update-two checkpoint nor a
partial checkpoint existed. A separate process then resumed from update one.

- update one loss in uninterrupted and interrupted processes: `0.717`
- update two loss in uninterrupted and resumed processes: `0.544`
- update-one checkpoint sidecar SHA-256:
  `05f2aa96b1a11944b53c086155f816679cdeb9b28c594a328ca2c653feb51810`
- update-one trainer state SHA-256:
  `830d717c784418902efeb77f40aad6d46c031a7c445ed866f5273a681abe47a3`
- interruption receipt SHA-256:
  `4c1e40fc9e92dd8087c186427236b4a1d416ac0b03cde137be06669405fb7b01`

## Five-role comparison

| Role | Bytes | SHA-256 | Exact |
| --- | ---: | --- | --- |
| `model_state` | 11,559,832 | `833b47ca98abb3c30f21faedf1f51e14b76a31ee8cecbe527e817c25337a4c72` | yes |
| `optimizer_state` | 23,222,479 | `0e46a180617dbd59d8dca11e5bc084cded82ff7fd521ca52b80bad1a1af1c52e` | yes |
| `scheduler_state` | 1,629 | `b46f311639adca3b7851bd0c69cf55ca35326296186657a4feeac76cda534d25` | yes |
| `trainer_state` | 82 | `45847d3cb16e60b681fe27510dd7a3aad74308a54deb4fa665b9fa3f15e66213` | yes |
| `rng_state` | 14,869 | `0a1928935a6ac5d21e410e7ae2a7959ad0336766a2c57689ae074f32da739338` | yes |

The repository mapper revalidated each role against its checkpoint sidecar.
The evaluator confirmed independent storage and no aliasing between compared
roles, conditioning inputs, or receipts.

## Retained evidence

- complete remote checkpoint and evidence root:
  `/mnt/work/chee-wei-jie/voice-models/instavar-f5-resume-serialized-live-045-20260814`
- compact remote export:
  `/mnt/work/chee-wei-jie/voice-model-outputs/evaluation/f5-resume-serialized-live-045-20260814`
- hash-verified local export:
  `/Users/CheeWeiJie/Downloads/desktop-tailscale-tts/f5-resume-serialized-live-045-20260814`
- compact manifest SHA-256:
  `8250cfde17fd3989e5c0a12c763b51602d638da031f8a654337a902c39e9843b`
- evaluator report internal SHA-256:
  `aac25fc5fb9267481321dc7758cc7b5fe97a55cc848d646a42cf4918b771e0fa`
- exported evaluator report file SHA-256:
  `65682c1891c7f1c1cae88758808e114a187c07901177296ddff8310d449ad930`
- exported run summary file SHA-256:
  `a01c38787b9177f59267f2a0ebf50d0c4dccce34ec29ac0dabd96f058c47838c`

All 37 manifest entries verified in the remote export and the local copy. The
manifest itself had the same hash in both locations.

## Evidence boundary and remaining gaps

The evaluator records `proves_training_semantics: false`,
`proves_numerical_resume_equivalence: false`, and `proves_model_quality:
false`. It rehashes declared live inputs and final artifacts but cannot prove a
trainer honored every declared byte, that no hidden state exists, or that
floating-point trajectories are generally equivalent. No audio was generated.

Remaining work includes frame batching, accumulated gradients, sample logging,
stochastic workers, mid-epoch interruption, longer runs, another dependency or
accelerator stack, and distributed training. Quality evaluation still requires
matched Base-versus-adapted outputs, objective checks, multi-seed coverage, and
blind listening.
