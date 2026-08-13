# F5 LoRA interrupted resume GPU evidence, 2026-08-14

## Result

A two-update uninterrupted F5-TTS LoRA run and an independently interrupted
then resumed run produced byte-identical final adapter weights, canonical
adapter configuration, optimizer and scheduler state, logical training state,
and Python, NumPy, Torch, CUDA, and scaler continuation state on an RTX 3090 Ti.
The two runs also reported the same loss at each update.

The interruption was real: the initial training process group was terminated
immediately after immutable `lora_1` publication. It left no `lora_2`, partial
checkpoint directory, or successful stage receipt. A separate process loaded
that exact numbered checkpoint with explicit trusted-state authority and
completed `lora_2`.

This validates epoch-boundary, world-size-one interruption recovery for one
tiny F5TTS v1 Base LoRA configuration. It does not validate mid-epoch resume,
gradient accumulation, frame batching, sample logging, multiple workers with
stochastic dataset transforms, distributed training, longer training, quality,
or another hardware and software stack.

## Gaps found before the successful comparison

The first live comparison was deliberately treated as a diagnostic rather than
a success based only on matching adapter weights. It found:

1. The CLI fixed data order at seed `666` but did not explicitly seed process
   RNG. The final CLI exposes `--seed`, seeds Python, NumPy, Torch, and CUDA, and
   binds the same value to epoch-addressable data order and the resume contract.
2. The shared `seed_everything` helper did not seed NumPy. The first resumed
   adapter was correct, but the retained NumPy continuation state differed.
3. PEFT serialized its target-module set in nondeterministic list order. The
   adapter configuration was semantically equal but not byte-equal. Checkpoint
   save now sorts the list and writes canonical JSON before hashing.
4. The guarded training CLI could resume, but the executable lifecycle wrapper
   could not authorize a future numbered checkpoint at preflight and route it
   later. The lifecycle now binds one strict `ALLOW_TRAIN_RESUME` bit and accepts
   `TRAIN_RESUME_FROM` only when it resolves to one immutable `lora_N` child of
   the lifecycle-owned output directory.

These were continuation-contract defects. Matching model weights alone would
have hidden the NumPy-state and serialization gaps.

## Bound comparison

- initial seeded implementation revision:
  `0e1e24b57259bffdecad5e8fc4fb2fbeaa0d5265`
- canonical continuation revision:
  `9519f6efe5e2c84d14369f03176a34519145c944`
- lifecycle resume revision:
  `af7b325df7be1d27ed84e4c7639fc0176f2bf4b1`
- upstream base revision:
  `82fc4fe622fe36047d1dff99b550e6018181ea11`
- GPU: NVIDIA GeForce RTX 3090 Ti, 24,564 MiB
- Python: 3.10.19
- PyTorch: 2.9.0+cu128
- CUDA runtime reported by PyTorch: 12.8
- Accelerate: 1.11.0
- PEFT: 0.18.1
- base checkpoint SHA-256:
  `670900fd14e6c458b95da6e9ed317cdb20dbaf7a1c02ac06a05475a9d32b6a38`
- training seed: `20260814`
- epochs: 2
- train rows: 1
- batch type: sample
- batch size: 1
- gradient accumulation: 1
- LoRA rank and alpha: 16 and 16
- learning rate: `1e-4`
- warmup updates: 0
- save interval: every update

The source checkouts were clean before and after both drills.

## Exact continuation result

Update 1 reported loss `0.717` in both the uninterrupted and interrupted
processes. Update 2 reported loss `0.544` in both the uninterrupted and resumed
processes.

At final `lora_2`, these files were byte-identical across the uninterrupted and
resumed output roots:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `README.md` | 5,074 | `514a54fd4623edc63a2156865a2f5c32eb8a32a22caf57c2d08cbc1b70fe4c90` |
| `adapter_model.safetensors` | 11,559,832 | `833b47ca98abb3c30f21faedf1f51e14b76a31ee8cecbe527e817c25337a4c72` |
| `adapter_config.json` | 1,046 | `97390f6c28b420875bbf28526a8d484d672a6a207477348243e1c5ec63a20b52` |
| `training-state.json` | 82 | `45847d3cb16e60b681fe27510dd7a3aad74308a54deb4fa665b9fa3f15e66213` |
| `training_state.pt` | 23,222,457 | `74ed3a5d50fa857b214e9f098d87bfba3cf35120c8ae0ca9ee77b064b6d69615` |
| `runtime-state.pt` | 14,869 | `0a1928935a6ac5d21e410e7ae2a7959ad0336766a2c57689ae074f32da739338` |

The logical state in both final checkpoints was two completed updates, data
epoch two, and zero consumed batches in the next epoch. A semantic load also
confirmed equality of Python, NumPy, Torch, CUDA, and scaler state.

The two `resume-contract.json` files were intentionally not byte-identical.
Each contract binds its own output-directory path and inode so that a checkpoint
cannot be transplanted silently between run roots. All continuation member
identities inside those contracts matched.

## Lifecycle integration result

A separate clean lifecycle drill set `ALLOW_TRAIN_RESUME=1` at preflight,
terminated the lifecycle train process group after `lora_1`, and reran the same
`train` stage with the exact lifecycle-owned checkpoint in
`TRAIN_RESUME_FROM`. The resumed stage produced a passed stage receipt and a
selected-adapter archive.

The lifecycle-resumed final checkpoint was byte-identical to the independent
uninterrupted final checkpoint for all six comparison files in the table above.
Its preflight receipt bound seed `20260814`, two epochs, the resume-authority
bit, data and reference inputs, local vocoder, vocabulary, candidate, and
training controls.

Retained lifecycle evidence includes:

- preflight receipt SHA-256:
  `84a9d89420a33b069c0df4b38cf474cb859e3419156f3070ae45878f235dc323`
- selected-adapter archive SHA-256:
  `ff905dee79ff2212c557c1fcde11485196afa8bd0310d110bdaaf8dbeafe7855`
- resumed-train stage receipt SHA-256:
  `da68bd29172deedfa427124936fcc33b4127a44aa2ec3c0cce88306c12b16e68`

## Retained evidence

- direct comparison root:
  `/mnt/work/chee-wei-jie/voice-models/instavar-f5-resume-equivalence-v2-20260814`
- lifecycle integration root:
  `/mnt/work/chee-wei-jie/voice-models/instavar-f5-lifecycle-resume-20260814`
- uninterrupted log SHA-256:
  `5d19b532e8b76ab7e0840a2b3f957ce0ecf5be0b9807248a71983f00995d9ffc`
- interrupted initial log SHA-256:
  `4991059b8b7e2616294dfa9e4859aa4735ab1c741d1ccc72e5b279f6a3d1ad39`
- resumed log SHA-256:
  `7a6f2178c0f2cbf0066e5bcbf5b23be9014bd4bc7e0930e3e43fce69e9678644`
- lifecycle interrupted log SHA-256:
  `34a8ba4e109b3ba537e33e44417dbdb445085f17d40c0b98abf2bfb6976b2b46`
- lifecycle resumed log SHA-256:
  `804921e8da8ee2074b21fee3ec7a7b07c2f0cae639366769da5858d90227493d`

These hashes make retained-evidence substitution detectable. They do not make
the host trusted, prove behavior outside the tested boundary, or replace a
larger resume matrix.
