# F5 LoRA package and restore GPU evidence, 2026-08-14

## Result

A clean F5-TTS companion checkout completed preflight, one-update LoRA
training, fresh-process inference, the frozen evaluator, content-addressed
packaging, and two independent restores on an RTX 3090 Ti. The original smoke
WAV and both restored smoke WAVs were byte-identical under package-bound seed
`42`.

Live negative probes changed the asserted seed and smoke text independently.
Both restores failed before publishing a destination. Dependency-free tests
also cover reference-audio drift, reference-transcript drift, base, vocabulary,
vocoder, archive, and package-member failures.

This qualifies the tested PyTorch CUDA LoRA package and restore path. It does
not qualify remote backup, recovery on a second host, inherited Triton, MLX, or
ONNX runtimes, perceptual quality, distribution rights, or production use.

## Why the contract changed

The first package drill found three lifecycle gaps in sequence:

1. Training launched a bare `accelerate` executable and failed when that script
   was absent from `PATH`, even though Accelerate was installed in the active
   Python environment. The lifecycle now launches
   `python -m accelerate.commands.launch`.
2. Fresh inference resolved Vocos from the Hugging Face Hub. The lifecycle now
   fingerprints a supplied local vocoder tree and forces local loading during
   inference, evaluation, and restore.
3. The first original and restored WAVs had the same shape and duration but
   different hashes because the smoke inference had no explicit seed. Package
   schema 1.2 now binds the authorised reference audio, reference-text hash,
   smoke-text hash, and seed, and passes the same seed through original and
   restored inference.

These were contract defects, not evidence that the model or adapter was bad.
The progression matters: an adapter archive alone is not a reproducible model
package when separately resolved runtime dependencies and generation controls
remain mutable.

## Bound execution

- companion revision: `1816042223d4e84c8bd174a59a18aa8fcdd9e601`
- upstream base revision: `82fc4fe622fe36047d1dff99b550e6018181ea11`
- evaluator revision: `2812e200233804fde685c35ea1da1cbf9fe8ef4b`
- GPU: NVIDIA GeForce RTX 3090 Ti, 24,564 MiB
- Python: 3.10.19
- PyTorch: 2.9.0+cu128
- CUDA runtime reported by PyTorch: 12.8
- Accelerate: 1.11.0
- PEFT: 0.18.1
- base checkpoint SHA-256:
  `670900fd14e6c458b95da6e9ed317cdb20dbaf7a1c02ac06a05475a9d32b6a38`
- vocabulary SHA-256:
  `2a05f992e00af9b0bd3800a8d23e78d520dbd705284ed2eedb5f4bd29398fa3c`
- vocoder tree SHA-256:
  `25102ee6232af26dc0a7c994b2ec20a3a5e201fb7cf81b61de53c12c04e6d316`
- reference-audio SHA-256:
  `b474ad28383240a64a965fb9a1ee7e49f5c4ccc4c05b20cc9a8de80edbcb7bb8`
- reference-text SHA-256:
  `4b063d5f7c618273e574e077796143ef54c40dea72d6df720d9ca5aa26991b8b`
- smoke-text SHA-256:
  `9759b1903791430bb4d624cd7341081b83ff0f790010bed9edc3f27b11d95f21`
- smoke seed: `42`
- retained host evidence root:
  `/mnt/work/chee-wei-jie/voice-models/instavar-f5-package-restore-seeded-20260814`

The companion checkout was clean before and after the drill. Raw audio,
reference text, model weights, adapter weights, and generated audio remain on
the private host and are not included in this repository.

## Training and fresh evaluation

The bounded train split contained one authorised row. One epoch produced one
optimizer update with loss `0.763`. LoRA exposed 2,883,584 trainable parameters
out of 339,980,388 total parameters. Training took 11.74 seconds wall time and
reached 3,723,080 KiB maximum host RSS.

The selected adapter artifact identities were:

- adapter model SHA-256:
  `ee0a8d014417a03bca61593b7333168956b44fcdc736528273b8260729b7ec5e`
- adapter config SHA-256:
  `09f63ef5d5d6ed59ee3b57e25bc0fbe7b7937072526b34e191f734806dfbfc9c`
- resume contract SHA-256:
  `7cea19789eb66e2684b1c5041b6198074269ee960d38e9b3bcd09c8c9183b9ac`
- runtime state SHA-256:
  `a04a40c713de940c404f87edf9c5d599002720cb2066d6d881ad4d4a5d66ecfb`
- selected adapter archive SHA-256:
  `9c231fba0e60f3eda6243aca64c8a87a4e0a768a83667dbde6e6cb60aeb58a3b`

The frozen evaluator generated its one planned held-out row successfully:

| Evidence | Value |
| --- | ---: |
| Valid rows | 1 of 1 |
| Audio duration | 5.7067 s |
| Generation time | 1.2353 s |
| Real-time factor | 0.2165 |
| Peak allocated CUDA memory | 798,397,440 bytes |
| Inference chunks | 1 |
| Evaluated WAV SHA-256 | `30da21db195ab72635ecc9725ca20d45d81d785cbffa4443840ac90737c44997` |

This row establishes executable generation and plan-bound runtime evidence. No
ASR, speaker embedding, prosody proxy, or blind listening result was produced
for this bounded package drill, so it does not support a content, identity,
cadence, accent, or naturalness claim.

## Package and restore

The lifecycle published a 35,379,200-byte package with SHA-256
`75de6c2da94f0fcfc4545b613fa79ad814969cc88dbe5229ecf593780b0160c7`.
Its manifest SHA-256 was
`62e57cfbe439f5e3b2a44ed7c15f736ce76a29733c5cc678697af28e9d56ad85`.
The archive contained exactly the package root, seven declared files, and its
manifest. No base checkpoint, reference audio, vocabulary, or vocoder bytes
were copied into the package; their external content identities were bound in
the manifest and supplied independently at restore time.

Both restores used fresh Python processes and local Vocos loading. Each took
about 11.9 seconds. The original and two restored WAVs all had:

- SHA-256:
  `b38aecc98257c41bfa430a95e4b5b5e0a8c5417cd7badd5953ad406eb8bfa5db`
- 224,812 bytes
- 112,384 frames
- 24,000 Hz
- one channel
- 16-bit samples

Exact byte equality is narrow but useful evidence. It establishes deterministic
repeatability for this model, package, prompt, seed, dependency set, software
revision, and host path. It does not imply determinism across devices, CUDA or
PyTorch versions, runtime implementations, arbitrary prompts, or longer
multi-chunk generation.

## OOD and fail-closed probes

The final implementation passed 41 dependency-free tests. The test matrix
includes malformed and out-of-range seeds, changed training controls, changed
generation controls, wrong base checkpoint, wrong or missing vocabulary, wrong
vocoder tree, wrong reference audio, changed reference transcript, changed
smoke text, wrong seed assertion, extra package files, member drift, corrupt
inner adapter archives, traversal, links, duplicate members, unsafe output
paths, persistence-root drift, and failed inference.

Two contract changes were also tested on the live GPU host:

| Probe | Exit | Destination published |
| --- | ---: | --- |
| Assert seed `43` against package seed `42` | 1 | No |
| Supply a different smoke text | 1 | No |

This is not an exhaustive adversarial security audit. The package is an
integrity and reproducibility envelope, not a signature, trust root, encrypted
backup, or sandbox for arbitrary untrusted model code.

## Evidence document hashes

- dataset lineage:
  `baa032ca58b8a331a9cafdd0509492ff97b549e4f4d1e24f3b2756dd00672d90`
- experiment manifest:
  `0322611154a9ccc8922c62f71703891922dda244f55f0abe348e97b9bb143014`
- generation plan:
  `abc06241fe7cb4553edfa67cde9ee22f8123ab88649535dbf065cdb11c90e333`
- preflight receipt:
  `bdc7123197d1d211f67a932010a064dc1819c76e04cb09b050f0017ea45fc3b6`
- generation observations:
  `c374d27702aa39ae9f127317c18e2eab18b9f95365941fa297d9803c82c6e55f`
- generation-attempt receipt:
  `d8d6456ea948846ada1e9b6fe4ad859f3ec0e660e97b9e3bba1dd2521a5ae5ba`
- bound objective observations:
  `f20f0bb8eae2bec8cc8493fa4a0cac24c4b68a8d292407c320cbbff46e505428`
- evaluation bundle:
  `cada4c1667f715e1c56176858bea734e7d039481a4706c8b73122473808896b0`
- persisted-package receipt:
  `25cdcf2c78a4c50526350b9315d54e997e195730f5fb71a16ccd35e8e7b73e88`

These hashes make substitution detectable against the retained private-host
artifacts. They do not make that host trusted or replace independent
reproduction.
