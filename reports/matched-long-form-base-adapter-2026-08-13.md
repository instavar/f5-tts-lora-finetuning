# Matched long-form base and adapter evidence, 2026-08-13

## Result

The unchanged F5TTS Base checkpoint and the selected FEMALE_01 LoRA step 1250
adapter each generated the frozen `cadence-two-minute` prompt at seed
`20260812`. Both paths used the same clean runner checkout, base checkpoint,
reference recording, reference transcript, chunking code, device, and decoding
settings. Adapter mode differed only by loading and merging the declared LoRA
artifact set.

This is one prompt and one seed. It is matched execution and evaluation
evidence, not evidence that the adapter improves voice quality.

## Bound inputs

- F5 runner revision: `01fd17f3b963e7b75d43ae3036a670b5f98c3fc8`
- evaluator revision used for final scoring: `982367abc7837cb6da5ebb94192c9642dea62fce`
- base checkpoint SHA-256: `4180310f91d592cee4bc14998cd37c781f779cf105e8ca8744d9bd48ca7046ae`
- LoRA adapter SHA-256: `16862585c142d1b07837484e7ccf0f318cdcea854f99a131c074064836391705`
- base artifact-set SHA-256: `bfc0f7fa41fb4e60be8c24ef35d1644fae6a30e1074188584e0891ea4048e133`
- adapter plus base artifact-set SHA-256: `46f11fa732307352169b3bafb02d879fd14306bca86ea1c84e3b9d16f9214096`
- generation-plan file SHA-256: `ea6bc9e25206c8d559779c68b65c606b19488db4a7efad837ccfdd51222c895c`
- canonical generation-plan SHA-256: `b91ea43c42f4194557d3930a845adc640d13788f765267577d73ab6f7392e618`
- prompt-pack canonical SHA-256: `6d6750188abd6b8db83527158bf689ee138c65167a36ede17c62013bdc1279b1`
- remote evidence root: `/mnt/work/chee-wei-jie/voice-model-outputs/conformance/20260813_f5_matched_long_form_v1`

The F5 checkout was clean before and after generation. The older dirty training
checkout supplied retained weights and the reference recording but was not used
as the runner checkout.

## Generated audio and runtime

| Candidate | WAV SHA-256 | Duration | Warm generation | RTF | Peak allocated CUDA memory |
| --- | --- | ---: | ---: | ---: | ---: |
| base | `0163d1958af73a634ad5a3c72cb0bfed71a973bd7965ca24a6bc2015e19335f6` | 68.4667 s | 7.5070 s | 0.10964 | 843,938,304 bytes |
| LoRA step 1250 | `3c113645531eb16e46633e6e309a247a464612555978b4d589d3e89860f90703` | 68.4667 s | 7.0914 s | 0.10357 | 852,363,776 bytes |

Both rows were valid 24 kHz mono PCM WAVs and used five sequential inference
chunks. Warm timing excludes model and vocoder loading. One run per candidate
cannot support a throughput or memory-efficiency conclusion.

## Objective evidence

The complete matched comparison passed all nine metrics required by prompt pack
1.2.0. The results used content-bound runtime attempts, PCM probes, local
faster-whisper `large-v3-turbo`, and SpeechBrain ECAPA against the exact retained
reference recording.

| Proxy | Base | LoRA step 1250 | LoRA minus base |
| --- | ---: | ---: | ---: |
| ASR word error rate | 0.0 | 0.0 | 0.0 |
| ECAPA cosine similarity | 0.796893 | 0.795741 | -0.001151 |
| Silence fraction | 0.303542 | 0.303341 | -0.000201 |
| Clipping fraction | 0.00009494 | 0.00009737 | 0.00000243 |
| Real-time factor | 0.10964 | 0.10357 | -0.00607 |

The ASR extractor revision was
`0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf`, with artifact-set SHA-256
`3433b5ac25f4b005aadfcde370f3615a5d2883fe40d251e823b80204071115d6`.
The ECAPA extractor revision was
`0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`, with artifact-set SHA-256
`5a8cd13222e7edf1c932b8695e34c6537c15230e8e47aabe9af454284906dd7c`.

The single speaker reference was fixed before speaker scoring but after audio
generation. The assignment rationale records that chronology. It therefore
does not establish pre-generation preregistration. It uses the exact reference
recording already shared by both generation paths, so it does establish a
symmetric one-reference comparison for this slice.

## Prosody proxy and listening boundary

Both WAVs exceeded the 30-second proxy eligibility threshold. The matched
prosody report contains one complete pair, no invalid outputs, no extractor
failures, no winner, and `proves_adaptation_benefit: false`. Its signed LoRA
minus base deltas include:

- phrase-duration coefficient of variation: `-0.023666`;
- pause-duration coefficient of variation: `-0.001556`;
- active RMS dB standard deviation: `+0.035294`;
- two-second window RMS dB standard deviation: `-0.002748`; and
- zero-crossing-rate standard deviation: `+99.9891 Hz`.

No direction is established for any prosody proxy. These differences do not
mean that cadence, monotony, naturalness, accent, or fatigue improved or
regressed.

A two-sample identity-neutral blind pack was generated. It assigns speaker
identity, cadence variation, long-form monotony, naturalness, artifact severity,
and listening fatigue. Because this is a focused cadence plan, assignment schema
1.3 explicitly excludes Singapore English accent fidelity, lexical
pronunciation, and emotion obedience. The private reveal mapping remains
separate. No ratings were invented or inferred.

## Evidence document hashes

- generation observations: `cd4b9883863d915b3f4c4fa68909f6a1bd8660301b25f7b94338308b0b9eae57`
- suite coverage: `d066827de279cb545c1271c1c44e1cf8a15c6e529d95d61a3a4562ec52b7809d`
- generation-attempt receipt: `77319df40056b610dca89c1b2ce6989ee15174c3931a18837b5568bc7d8b00e3`
- objective report: `724796c9fd3515cf3f9569a4c31245d1e249adf9dfb4c0c73671e49f6bce15d1`
- matched objective comparison: `88d0d73aba33a1c93cd650d22bb1fb3a17e8833050ba6860e625c2f653f12788`
- matched prosody comparison: `67f406f557fba128fb5577b0a4c6c2a4d416b7dbb42849e0220a830ce048a67a`
- focused listening assignment: `46945046258d0789e3a7710c8a28246e11e0876978b1d8c00642e7d265e8b09b`
- blind review: `1e293ce19f25c521c817f387e85760f41dd850a9fbd7501c414d4b5f9a9b4859`
- blind audio manifest: `637ab45f3a53639c227c0728b245a9c457c2764c28d977afa5684ea011bd87f3`

These hashes make substitution detectable against the retained host evidence.
They do not make the host trusted, prove that extractors are scientifically
valid, or replace independent reproduction and human review.
