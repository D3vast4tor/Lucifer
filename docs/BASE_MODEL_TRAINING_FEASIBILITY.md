# Training Lucifer's installed base: hardware assessment and implementation plan

Assessed 13 November 2026. This is an assessment and a proposed implementation,
not an installed trainer or a completed training experiment.

This document preserves the original assessment. The subsequent implementation,
two training modes and current validation status are documented in
[Meditation](MEDITATION.md) and its [implementation plan](MEDITATION_PLAN.md).

## Decision

**Full-parameter training of the installed primary Lucifer model is not a
practical option on this workstation with the available training stacks, even
with CPU RAM and SSD offloading.** The main model plus Engram contains about
748.495 billion logical parameters. Ordinary mixed-precision Adam configurations
would need approximately 8.98–11.98 TB of weights, gradients and optimizer state,
before activations, temporary storage and checkpoints.

**A narrower experiment fits the estimated capacity: train Lucifer's existing
output head while freezing the backbone, Engram, final norm, draft and vision
weights.** That changes 661,913,600 original model weights (about 0.0884% of the
primary model). It is partial base-weight training, not full-model training or
a LoRA adapter. Its usefulness and end-to-end compatibility remain unproven;
the gated plan below establishes those before integration into Sleep.

Creating additional numbered checkpoint files only shards storage. It does not
add parameters, increase context capacity, or provide a training implementation.
The existing Qwen-based `lucifer-train` adapter workflow is separate from this
assessment of the primary model.

## Measured machine and exact installed model

Hardware was read without starting an inference job, training job, disk benchmark
or GPU benchmark. GPU values came from the backend's cached telemetry. Lucifer
was busy at inspection and was left running.

| Resource | Observed |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5070, 12,227 MiB (11.94 GiB) VRAM |
| CPU | AMD Ryzen 9 9900X, 12 cores / 24 threads |
| RAM | 67,042,668,544 bytes visible to Linux (62.44 GiB; 64 GB class) |
| Model filesystem | ext4, Lexar NM1090 PRO 2 TB NVMe, PCIe 5.0 ×4 link |
| Available filesystem space | About 1.415 TB at inspection |
| Other disks | Unmounted NTFS volumes; excluded from usable capacity |
| SSD throughput / training throughput | Not measured |

GB and TB below use decimal units; GiB uses powers of two. RAM availability and
free disk space change while Lucifer works. The existing inference cache occupies
most RAM and VRAM, so even the small training option needs separate
capture / unload / optimize / reload stages.

Parameter counts were calculated from the dimensions of all 48,786 manifest
tensors, with no repeated storage ranges found:

| Installed component | Logical parameters | Packed tensor payload |
| --- | ---: | ---: |
| Main backbone, files 0–3 | 551,880,819,824 | 138.233 GB |
| Engram tables, file 4 | 196,613,849,600 | 110.595 GB |
| **Primary total** | **748,494,669,424** | **248.828 GB** |
| Optional DSpark draft, file 5 | 14,225,362,146 | 7.972 GB |
| Vision, file 6 | 485,268,480 | 0.972 GB |
| Entire installed pack | 763,205,300,050 | 257.772 GB |

The actual seven referenced files total 257.783 GB including file metadata.
Six routed experts per token does not remove the stored parameters of the other
experts or their potential training state.

The installed pack is
`/home/Lucifer/Soul/var/strata/models/packs/deepseek-v4.1-flash-ist-das-2.00bpw`.
Its `tensors.txt` SHA-256 is
`46197620c7113b7d661761ca29ef4ebf0a7aa03cda5cf7c2e5665e3c6d097395`.
The upstream architecture is `deepseek_v41`, checkpoint
`deepseek-ai/DeepSeek-V4.1-Flash-1017`, inspected at revision
`08c8657f02291e11b6f6b63e0e16014f8e748518`. Its public checkpoint already uses
mixed FP8/FP4 quantization; availability of an original BF16 release was not
established. Downloading an unspecified “unquantized original” is therefore not
a valid implementation step. See the [pinned upstream configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/08c8657f02291e11b6f6b63e0e16014f8e748518/config.json)
and [model provenance](MODEL_PROVENANCE.md).

## Why full training does not fit

For the primary model only, excluding the optional draft and vision components:

| Example allocation | Bytes per parameter | Capacity |
| --- | ---: | ---: |
| BF16 weights | 2 | 1.497 TB |
| BF16 gradients | 2 | 1.497 TB |
| Two FP32 Adam moments | 8 | 5.988 TB |
| Subtotal | 12 | **8.982 TB** |
| Optional FP32 master weights | 4 | 2.994 TB |
| Total with master weights | 16 | **11.976 TB** |

These are explicit ordinary Adam scenarios, **not universal lower bounds for
every possible optimizer**. Other gradient/state precisions change the totals.
Activations, optimizer workspaces, conversion buffers and checkpoint copies are
additional. Even freezing Engram leaves 551.881B backbone parameters and about
6.62–8.83 TB under these same scenarios.

Offloading moves allocations between GPU, RAM and NVMe; it does not eliminate
them. Standard ZeRO-3 also requires each forward/backward submodule to fit unless
it is split or tiled. Here a large backbone layer alone would be about 27.85 GB
in BF16, and either Engram table would be about 196.61 GB. Expert/row tiling and
custom architecture support would be necessary in addition to much more storage.
See [DeepSpeed ZeRO-3 and NVMe offload documentation](https://deepspeed.readthedocs.io/en/latest/zero3.html).

The installed Strata C++/CUDA runtime and the inspected upstream reference are
inference implementations, not an autograd trainer. Upstream reference forward
methods use `torch.inference_mode()`. A model loading successfully for inference
does not demonstrate that its quantized kernels have backward implementations.
The exact-architecture [ms-swift full-parameter training example](https://github.com/modelscope/ms-swift/blob/3aadbf2aceeeec86e286d9252b4bffbc609b68b1/examples/megatron/grpo/deepseek_v41_colocate_full.sh)
uses an H200-class cluster and custom Megatron components; it also freezes Engram.
This establishes that training integrations exist, not that they fit this PC.

Alternative methods were checked rather than treating Adam's footprint as an
absolute impossibility proof:

| Route | Finding for this exact installation |
| --- | --- |
| Ordinary full-weight training with RAM/NVMe offload | Storage capacity and training implementation are insufficient. |
| Native low-precision LoRA | The [KTransformers recipe](https://github.com/kvcache-ai/ktransformers/blob/1833914c91bf318c5fcd302094beb5df707e94fc/doc/en/SFT/DeepSeek-V4-Flash-LoRA-E2E.md) targets a different Flash release and representation; it does not validate this V4.1 GSQ/RCO GGUF pack. |
| Fused low-state backpropagation | [LOMO/AdaLOMO](https://github.com/OpenLMLab/LOMO) reduce optimizer memory but still need compatible backward operations; no working route for this pack was established. |
| Forward-only zeroth-order updates | [MeZO](https://github.com/princeton-nlp/MeZO) and [ZO2](https://github.com/liangyuwang/zo2) make lower-memory training plausible in other models. Neither establishes this 748B, 12 GB VRAM, GSQ/RCO/Engram configuration; ZO2's published disk-offload roadmap is incomplete. This remains research, not a concrete supported full-base plan. |
| Existing output-head training | Capacity fits on paper; source hooks exist. Proceed only through the compatibility and quality gates below. |

## Concrete plan: update the existing output head

All files, flags, workers and job types described below are **proposed additions**.
The only executable delivered by this assessment is the read-only capacity script.
No head exporter or head trainer is currently installed by this work.

### 0. Pin the source and establish a reproducible baseline

Obtain the complete source and license/build instructions matching the deployed
Strata engine 0.1.59 (control 0.1.59.1, Strata v0.2.13). The local research
snapshots lack `ds4_model.hpp`; neither `nvcc` nor `cmake` was found on PATH.
Do not assume these partial snapshots can rebuild the deployed engine.

Build an isolated engine under `/home/Lucifer/Soul/var/strata/builds/`; preserve
the installed binary and original model files. Record source revision, toolchain,
binary hashes, tokenizer, template, pack and profile revision. Hash original
weights during a scheduled idle window; do not read 258 GB merely to repeat this
assessment. Record baseline logits, heldout loss, tool-call validity, instruction
following and repetition/loop behavior on a small fixed evaluation set.

**Gate:** compatible source/build is available, and the rebuilt unchanged engine
matches the baseline's measured numerical repeatability. Otherwise stop here.

### 1. Prove that a head sidecar can load without changing the model

The existing independent `head.weight` is Q5_K, shape `[129280, 5120]`, stored
in file 3 at offset 19,856,318,592, length 455,065,600 bytes. `embed.weight`
has a different storage range; the two are not tied.

First copy just this original Q5_K payload into a candidate sidecar. Create a new
pack manifest referencing all the original files and the copied head. Use the
existing dtype and offset zero. Original files remain read-only.

Source inspection shows `Pack::load` reads raw bytes at manifest offsets, and
`mat()` accepts BF16. File names resolve relative to `model_dir`, not the pack
directory. Generate sidecar paths with `os.path.relpath`; do not assume absolute
file entries work. **Replace** the old `head.weight` row: duplicate tensor names
retain the first entry.

**Gate:** payload hashes match exactly; no-op sidecar logits match the untouched
pack within the baseline repeatability envelope. This tests actual deployed
loader behavior before any numerical conversion or optimization.

### 2. Export frozen-backbone features with correct labels

Add a versioned teacher-forced feature-export interface to the isolated engine.
The exact feature is the 5,120-element `d.x` after final
`hc_pre_norm(... d.norm_w ...)` in `head_rows_dev`, immediately before
`head_share` (`ds4_model.cu:13334–13341` in the inspected snapshot).
Synchronize and copy it before the device buffer is reused.

For the first exporter, force the main window path with `--prefill-chunk 0`,
clear/override `DS4_CHUNK` (which can override that CLI setting), and assert the
effective chunk size is zero. Skip draft loading with `--dspark none`, and
disable all speculation and sampling.
The existing `--dump-logits` teacher-forced loop in `ds4_main.cpp:2694–2708`
is a useful pattern, but it does not export hidden features. A blind hook in
`head_share` is insufficient because draft/confidence paths also use it. Supporting
chunked prefill later requires handling its separate head sites too.

Use the pinned tokenizer and actual chat template. Pair the feature at position
*t* with token *t+1*, streaming every intended position. Retain assistant-target
masks; exclude system/user targets, padding and raw provider reasoning. Consume
currently consented, reviewed examples from the existing knowledge/chat pipeline;
reading a document does not itself authorize training. Split by source document
or conversation **before** feature extraction to prevent train/heldout leakage.

Cache headers must bind backbone/Engram/norm hashes, source content revisions,
tokenizer/template, causal position, mask, dtype and exporter build. Source or
consent changes invalidate dependent caches. Start with 1,000–10,000 supervised
tokens and measure extraction throughput and peak memory before estimating a
larger run. BF16 features cost 10,240 bytes/token: 1.024 GB for 100,000 tokens,
10.24 GB for one million. Bound the cache and stream shards from SSD.

**Gate:** features reproduce reference logits within a recorded tolerance;
alignment and masks are independently checked on hand-inspected examples.
Measure BF16 feature-rounding error against float32 features; use float32 if the
rounding budget is unacceptable.

### 3. Verify a BF16 head before training it

Implement an engine-consistent Q5_K decoder for the existing head, producing
row-major BF16 `[out, in]`. Dequantization does not recover discarded precision.
The BF16 tensor is 1,323,827,200 bytes. Test numerical/logit parity and heldout
loss drift with the backbone unchanged, separating head conversion error from
feature-cache rounding error.

A candidate-only manifest can then reference a raw little-endian sidecar:

```text
file 7 <relative-path-from-original-model-dir-to-candidate/head.bf16.bin>
t head.weight BF16 7 0 1323827200 129280 5120
```

The replacement pack's `model_dir` remains the original model directory so all
other file entries resolve correctly. Source supports this format; the no-op
and BF16 loading gates must still prove it works with the installed binary.

**Gate:** BF16 conversion causes no unacceptable heldout/task regression. Do not
attribute conversion changes to learning or continue past a failed parity gate.

### 4. Train in a separate worker after unloading inference

Add an isolated `native_head` worker, not a PEFT adapter job and not a fallback
to the Qwen training profile. Wait for an idle mission, acquire the existing
model lifecycle lock, save the current profile, then unload primary inference.
Train cross-entropy on cached features with the existing head as initialization.
Only the head is trainable; all upstream tensors and features remain fixed.

Use GPU BF16 matrix multiplication with FP32 loss/accumulation where needed,
FP32 CPU master weights and Adam moments. A small explicit CPU optimizer avoids
loading the full architecture into a Python training framework. Start with 32
feature rows per microbatch and increase only after measuring allocations;
accumulate gradients when useful. Copy accumulated gradients to CPU, update the
master head, then synchronize the updated GPU copy. CPU updates trade speed for
memory; pinned buffers and optimizer temporary allocations must be included.

| Head allocation | Estimated size |
| --- | ---: |
| GPU BF16 weights and BF16 gradients | 2.648 GB |
| CPU FP32 master weights and two moments | 7.943 GB |
| Combined persistent mixed state | 10.591 GB |
| Possible additional CPU FP32 gradient copy | 2.648 GB |
| Activations, logits, framework/optimizer temporaries | Additional; measure |

Initial engineering caps are **6 GiB GPU, 24 GiB RAM and 50 GB additional disk**
for the pilot, with bounded cache, checkpoint retention and free-space reserve.
These are proposed limits, not measured peaks. Check limits before allocations
and abort cleanly on excess. Checkpoint optimizer/seed/data cursor, hashes and
loss history; keep resumable checkpoints bounded. Verify finite gradients and
real optimizer steps. Use a disposable overfit check, then select a conservative
learning rate on the training/validation split; leave the final heldout split
untouched. Benchmark enough updates to estimate a cycle; inference decode
tokens/second cannot establish training time.

**Gate:** real gradients and updates, measured resource use within caps, improved
validation loss, and a useful measured runtime. Otherwise discard the candidate
and restore inference. The head experiment can stop without a full-model download.

### 5. Evaluate a candidate pack, then permit explicit activation

Write the learned BF16 head and replacement manifest to
`/home/Lucifer/Soul/var/strata/head-candidates/<id>/`; never overwrite the original
head. Hashes must prove that only the candidate head changed. Register it as a
candidate **base variant**, with exact parent pack and tokenizer, rather than
pretending it is an adapter.

Serving BF16 instead of the current Q5_K head adds about **0.869 GB of weight
storage**, and may alter workspace needs. There was less free VRAM than this
at inspection. Reserve memory by reducing the GPU expert cache and measure peak
serving allocations before loading. Do not attempt an unchanged serving budget.
An optional Q8/Q5 export is a later optimization with its own quality/parity gate.

Evaluate against both the original Q5_K model and the untrained BF16-head
baseline. Require improved final heldout performance on target sources and
predeclared non-regression limits for general instructions, tool syntax, repeated
calls/loops and unrelated tasks. Restore the prior inference profile on success,
failure or cancellation. Record training status separately from quality status.
Promotion stays manual; activation and rollback switch an atomic profile pointer
between intact packs. Consent revocation blocks future use and affected candidate
promotion under the existing policy; removing training data cannot undo prior
learning in an already trained tensor.

**Gate:** resource, heldout and regression checks pass and a user explicitly
activates the candidate. A loss reduction alone is insufficient for promotion.

### 6. Connect to Sleep only after the isolated pilot passes

Add an explicit `native_head` job kind across worker dispatch, candidate registry,
model lifecycle, resume/cancellation and Sleep/Cortex display. Display exactly
“primary model, output head only, 661.9M trainable parameters”; do not label it
full-base training or silently redirect it to `lucifer-train`. Reuse reviewed
knowledge/document examples and the existing consent ledger. Unsupported image
or video feature exports must fail clearly rather than becoming text-only
training silently; initial support can be text examples only.

Reuse a valid frozen-backbone cache across head-only cycles; invalidate it when
any dependency changes. Serialize capture, optimization and candidate evaluation
with chat inference. Capture and evaluation use the streamed primary engine;
only the optimizer stage needs to unload it. Add focused tests for tensor
remapping, labels/masks, resource aborts, interrupted profile restoration,
consent invalidation and candidate rollback when implementing these features.

This is the point at which a successful prototype can become a repeatable Sleep
mode. Until then, keep existing Sleep behavior and its separate training base
unchanged.

## What this can and cannot establish

Head training can change next-token preferences using features the existing
backbone already represents. It cannot update internal attention, reasoning,
expert routing or Engram representations. Useful learning from new documents is
an empirical question; the parameter count does not guarantee it. Retrieval
remains necessary for reliable access to source documents.

No training speed, quality gain, complete source build, backward compatibility
or end-to-end head update has been demonstrated by this assessment. No model
weights were downloaded, no service was restarted and no running mission was
interrupted. Only public metadata/source evidence, the audit script and this
plan were added.

## Reproduce the capacity calculation

Run from `/home/Lucifer/Soul` (reads the manifest, Linux hardware data and cached
backend status; starts no model job):

```bash
.venv/bin/python scripts/assess_base_training.py \
  --output var/validation/base-training-feasibility-20261113/hardware-and-capacity.json
```

Evidence saved under
`/home/Lucifer/Soul/var/validation/base-training-feasibility-20261113/`:

- [Hardware and capacity snapshot](../var/validation/base-training-feasibility-20261113/hardware-and-capacity.json)
- [Per-tensor and file audit](../var/validation/base-training-feasibility-20261113/model-tensor-audit.json)
- [Head pilot source assessment](../var/validation/base-training-feasibility-20261113/head-pilot-source-assessment.json)
- [Line-numbered Strata source excerpts and hashes](../var/validation/base-training-feasibility-20261113/strata-head-source-excerpts.txt)
- [Pinned upstream source URLs and hashes](../var/validation/base-training-feasibility-20261113/upstream-source-receipts.json)

The read-only helper is [assess_base_training.py](../scripts/assess_base_training.py).
Its file-group interpretation is specific to the installed seven-file Lucifer
pack; inspect grouping before reusing it with a different model.
