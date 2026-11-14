# Lucifer model provenance

Lucifer is the assistant's public identity and the default model profile in this
project. Interface labels and chat identity use this name for current and future
model profiles. Original checkpoint names remain in technical records because
the loaders, checksums, tokenizer definitions, and license notices need them.
Branding does not change the origin or authorship of upstream weights.

The primary inference profile uses:

- Original checkpoint: `deepseek-ai/DeepSeek-V4.1-Flash-1017` (MIT).
- Quantized weights: `ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF`,
  revision `cb02422c900df489cea5833e8a92171224059b3c`.
- Format: GSQ/RCO 2.00 bpw backbone (`IQ2_XXS`) with the shared `IQ4_NL` Engram table.
- Runtime: Strata v0.2.13, control 0.1.59.1 and CUDA engine 0.1.59, installed
  privately under `var/strata/`.
- Public runtime name: `Lucifer`; alias: `lucifer`.
- Current settings: six routed experts, model-native context resolved from the
  loaded pack on each launch (currently 1,048,576 tokens), one concurrent request,
  loopback HTTP only. Thinking defaults to off and remains selectable per request.

The install manifest, file hashes, download state, and runtime logs are retained
under `var/strata/`. These large machine-local artifacts are excluded from the
source distribution. Model downloads use Axel with 100 connections and are
verified against the pinned SHA-256 manifest before packing and loading.

The streamed primary model is an inference profile. Dataset preparation can run
as a dry run, but real LoRA jobs need a separate trainable profile. The smaller
training profile's source is `Qwen/Qwen3.5-4B` (Apache 2.0), served by the existing
vLLM installation. Its public assistant identity is also Lucifer.
This machine pins inference and training to cached revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, matching the tokenizer used for the
reviewed-data preflight. The profile records the local snapshot path rather than
following a moving repository branch during a sleep job.

The [primary base-training feasibility assessment](BASE_MODEL_TRAINING_FEASIBILITY.md)
records this workstation's capacity limits and the initial plan for training
only the primary model's existing output head. [Meditation](MEDITATION.md)
implements that limited native-weight route with supervised and unsupervised
modes; its guide records validation status. Ordinary full-parameter training
exceeds the current machine's resources.

Sources and upstream terms:

- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017
- https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF
- https://github.com/Niko1221/Strata/tree/v0.2.13
- https://huggingface.co/Qwen/Qwen3.5-4B
