"""Isolated PEFT worker. Run only with an explicit reviewed dataset/config.

Core installation does not import or install GPU packages. Use a training venv
and LUCIFER_TRAIN_PYTHON when the inference and training stacks differ.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path


def configure_training_allocator(config, environ=None):
    """Choose a worker-local default before importing Torch; honor overrides."""
    environ = os.environ if environ is None else environ
    names = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF")
    supplied = {key: environ[key] for key in names if key in environ}
    single_worker = int(environ.get("WORLD_SIZE", "1")) == 1
    applied = config.get("method", "qlora") == "qlora" and single_worker and not supplied
    if applied:
        environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    return {"default_applied": applied, "single_worker": single_worker,
            "environment": {key: environ[key] for key in names if key in environ},
            "reason": "single_worker_qlora_variable_allocations" if applied else "explicit_or_non_qlora_or_distributed_policy"}


def capture_embedding_precision(model):
    """Capture unique parameters before PEFT's bulk FP32 preparation."""
    unique = {}
    for role, getter_name in (("input_embeddings", "get_input_embeddings"), ("output_embeddings", "get_output_embeddings")):
        getter = getattr(model, getter_name, None)
        module = getter() if callable(getter) else None
        parameter = getattr(module, "weight", None)
        if parameter is None:
            continue
        record = unique.setdefault(id(parameter), {"parameter": parameter, "dtype": parameter.dtype, "roles": []})
        record["roles"].append(role)
    return list(unique.values())


def restore_frozen_embedding_precision(records):
    """Restore only frozen embedding/head matrices, retaining tied aliases."""
    import torch
    result, saved = [], 0
    for record in records:
        parameter, original_dtype = record["parameter"], record["dtype"]
        if parameter.requires_grad:
            raise ValueError("Embedding precision restoration requires an already frozen base parameter")
        before_dtype = parameter.dtype
        before_bytes = parameter.numel() * parameter.element_size()
        if original_dtype in {torch.float16, torch.bfloat16} and before_dtype == torch.float32:
            # Change storage, not the Parameter object: tied input/output
            # weights must remain shared and excluded from the optimizer.
            parameter.data = parameter.data.to(dtype=original_dtype)
        after_bytes = parameter.numel() * parameter.element_size()
        saved += before_bytes - after_bytes
        result.append({"roles": record["roles"], "shape": list(parameter.shape),
            "original_dtype": str(original_dtype), "prepared_dtype": str(before_dtype), "training_dtype": str(parameter.dtype),
            "requires_grad": parameter.requires_grad, "bytes_before_policy": before_bytes, "bytes_after_policy": after_bytes})
    return {"strategy": "restore_original_frozen_embedding_precision", "unique_embedding_parameters": len(result),
            "embedding_bytes_saved": saved, "normalization_policy": "PEFT FP32 normalization preparation retained", "embeddings": result}


def _token_pair(tokenizer, messages):
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("Training examples must end with an assistant answer")
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False
    )
    full = tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False, add_generation_prompt=False, enable_thinking=False
    )
    if full[:len(prompt)] != prompt:
        raise ValueError("Chat template changed the generation prefix; cannot safely mask assistant-only loss")
    return prompt, full


def _encoded_pair(prompt, full, max_length):
    full = full[:max_length]
    labels = [-100] * min(len(prompt), len(full)) + full[len(prompt):]
    if not any(label != -100 for label in labels):
        raise ValueError("Example has no assistant tokens after truncation; increase max_seq_length")
    return {"input_ids": full, "attention_mask": [1] * len(full), "labels": labels}


def encode_example(tokenizer, messages: list[dict], max_length: int) -> dict:
    """Supervise only the final assistant, with an exact template prefix check."""
    return _encoded_pair(*_token_pair(tokenizer, messages), max_length)


def encode_examples(tokenizer, rows, max_length, *, dynamic=False):
    pairs = [_token_pair(tokenizer, row["messages"]) for row in rows]
    if not pairs:
        raise ValueError("Dataset is empty")
    if dynamic:
        required = max(len(full) for _, full in pairs)
        if required > 32768:
            raise ValueError("A reviewed target exceeds the supported training window; refusing silent truncation")
        max_length = max(max_length, ((required + 127) // 128) * 128)
    return [_encoded_pair(prompt, full, max_length) for prompt, full in pairs], max_length



class AdaptiveSegmentTracker:
    """Count actual optimizer updates and stop only at a safe update boundary."""

    def __init__(self, state, *, native_epoch=0.0):
        from .adaptive import AdaptiveState
        self.state = AdaptiveState.from_dict(state.to_dict())
        if not self.state.round_remaining:
            raise ValueError("Adjust the completed round before starting another worker")
        if self.state.global_successful_updates == self.state.next_refresh_at:
            raise ValueError("Consume the pending refresh before starting another worker")
        self.start_successful_updates = self.state.global_successful_updates
        self.native_pass = math.floor(native_epoch + 1e-8)
        self.native_epoch = native_epoch
        self.last_native_step = None
        self.boundary_reasons = []

    @property
    def successful_updates(self):
        return self.state.global_successful_updates - self.start_successful_updates

    def optimizer_boundary(self, *, native_step, native_epoch, skipped):
        if type(native_step) is not int or native_step < 1:
            raise ValueError("Invalid native optimizer boundary")
        if self.last_native_step is not None and native_step <= self.last_native_step:
            raise ValueError("Native optimizer boundary was repeated or moved backwards")
        if type(skipped) is not bool:
            raise ValueError("Optimizer success must be explicitly observed")
        if self.boundary_reasons:
            raise ValueError("Worker crossed an already completed segment boundary")
        if type(native_epoch) not in (int, float) or not math.isfinite(native_epoch) or native_epoch < 0:
            raise ValueError("Invalid native data-pass counter")
        self.last_native_step = native_step
        self.native_epoch = float(native_epoch)
        if not skipped:
            self.state.consume_updates(1)
        completed_pass = math.floor(native_epoch + 1e-8)
        if completed_pass > self.native_pass:
            self.state.dataset_passes_completed += completed_pass - self.native_pass
            self.native_pass = completed_pass
            self.boundary_reasons.append("dataset_pass")
        if self.state.global_successful_updates == self.state.next_refresh_at:
            self.boundary_reasons.append("refresh")
        if not self.state.round_remaining:
            self.boundary_reasons.append("round")
        return bool(self.boundary_reasons)


def _cuda_rng_uses_all_states(distributed, setter=None):
    """Discover the installed Trainer's RNG container contract without CUDA.

    The native helper receives a fake CUDA module: it only chooses a setter,
    never allocates or queries a device. Older/newer Transformers use different
    single/distributed container conventions, so world size alone is insufficient.
    """
    if setter is None:
        from transformers.trainer_pt_utils import set_rng_state_for_device
        setter = set_rng_state_for_device
    observed = []
    class RandomSetters:
        @staticmethod
        def set_rng_state_all(value):
            observed.append(True)
        @staticmethod
        def set_rng_state(value):
            observed.append(False)
    class Device:
        random = RandomSetters
    setter("CUDA", Device, {"cuda": object()}, distributed)
    if len(observed) != 1:
        raise ValueError("Installed Trainer RNG container contract is unknown")
    return observed[0]


def _adaptive_argument_options(arguments_class, data_revision, sampler_reset):
    options = {"lr_scheduler_type": "constant", "save_only_model": False,
               "data_seed": int(data_revision[:8], 16) % (2 ** 31), "ignore_data_skip": sampler_reset}
    # Transformers 5 saves safetensors by default and removed this keyword.
    # Older supported releases still expose the explicit switch.
    if "save_safetensors" in inspect.signature(arguments_class).parameters:
        options["save_safetensors"] = True
    return options


def _rewrite_topology_rngs(source, destination, old_world_size, world_size, torch, *, cuda_rng_all_states=None):
    """Map committed rank RNGs explicitly when the controller changes topology.

    These are our own hash-verified checkpoints, never arbitrary pickle inputs.
    Preserve each old rank's active CUDA state while adapting the installed
    Trainer's discovered single/distributed container shape.
    """
    for rank in range(world_size):
        old_rank = rank % old_world_size
        source_name = "rng_state.pth" if old_world_size == 1 else f"rng_state_{old_rank}.pth"
        state = torch.load(Path(source) / source_name, map_location="cpu", weights_only=False)
        if "cuda" in state:
            cuda_state = state["cuda"]
            if isinstance(cuda_state, (list, tuple)):
                if not cuda_state:
                    raise ValueError("Committed CUDA RNG state is empty")
                cuda_state = cuda_state[old_rank % len(cuda_state)]
            use_all = _cuda_rng_uses_all_states(world_size > 1) if cuda_rng_all_states is None else cuda_rng_all_states
            state["cuda"] = [cuda_state] * world_size if use_all else cuda_state
        target_name = "rng_state.pth" if world_size == 1 else f"rng_state_{rank}.pth"
        target = Path(destination) / target_name
        temporary = target.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)


def _adaptive_setup(config, dataset_bytes, output, world_size, rank, torch):
    from .adaptive import AdaptiveState, prepare_segment_resume
    settings = config.get("adaptive")
    if settings is None:
        return None
    if not isinstance(settings, dict):
        raise ValueError("adaptive configuration must be an object")
    revision = settings.get("data_revision")
    if not isinstance(revision, str) or hashlib.sha256(dataset_bytes).hexdigest() != revision:
        raise ValueError("Adaptive dataset bytes do not match the committed revision")
    segment_id = settings.get("segment_id")
    if not isinstance(segment_id, str) or not segment_id or len(segment_id) > 200:
        raise ValueError("An adaptive segment needs a stable segment ID")
    state = AdaptiveState.from_dict(settings.get("state"))
    checkpoint = settings.get("resume_checkpoint")
    expected = settings.get("resume_manifest_sha256")
    native_epoch, manifest, sampler_reset = 0.0, None, False
    if checkpoint:
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("Resuming requires the committed checkpoint manifest SHA256")
        # Only rank0 materializes a new data/topology cursor. All ranks read it
        # after a barrier in train(); verification itself remains rank-local.
        if rank == 0:
            checkpoint, native_epoch, manifest, sampler_reset = prepare_segment_resume(
                checkpoint, output / "resume-input", expected_sha256=expected,
                data_revision=revision, world_size=world_size, state=state)
            if world_size != manifest["world_size"]:
                _rewrite_topology_rngs(settings["resume_checkpoint"], checkpoint, manifest["world_size"], world_size, torch)
        else:
            from .adaptive import verify_checkpoint
            manifest = verify_checkpoint(checkpoint, expected)
            sampler_reset = revision != manifest.get("data_revision") or world_size != manifest["world_size"]
            if sampler_reset:
                checkpoint = output / "resume-input"
            else:
                native = json.loads((Path(checkpoint) / "trainer_state.json").read_text())
                native_epoch = float(native.get("epoch") or 0)
        for name in ("base_model", "method", "rank", "batch_size", "gradient_accumulation", "learning_rate"):
            if manifest.get("training_identity", {}).get(name) != config.get(name, {
                    "method": "qlora", "rank": 16, "batch_size": 1, "gradient_accumulation": 4,
                    "learning_rate": 0.0001}.get(name)):
                raise ValueError("Adaptive resume training identity changed")
    elif state.global_successful_updates:
        raise ValueError("Nonzero adaptive update counters require a committed checkpoint")
    return {"settings": settings, "state": state, "resume_checkpoint": str(checkpoint) if checkpoint else None,
            "native_epoch": native_epoch, "manifest": manifest, "sampler_reset": sampler_reset,
            "tracker": AdaptiveSegmentTracker(state, native_epoch=native_epoch)}


def _complete_adaptive_segment(config, adaptive, trainer, metrics, output, world_size, rank, tokenizer, torch):
    from .adaptive import atomic_json, commit_checkpoint
    tracker = adaptive["tracker"]
    if tracker.successful_updates < 1:
        raise ValueError("Adaptive segment completed without a successful optimizer update")
    if not tracker.boundary_reasons:
        raise ValueError("Trainer stopped without a committed adaptive boundary")
    if not isinstance(metrics.get("train_loss"), (int, float)) or not math.isfinite(metrics["train_loss"]):
        raise ValueError("Adaptive segment has no finite measured training loss")
    checkpoint = output / "checkpoints" / f"checkpoint-{int(trainer.state.global_step)}"
    if world_size > 1:
        torch.distributed.barrier()
    result = None
    if rank == 0:
        # Native Trainer saved the PEFT adapter and its complete continuation
        # state via should_save. These metrics make that same root evaluable.
        tokenizer.save_pretrained(checkpoint)
        atomic_json(checkpoint / "training_metrics.json", metrics)
        identity = {"base_model": config["base_model"], "method": config.get("method", "qlora"),
                    "rank": config.get("rank", 16), "batch_size": config.get("batch_size", 1),
                    "gradient_accumulation": config.get("gradient_accumulation", 4),
                    "learning_rate": config.get("learning_rate", 0.0001)}
        scaler = getattr(trainer.accelerator, "scaler", None)
        metadata = {"segment_id": adaptive["settings"]["segment_id"],
                    "data_revision": adaptive["settings"]["data_revision"],
                    "state": tracker.state.to_dict(), "world_size": world_size,
                    "training_identity": identity, "native_epoch": tracker.native_epoch,
                    "dataset_passes_completed": tracker.state.dataset_passes_completed,
                    "boundary_reasons": tracker.boundary_reasons, "scaler_required": scaler is not None,
                    "sampler_reset": adaptive["sampler_reset"], "lr_scheduler": "constant",
                    "sampler_policy": "native_trainer_cursor_with_revision_seed",
                    "sampler": {"data_revision": adaptive["settings"]["data_revision"],
                                "seed": trainer.args.data_seed, "native_epoch": tracker.native_epoch,
                                "native_global_step": int(trainer.state.global_step), "world_size": world_size,
                                "batch_size": config.get("batch_size", 1),
                                "gradient_accumulation": config.get("gradient_accumulation", 4),
                                "cursor_reset": adaptive["sampler_reset"]},
                    "data_seed": trainer.args.data_seed,
                    "rng_topology_policy": "rank_modulo_committed_world_with_cuda_shape_remap",
                    "source_base_revision": config.get("base_model_revision"),
                    "parent_manifest_sha256": adaptive["settings"].get("resume_manifest_sha256")}
        digest = commit_checkpoint(checkpoint, metadata)
        result = {"event": "segment_completed", "segment_id": metadata["segment_id"],
                  "state": tracker.state.to_dict(), "boundary_reasons": tracker.boundary_reasons,
                  "checkpoint_path": str(checkpoint), "checkpoint_manifest_sha256": digest,
                  "data_revision": metadata["data_revision"], "world_size": world_size,
                  "dataset_passes_completed": tracker.state.dataset_passes_completed,
                  "native_epoch": tracker.native_epoch, "sampler_reset": adaptive["sampler_reset"],
                  "metrics": metrics}
        atomic_json(output / "segment_result.json", result)
        print(json.dumps(result, allow_nan=False), flush=True)
    if world_size > 1:
        result_box = [result]
        torch.distributed.broadcast_object_list(result_box, src=0)
        result = result_box[0]
    return result


def train(config: dict) -> dict:
    allocator_policy = configure_training_allocator(config)
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainerCallback, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("Training dependencies missing. Install lucifer-local[train] in the worker Python environment.") from exc

    base = config["base_model"]
    if base.lower().endswith(".gguf"):
        raise ValueError("GGUF is inference-only; supply the original HF checkpoint for LoRA training")
    method = config.get("method", "qlora")
    if method == "qlora" and not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires a CUDA GPU and bitsandbytes; use method=lora for CPU smoke training")
    cuda = torch.cuda.is_available()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if cuda:
        torch.cuda.set_device(local_rank)
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl" if cuda else "gloo")
    output = Path(config["output_dir"])
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    if world_size > 1:
        torch.distributed.barrier()
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ValueError("Base tokenizer has no chat template; configure a compatible instruction model")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer needs a pad or EOS token")
    dataset_bytes = Path(config["dataset_path"]).read_bytes()
    rows = [json.loads(line) for line in dataset_bytes.decode().splitlines() if line.strip()]
    adaptive = _adaptive_setup(config, dataset_bytes, output, world_size, rank, torch)
    if world_size > 1:
        torch.distributed.barrier()
    encoded, maximum_length = encode_examples(tokenizer, rows, int(config.get("max_seq_length", 2048)), dynamic=config.get("dynamic", False))
    bf16 = cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else (torch.float16 if cuda else torch.float32)
    kwargs = {"trust_remote_code": False, "torch_dtype": dtype}
    if method == "qlora":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        kwargs["device_map"] = {"": torch.cuda.current_device()}
    base_config = AutoConfig.from_pretrained(base, trust_remote_code=False)
    text_config = base_config.get_text_config() if hasattr(base_config, "get_text_config") else getattr(base_config, "text_config", base_config)
    native_limit = getattr(text_config, "max_position_embeddings", None)
    if isinstance(native_limit, int) and native_limit > 0 and max(len(row["input_ids"]) for row in encoded) > native_limit:
        raise ValueError("A complete reviewed example exceeds the training model's native context; refusing silent truncation")
    conditional = bool(getattr(base_config, "text_config", None)) and any(
        "ForConditionalGeneration" in name for name in (base_config.architectures or [])
    )
    loader = AutoModelForCausalLM
    if conditional:
        # Qwen3.5 checkpoints include a vision tower and a nested language
        # model. Preserve checkpoint namespaces while training text examples.
        from transformers import AutoModelForImageTextToText
        loader = AutoModelForImageTextToText
    model = loader.from_pretrained(base, **kwargs)
    memory_policy = {"allocator": allocator_policy, "gradient_checkpointing_use_reentrant": False,
                     "complete_targets": bool(config.get("dynamic"))}
    if method == "qlora":
        original_embeddings = capture_embedding_precision(model)
        model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": False})
        memory_policy.update(restore_frozen_embedding_precision(original_embeddings))
    model.config.use_cache = False
    if hasattr(model.config, "get_text_config"):
        model.config.get_text_config().use_cache = False
    target_modules = "all-linear"
    if conditional:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora = LoraConfig(
        r=config.get("rank", 16), lora_alpha=config.get("rank", 16) * 2,
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=target_modules,
    )
    model = get_peft_model(model, lora)
    if rank == 0:
        model.print_trainable_parameters()
        print(json.dumps({"event": "memory_policy", **memory_policy}), flush=True)
    class Examples(torch.utils.data.Dataset):
        def __len__(self):
            return len(encoded)

        def __getitem__(self, index):
            return encoded[index]

    def collate(batch):
        size = max(len(row["input_ids"]) for row in batch)
        return {key: torch.tensor([
            row[key] + [(-100 if key == "labels" else tokenizer.pad_token_id if key == "input_ids" else 0)] * (size - len(row[key]))
            for row in batch
        ], dtype=torch.long) for key in ("input_ids", "attention_mask", "labels")}

    adaptive_args = {}
    if adaptive is not None:
        adaptive_args = _adaptive_argument_options(TrainingArguments, adaptive["settings"]["data_revision"],
                                                  adaptive["sampler_reset"])
    args = TrainingArguments(
        output_dir=str(output / "checkpoints"),
        num_train_epochs=math.floor(adaptive["native_epoch"] + 1e-8) + 1 if adaptive else config.get("epochs", 1),
        learning_rate=config.get("learning_rate", 0.0001),
        per_device_train_batch_size=config.get("batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation", 4),
        bf16=bf16, fp16=cuda and not bf16, logging_steps=1, save_strategy="no",
        disable_tqdm=True,
        report_to=[], remove_unused_columns=False, seed=3407,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch", dataloader_pin_memory=cuda,
        local_rank=local_rank if world_size > 1 else -1, use_cpu=not cuda and world_size > 1,
        ddp_find_unused_parameters=False if world_size > 1 else None,
        **adaptive_args,
    )
    class Progress(TrainerCallback):
        def emit(self, state, logs=None):
            if rank == 0:
                event = {"event": "training_progress", "step": int(state.global_step),
                         "total_steps": int(state.max_steps), "epoch": state.epoch}
                if logs and isinstance(logs.get("loss"), (int, float)):
                    event["loss"] = logs["loss"]
                print(json.dumps(event), flush=True)

        def on_train_begin(self, args, state, control, **kwargs):
            self.emit(state)

        def on_log(self, args, state, control, logs=None, **kwargs):
            self.emit(state, logs)

    class SegmentBoundary(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            # Some Trainer versions floor update steps per epoch when the last
            # accumulation window is partial. Keep its default global-step stop
            # beyond a complete data pass; our callback stops at the actual
            # round/refresh/pass boundary and the outer epoch loop stays finite.
            dataloader = kwargs.get("train_dataloader")
            if dataloader is None:
                raise ValueError("Adaptive Trainer did not expose its finite data loader")
            safe_updates = math.ceil(len(dataloader) / args.gradient_accumulation_steps) + 1
            state.max_steps = max(int(state.max_steps), int(state.global_step) + safe_updates)
            return control

        def on_step_end(self, args, state, control, **kwargs):
            tracker = adaptive["tracker"]
            skipped = trainer.accelerator.optimizer_step_was_skipped
            if type(skipped) is not bool:
                raise ValueError("Optimizer success must be explicitly observed")
            if world_size > 1:
                votes = torch.tensor(int(skipped), dtype=torch.int64,
                                     device=f"cuda:{local_rank}" if cuda else "cpu")
                torch.distributed.all_reduce(votes)
                if votes.item() not in (0, world_size):
                    raise ValueError("Distributed ranks disagree about optimizer success")
                skipped = votes.item() == world_size
            if tracker.optimizer_boundary(native_step=int(state.global_step), native_epoch=state.epoch,
                                          skipped=skipped):
                control.should_save = True
                control.should_training_stop = True
            if rank == 0:
                print(json.dumps({"event": "adaptive_progress", "segment_id": adaptive["settings"]["segment_id"],
                                  "global_successful_updates": tracker.state.global_successful_updates,
                                  "segment_successful_updates": tracker.successful_updates,
                                  "round_remaining": tracker.state.round_remaining,
                                  "randint_min": tracker.state.randint_min,
                                  "randint_max": tracker.state.randint_max,
                                  "next_refresh_at": tracker.state.next_refresh_at,
                                  "native_epoch": tracker.native_epoch, "optimizer_skipped": skipped}), flush=True)
            return control

    callbacks = [Progress()] + ([SegmentBoundary()] if adaptive else [])
    trainer = Trainer(model=model, args=args, train_dataset=Examples(), data_collator=collate, processing_class=tokenizer, callbacks=callbacks)
    result = trainer.train(resume_from_checkpoint=adaptive["resume_checkpoint"]) if adaptive else trainer.train()
    metrics = {key: value for key, value in result.metrics.items() if isinstance(value, (int, float))}
    metrics.update(optimizer_steps=int(trainer.state.global_step), trained_examples=len(encoded),
                   trainable_parameters=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                   effective_max_seq_length=maximum_length, longest_example_tokens=max(len(row["input_ids"]) for row in encoded),
                   memory_policy=memory_policy)
    if adaptive is not None:
        metrics = {key: value for key, value in metrics.items()
                   if not isinstance(value, (int, float)) or math.isfinite(value)}
        metrics.update(optimizer_steps=adaptive["tracker"].state.global_successful_updates,
                       segment_optimizer_steps=adaptive["tracker"].successful_updates,
                       native_trainer_global_step=int(trainer.state.global_step))
        return _complete_adaptive_segment(config, adaptive, trainer, metrics, output, world_size, rank, tokenizer, torch)
    if rank == 0:
        model.save_pretrained(output)
        tokenizer.save_pretrained(output)
        (output / "training_metrics.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps({"event": "completed", "output_dir": str(output), "metrics": metrics, "world_size": world_size}), flush=True)
    if world_size > 1:
        torch.distributed.barrier()
    return metrics


def worker_failure_event(error):
    """Structural frames only: never format exception text, code or locals."""
    frames = []
    trace = error.__traceback__
    while trace is not None:
        code = trace.tb_frame.f_code
        frames.append({"filename": Path(code.co_filename).name,
                       "line": trace.tb_lineno, "function": code.co_name})
        trace = trace.tb_next
    return {"event": "worker_failed", "error_type": type(error).__name__, "frames": frames[-64:]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get("adaptive") is None:
        train(config)
        return 0
    try:
        train(config)
    except Exception as error:
        print(json.dumps(worker_failure_event(error), allow_nan=False), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
