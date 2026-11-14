"""Worker boundary semantics and a fully mocked native Trainer round trip."""
import hashlib
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest

from lucifer.adaptive import AdaptiveState, verify_checkpoint
from lucifer.experts import verify_adapter_artifacts
from lucifer.train_worker import (
    AdaptiveSegmentTracker, _adaptive_argument_options, _adaptive_setup, _cuda_rng_uses_all_states,
    _rewrite_topology_rngs, train,
)


def test_skipped_updates_do_not_advance_round_or_global_refresh():
    tracker = AdaptiveSegmentTracker(AdaptiveState.initial(seed=1))
    assert not tracker.optimizer_boundary(native_step=1, native_epoch=.1, skipped=True)
    assert tracker.successful_updates == tracker.state.global_successful_updates == 0
    assert tracker.state.round_remaining == 50
    assert tracker.optimizer_boundary(native_step=2, native_epoch=1.0, skipped=False)
    assert tracker.boundary_reasons == ["dataset_pass"]
    assert tracker.state.global_successful_updates == 1 and tracker.state.round_remaining == 49
    assert tracker.state.dataset_passes_completed == 1


def test_midround_refresh_preserves_remaining_budget():
    state = AdaptiveState.initial(seed=1)
    state.consume_updates(50)
    state.consume_refresh()
    state.adjust(1)
    budget, round_index = state.round_budget, state.round_index
    tracker = AdaptiveSegmentTracker(state)
    for index in range(1, 51):
        stop = tracker.optimizer_boundary(native_step=index, native_epoch=index / 100, skipped=False)
        assert stop == (index == 50)
    assert tracker.boundary_reasons == ["refresh"]
    assert tracker.state.round_remaining == budget - 50
    assert tracker.state.global_successful_updates == 100 and tracker.state.round_index == round_index
    tracker.state.consume_refresh()
    continued = AdaptiveSegmentTracker(tracker.state, native_epoch=.5)
    for index in range(1, budget - 50 + 1):
        continued.optimizer_boundary(native_step=50 + index, native_epoch=.5 + index / 100, skipped=False)
    assert continued.boundary_reasons == ["round"]
    assert continued.state.global_successful_updates == 50 + budget


def test_tracker_deduplicates_coincident_boundaries_and_rejects_double_updates():
    state = AdaptiveState.initial(seed=1)
    state.consume_updates(49)
    tracker = AdaptiveSegmentTracker(state, native_epoch=.98)
    assert tracker.optimizer_boundary(native_step=50, native_epoch=1.0, skipped=False)
    assert tracker.boundary_reasons == ["dataset_pass", "refresh", "round"]
    with pytest.raises(ValueError, match="repeated"):
        tracker.optimizer_boundary(native_step=50, native_epoch=1, skipped=False)
    with pytest.raises(ValueError, match="already completed"):
        tracker.optimizer_boundary(native_step=51, native_epoch=1, skipped=False)


def test_setup_requires_exact_dataset_revision_and_nonzero_resume_proof(tmp_path):
    data = b"reviewed dataset bytes"
    base = {"base_model": "org/student", "adaptive": {"state": AdaptiveState.initial(seed=1).to_dict(),
        "segment_id": "segment-1", "data_revision": "0" * 64}}
    with pytest.raises(ValueError, match="dataset bytes"):
        _adaptive_setup(base, data, tmp_path, 1, 0, None)
    state = AdaptiveState.initial(seed=1)
    state.consume_updates(2)
    base["adaptive"].update(state=state.to_dict(), data_revision=hashlib.sha256(data).hexdigest())
    with pytest.raises(ValueError, match="require a committed"):
        _adaptive_setup(base, data, tmp_path, 1, 0, None)


def _peft_files(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 1, "base_model_name_or_path": "org/student"}))
    header = json.dumps({"layer.lora_A.weight": {"dtype": "F32", "shape": [1, 1], "data_offsets": [0, 4]}}).encode()
    (path / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", .1))


@pytest.fixture
def fake_training(monkeypatch):
    """Exercise worker orchestration without importing Torch or Transformers."""
    calls = []
    class Tokenizer:
        chat_template = "template"
        pad_token_id = eos_token_id = 0
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()
        def apply_chat_template(self, messages, **kwargs):
            return [1, 2, 3, 4] if messages[-1]["role"] == "assistant" else [1, 2]
        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "tokenizer_config.json").write_text("{}")
    class Model:
        config = SimpleNamespace(use_cache=True)
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()
        def parameters(self):
            return [SimpleNamespace(requires_grad=True, numel=lambda: 1)]
        def print_trainable_parameters(self):
            pass
        def save_pretrained(self, path):
            _peft_files(Path(path))
    class Arguments:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    class Trainer:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.state = SimpleNamespace(global_step=0, epoch=0., max_steps=int(self.args.num_train_epochs) * 4)
            self.accelerator = SimpleNamespace(optimizer_step_was_skipped=False, scaler=None)
            calls.append(self)
        def train(self, resume_from_checkpoint=None):
            self.resumed_from = resume_from_checkpoint
            if resume_from_checkpoint:
                native = json.loads((Path(resume_from_checkpoint) / "trainer_state.json").read_text())
                self.state.global_step, self.state.epoch = native["global_step"], native["epoch"]
                assert (Path(resume_from_checkpoint) / "optimizer.pt").read_bytes() == b"owned-optimizer"
            control = SimpleNamespace(should_save=False, should_training_stop=False)
            for callback in self.callbacks:
                if hasattr(callback, "on_train_begin"):
                    callback.on_train_begin(self.args, self.state, control, train_dataloader=list(range(4)))
            while self.state.global_step < self.state.max_steps:
                self.state.global_step += 1
                self.state.epoch = self.state.global_step / 4
                self.accelerator.optimizer_step_was_skipped = self.state.global_step == 2
                for callback in self.callbacks:
                    if hasattr(callback, "on_step_end"):
                        callback.on_step_end(self.args, self.state, control)
                if control.should_save:
                    checkpoint = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
                    _peft_files(checkpoint)
                    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": self.state.global_step, "epoch": self.state.epoch}))
                    (checkpoint / "optimizer.pt").write_bytes(b"owned-optimizer")
                    (checkpoint / "scheduler.pt").write_bytes(b"owned-constant-scheduler")
                    (checkpoint / "rng_state.pth").write_bytes(b"owned-rng")
                    control.should_save = False
                if control.should_training_stop:
                    break
            return SimpleNamespace(metrics={"train_loss": .4})
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), float32="float32",
                            utils=SimpleNamespace(data=SimpleNamespace(Dataset=object)))
    peft = SimpleNamespace(LoraConfig=lambda **kwargs: kwargs, get_peft_model=lambda model, config: model,
                           prepare_model_for_kbit_training=lambda *args, **kwargs: None)
    transformers = SimpleNamespace(AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **kw:
        SimpleNamespace(architectures=[], max_position_embeddings=32768)), AutoModelForCausalLM=Model,
        AutoTokenizer=Tokenizer, BitsAndBytesConfig=object, Trainer=Trainer, TrainerCallback=object,
        TrainingArguments=Arguments)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return calls


def config(tmp_path, name, data, state, **adaptive):
    path = tmp_path / f"{name}.jsonl"
    path.write_bytes(data)
    return {"base_model": "org/student", "method": "lora", "dataset_path": str(path),
            "output_dir": str(tmp_path / name), "rank": 1,
            "adaptive": {"state": state.to_dict(), "segment_id": name,
                         "data_revision": hashlib.sha256(data).hexdigest(), **adaptive}}


def test_worker_saves_verified_full_checkpoint_and_continues_same_data(tmp_path, fake_training):
    data = json.dumps({"messages": [{"role": "user", "content": "literal task"},
                                    {"role": "assistant", "content": "reviewed final answer"}]}).encode() + b"\n"
    first = train(config(tmp_path, "segment1", data, AdaptiveState.initial(seed=1)))
    assert first["boundary_reasons"] == ["dataset_pass"]
    assert first["state"]["global_successful_updates"] == 3  # native step2 was skipped
    assert first["state"]["round_remaining"] == 47
    assert first["metrics"]["optimizer_steps"] == 3
    proof = verify_adapter_artifacts(first["checkpoint_path"])
    assert proof["optimizer_steps"] == 3 and proof["tensor_count"] == 1
    verify_checkpoint(first["checkpoint_path"], first["checkpoint_manifest_sha256"])
    second = train(config(tmp_path, "segment2", data, AdaptiveState.from_dict(first["state"]),
        resume_checkpoint=first["checkpoint_path"], resume_manifest_sha256=first["checkpoint_manifest_sha256"]))
    assert second["state"]["global_successful_updates"] == 7
    assert second["dataset_passes_completed"] == 2
    assert not second["sampler_reset"] and fake_training[-1].args.num_train_epochs == 2
    assert fake_training[-1].args.lr_scheduler_type == "constant"
    assert fake_training[-1].resumed_from == first["checkpoint_path"]


def test_worker_new_data_uses_private_cursor_reset_without_resetting_update_count(tmp_path, fake_training):
    data = json.dumps({"messages": [{"role": "user", "content": "task"}, {"role": "assistant", "content": "answer"}]}).encode()
    first = train(config(tmp_path, "old", data, AdaptiveState.initial(seed=2)))
    changed = data + b"\n"  # exact snapshot revision changes, preserving valid rows
    second = train(config(tmp_path, "new", changed, AdaptiveState.from_dict(first["state"]),
        resume_checkpoint=first["checkpoint_path"], resume_manifest_sha256=first["checkpoint_manifest_sha256"]))
    assert second["sampler_reset"] and second["state"]["global_successful_updates"] == 6
    assert fake_training[-1].args.ignore_data_skip
    assert fake_training[-1].args.num_train_epochs == 1
    assert fake_training[-1].resumed_from.endswith("resume-input")
    assert json.loads((Path(first["checkpoint_path"]) / "trainer_state.json").read_text())["global_step"] == 4
    verify_checkpoint(first["checkpoint_path"], first["checkpoint_manifest_sha256"])


def test_one_shot_worker_still_has_no_adaptive_scheduler_or_resume(tmp_path, fake_training):
    data = json.dumps({"messages": [{"role": "user", "content": "task"}, {"role": "assistant", "content": "answer"}]}).encode()
    settings = config(tmp_path, "one-shot", data, AdaptiveState.initial(seed=2))
    del settings["adaptive"]
    result = train(settings)
    assert result["optimizer_steps"] == 4
    assert not hasattr(fake_training[-1].args, "lr_scheduler_type")
    assert fake_training[-1].resumed_from is None
    assert (Path(settings["output_dir"]) / "adapter_model.safetensors").exists()


def test_partial_accumulation_window_cannot_trigger_early_native_max_steps(tmp_path, fake_training):
    data = json.dumps({"messages": [{"role": "user", "content": "task"}, {"role": "assistant", "content": "answer"}]}).encode()
    train(config(tmp_path, "partial", data, AdaptiveState.initial(seed=2)))
    callback = next(callback for callback in fake_training[-1].callbacks if hasattr(callback, "on_step_end"))
    state = SimpleNamespace(global_step=0, max_steps=2)
    callback.on_train_begin(SimpleNamespace(gradient_accumulation_steps=4), state, SimpleNamespace(),
                            train_dataloader=list(range(10)))
    assert state.max_steps == 4  # beyond ceil(10/4), not old floor(10/4)=2


def test_removed_safetensors_argument_is_not_passed_to_new_trainer_api():
    class ModernArguments:
        def __init__(self, *, lr_scheduler_type, save_only_model, data_seed, ignore_data_skip):
            pass
    class OlderArguments:
        def __init__(self, *, save_safetensors=False, **kwargs):
            pass
    modern = _adaptive_argument_options(ModernArguments, "a" * 64, False)
    assert "save_safetensors" not in modern
    ModernArguments(**modern)  # the installed 5.x signature must accept these keys
    older = _adaptive_argument_options(OlderArguments, "a" * 64, True)
    assert older["save_safetensors"] is True and older["ignore_data_skip"]


def test_rng_container_discovery_handles_both_native_conventions_without_cuda():
    def modern(name, device, state, distributed):
        callback = device.random.set_rng_state_all if distributed else device.random.set_rng_state
        callback(state["cuda"])
    def older(name, device, state, distributed):
        callback = device.random.set_rng_state if distributed else device.random.set_rng_state_all
        callback(state["cuda"])
    assert _cuda_rng_uses_all_states(True, modern)
    assert not _cuda_rng_uses_all_states(False, modern)
    assert not _cuda_rng_uses_all_states(True, older)
    assert _cuda_rng_uses_all_states(False, older)


def test_topology_rng_remap_uses_active_old_rank_and_target_native_shape(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    destination.mkdir()
    states = {"rng_state_0.pth": {"cuda": ["rank0-device0", "unused-device1"], "cpu": "cpu0"},
              "rng_state_1.pth": {"cuda": ["unused-device0", "rank1-device1"], "cpu": "cpu1"}}
    saved = {}
    class FakeTorch:
        @staticmethod
        def load(path, **kwargs):
            return dict(states[path.name])
        @staticmethod
        def save(state, handle):
            saved[Path(handle.name).stem] = state
            handle.write(b"owned-checkpoint")
    _rewrite_topology_rngs(source, destination, 2, 3, FakeTorch, cuda_rng_all_states=True)
    assert saved["rng_state_0"]["cuda"] == ["rank0-device0"] * 3
    assert saved["rng_state_1"]["cuda"] == ["rank1-device1"] * 3
    assert saved["rng_state_2"]["cuda"] == ["rank0-device0"] * 3
    saved.clear()
    _rewrite_topology_rngs(source, destination, 2, 1, FakeTorch, cuda_rng_all_states=False)
    assert saved["rng_state"]["cuda"] == "rank0-device0"


def test_adaptive_main_sanitizes_worker_failure_and_returns_nonzero(tmp_path, monkeypatch, capsys):
    import lucifer.train_worker as worker
    path = tmp_path / "configuration.json"
    path.write_text(json.dumps({"adaptive": {"segment_id": "one"}}))
    monkeypatch.setattr(sys, "argv", ["worker", "--config", str(path)])
    def failing_train(config):
        private_row = "PRIVATE_TRAINING_ROW"
        try:
            raise KeyError("PRIVATE_INNER_MESSAGE")
        except KeyError as cause:
            raise RuntimeError("PRIVATE_EXCEPTION_MESSAGE " + private_row) from cause
    monkeypatch.setattr(worker, "train", failing_train)
    assert worker.main() == 1
    captured = capsys.readouterr()
    assert "PRIVATE" not in captured.out + captured.err
    assert not captured.err
    event = json.loads(captured.out)
    assert event["event"] == "worker_failed" and event["error_type"] == "RuntimeError"
    assert event["frames"][-1]["function"] == "failing_train"
    assert all(set(frame) == {"filename", "line", "function"} for frame in event["frames"])
    assert all(type(frame["line"]) is int and frame["line"] > 0 for frame in event["frames"])
    assert all("/" not in frame["filename"] for frame in event["frames"])


def test_one_shot_main_preserves_existing_exception_behavior(tmp_path, monkeypatch, capsys):
    import lucifer.train_worker as worker
    path = tmp_path / "configuration.json"
    path.write_text("{}")
    monkeypatch.setattr(sys, "argv", ["worker", "--config", str(path)])
    def failing_train(config):
        raise RuntimeError("one-shot diagnostic")
    monkeypatch.setattr(worker, "train", failing_train)
    with pytest.raises(RuntimeError, match="one-shot diagnostic"):
        worker.main()
    assert not capsys.readouterr().out
