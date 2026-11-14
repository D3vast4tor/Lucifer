"""Pure adaptive-round policy and committed, hash-bound training checkpoints.

Adaptive epochs are budget regimes, not data passes. These helpers never train,
load a model, change admission gates, or activate an adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile


STATE_VERSION = 2
CHECKPOINT_VERSION = 1
MANIFEST_NAME = "checkpoint_manifest.json"
REFRESH_INTERVAL = 50


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _tuples(value):
    return tuple(_tuples(item) for item in value) if isinstance(value, (list, tuple)) else value


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    """Publish a complete JSON file last; fsync its bytes and parent directory."""
    path = Path(path)
    raw = canonical_json(value) + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return hashlib.sha256(raw).hexdigest()


@dataclass
class AdaptiveState:
    adaptive_epoch: int
    round_budget: int
    round_remaining: int
    global_successful_updates: int
    next_refresh_at: int
    round_index: int
    rng_state: list
    dataset_passes_completed: int = 0
    randint_min: int = 1
    randint_max: int = 7

    @classmethod
    def initial(cls, seed=None):
        rng = random.Random(seed)
        return cls(1, 50, 50, 0, 50, 1, json.loads(json.dumps(rng.getstate())))

    @classmethod
    def from_dict(cls, payload):
        if (not isinstance(payload, dict) or type(payload.get("schema_version")) is not int
                or payload.get("schema_version") not in {1, STATE_VERSION}):
            raise ValueError("Unsupported adaptive state schema")
        payload = dict(payload)
        if payload["schema_version"] == 1:
            # Old fixed-range states continue from their documented 1..7 range.
            payload.setdefault("randint_min", 1)
            payload.setdefault("randint_max", 7)
        names = cls.__dataclass_fields__
        if set(payload) != {"schema_version", *names}:
            raise ValueError("Adaptive state fields changed")
        state = cls(**{key: payload[key] for key in names})
        state.validate()
        # Do not retain mutable references to a caller's RNG state.
        state.rng_state = json.loads(json.dumps(state.rng_state, allow_nan=False))
        return state

    def validate(self):
        for name in ("adaptive_epoch", "round_index", "randint_min", "randint_max"):
            _integer(getattr(self, name), name, 1)
        for name in ("round_budget", "round_remaining", "global_successful_updates", "next_refresh_at", "dataset_passes_completed"):
            _integer(getattr(self, name), name)
        if not 11 <= self.round_budget <= 89 or not 0 <= self.round_remaining <= self.round_budget:
            raise ValueError("Adaptive round budget must be 11..89 and remaining cannot exceed it")
        if self.randint_max - self.randint_min != 6:
            raise ValueError("Adaptive randint range must retain its seven-integer width")
        distance = self.next_refresh_at - self.global_successful_updates
        if self.next_refresh_at < 50 or self.next_refresh_at % 50 or not 0 <= distance <= 50:
            raise ValueError("Global refresh boundary changed or was skipped")
        try:
            rng = random.Random(0)
            rng.setstate(_tuples(self.rng_state))
        except (ValueError, TypeError, OverflowError):
            raise ValueError("Invalid adaptive RNG state") from None

    def to_dict(self):
        self.validate()
        return json.loads(json.dumps({"schema_version": STATE_VERSION,
                                      **{key: getattr(self, key) for key in self.__dataclass_fields__}}, allow_nan=False))

    def consume_updates(self, updates):
        self.validate()
        _integer(updates, "successful updates", 1)
        if updates > self.round_remaining or self.global_successful_updates + updates > self.next_refresh_at:
            raise ValueError("An adaptive round or refresh boundary cannot be crossed")
        self.round_remaining -= updates
        self.global_successful_updates += updates

    def consume_refresh(self):
        self.validate()
        if self.global_successful_updates != self.next_refresh_at:
            raise ValueError("Refresh may be consumed only at its global successful-update boundary")
        previous = self.next_refresh_at
        self.next_refresh_at += REFRESH_INTERVAL
        return {"refresh_at": previous, "next_refresh_at": self.next_refresh_at}

    def adjust(self, comparison, same_dev_revision=True):
        """Advance a completed round once, after the controller checks admission."""
        self.validate()
        if type(comparison) is not int or comparison not in {-1, 0, 1}:
            raise ValueError("Comparison must be -1, 0 or 1")
        if same_dev_revision is not True:
            raise ValueError("Adaptive comparison requires the same dev revision")
        if self.round_remaining:
            raise ValueError("Cannot adjust an unfinished adaptive round")
        before, epoch_before = self.round_budget, self.adaptive_epoch
        minimum_before, maximum_before = self.randint_min, self.randint_max
        delta = 0
        if comparison:
            rng = random.Random(0)
            rng.setstate(_tuples(self.rng_state))
            delta = 3 + rng.randint(self.randint_min, self.randint_max)
            self.rng_state = json.loads(json.dumps(rng.getstate()))
        proposed = before + comparison * delta
        reset = proposed <= 10 or proposed >= 90
        if reset:
            self.adaptive_epoch += 1
            self.randint_min, self.randint_max = 1, 7
        else:
            # Every completed round grows the next range, including ties. A tie
            # advances these bounds while leaving the RNG state untouched.
            self.randint_min += 1
            self.randint_max += 1
        self.round_budget = 50 if reset else proposed
        self.round_remaining = self.round_budget
        self.round_index += 1
        return {"comparison": comparison, "delta": delta, "proposed_budget": proposed,
                "budget_before": before, "round_budget": self.round_budget, "budget": self.round_budget,
                "epoch_before": epoch_before, "adaptive_epoch": self.adaptive_epoch,
                "threshold_reset": reset, "round_index": self.round_index,
                "randint_min_before": minimum_before, "randint_max_before": maximum_before,
                "randint_min": self.randint_min, "randint_max": self.randint_max}


def _required_checkpoint_files(manifest):
    required = {"adapter_config.json", "adapter_model.safetensors", "training_metrics.json",
                "trainer_state.json", "optimizer.pt", "scheduler.pt"}
    world_size = _integer(manifest.get("world_size"), "checkpoint world size", 1)
    required.update({"rng_state.pth"} if world_size == 1 else {f"rng_state_{rank}.pth" for rank in range(world_size)})
    if manifest.get("scaler_required"):
        required.add("scaler.pt")
    return required


def commit_checkpoint(checkpoint, metadata):
    """Seal native Trainer files after every worker rank has finished saving."""
    checkpoint = Path(checkpoint)
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("Checkpoint must be an owned regular directory")
    if (checkpoint / MANIFEST_NAME).exists():
        raise ValueError("A committed checkpoint is immutable")
    manifest = {**metadata, "schema_version": CHECKPOINT_VERSION}
    AdaptiveState.from_dict(manifest["state"])
    files = {}
    for path in sorted(checkpoint.rglob("*")):
        if path.is_symlink():
            raise ValueError("Checkpoint symlinks are forbidden")
        if not path.is_file():
            continue
        relative = path.relative_to(checkpoint).as_posix()
        files[relative] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    if not _required_checkpoint_files(manifest) <= files.keys():
        raise ValueError("Native Trainer checkpoint is incomplete")
    for directory in sorted((path for path in checkpoint.rglob("*") if path.is_dir()), reverse=True):
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    manifest["files"] = files
    digest = atomic_json(checkpoint / MANIFEST_NAME, manifest)
    return digest


def verify_checkpoint(checkpoint, expected_sha256=None):
    checkpoint = Path(checkpoint)
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("Checkpoint must be a regular directory")
    manifest_path = checkpoint / MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Checkpoint has no committed manifest")
    if expected_sha256 is not None and file_sha256(manifest_path) != expected_sha256:
        raise ValueError("Checkpoint manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint schema")
    AdaptiveState.from_dict(manifest["state"])
    files = manifest.get("files")
    if not isinstance(files, dict) or not _required_checkpoint_files(manifest) <= files.keys():
        raise ValueError("Native Trainer checkpoint is incomplete")
    for relative, proof in files.items():
        if not isinstance(relative, str) or not isinstance(proof, dict):
            raise ValueError("Invalid checkpoint file proof")
        path = checkpoint / relative
        if (not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts
                or any(parent.is_symlink() for parent in (path, *path.parents) if parent != checkpoint.parent)
                or not path.is_file()):
            raise ValueError("Checkpoint file escaped its immutable directory")
        if path.stat().st_size != proof.get("bytes") or file_sha256(path) != proof.get("sha256"):
            raise ValueError("Checkpoint file changed")
    inventory = list(checkpoint.rglob("*"))
    if any(path.is_symlink() for path in inventory):
        raise ValueError("Checkpoint symlinks are forbidden")
    actual = {path.relative_to(checkpoint).as_posix() for path in inventory if path.is_file()}
    if actual != {*files, MANIFEST_NAME}:
        raise ValueError("Checkpoint file inventory changed")
    return manifest


def prepare_segment_resume(checkpoint, destination, *, expected_sha256, data_revision, world_size, state):
    """Return native resume path and epoch; reset only the dataset cursor if needed.

    A private copied view preserves immutable committed files. Optimizer,
    scheduler and RNG state are retained; Trainer's old data skip offsets must
    never be applied to a new dataset revision or changed rank topology.
    """
    checkpoint, destination = Path(checkpoint), Path(destination)
    manifest = verify_checkpoint(checkpoint, expected_sha256)
    previous = AdaptiveState.from_dict(manifest["state"])
    if state.global_successful_updates != previous.global_successful_updates:
        raise ValueError("Resume successful-update counter changed")
    topology_changed = world_size != manifest["world_size"]
    if topology_changed and previous.round_remaining:
        raise ValueError("Rank topology may change only at a committed round boundary")
    changed = data_revision != manifest.get("data_revision") or topology_changed
    native = json.loads((checkpoint / "trainer_state.json").read_text())
    epoch = native.get("epoch") or 0
    if not isinstance(epoch, (int, float)) or not math.isfinite(epoch) or epoch < 0:
        raise ValueError("Invalid native Trainer data cursor")
    if not changed:
        return checkpoint, float(epoch), manifest, False
    destination.mkdir(parents=True, exist_ok=False)
    for relative in manifest["files"]:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint / relative, target)
    native.update(global_step=0, epoch=0.0)
    # Native Trainer saving expects its registered callback keys to exist even
    # when callback-state restoration is disabled. Retain those keys while
    # clearing the old segment's stop/save signals in this private resume view.
    callbacks = native.get("stateful_callbacks")
    if not isinstance(callbacks, dict):
        callbacks = {}
    control = callbacks.get("TrainerControl")
    if not isinstance(control, dict):
        control = {"args": {}, "attributes": {}}
    control_args = control.get("args")
    if not isinstance(control_args, dict):
        control_args = {}
    for flag in ("should_training_stop", "should_epoch_stop", "should_save", "should_evaluate", "should_log"):
        control_args[flag] = False
    control["args"] = control_args
    callbacks["TrainerControl"] = control
    native["stateful_callbacks"] = callbacks
    atomic_json(destination / "trainer_state.json", native)
    if topology_changed:
        # A changed world cannot reuse per-rank RNG filenames. Rank0's committed
        # random state seeds each new rank deterministically; data_seed and the
        # new distributed sampler separate their examples. No old rank cursor is reused.
        source_rng = checkpoint / ("rng_state.pth" if manifest["world_size"] == 1 else "rng_state_0.pth")
        for path in destination.glob("rng_state*.pth"):
            path.unlink()
        names = ["rng_state.pth"] if world_size == 1 else [f"rng_state_{rank}.pth" for rank in range(world_size)]
        for name in names:
            shutil.copy2(source_rng, destination / name)
    return destination, 0.0, manifest, True
