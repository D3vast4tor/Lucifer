"""Adaptive budgets and immutable continuation checkpoints, without ML runtimes."""
import copy
import json
import random

import pytest

from lucifer.adaptive import AdaptiveState, commit_checkpoint, file_sha256, prepare_segment_resume, verify_checkpoint


def complete_round(state):
    while state.round_remaining:
        if state.global_successful_updates == state.next_refresh_at:
            state.consume_refresh()
        state.consume_updates(min(state.round_remaining, state.next_refresh_at - state.global_successful_updates))


def test_round_transition_persists_rng_and_ties_do_not_draw():
    state = AdaptiveState.initial(seed=14)
    reference = random.Random(14)
    complete_round(state)
    before_rng = copy.deepcopy(state.rng_state)
    tie = state.adjust(0)
    assert tie["delta"] == 0 and state.round_budget == 50 and state.rng_state == before_rng
    assert (state.randint_min, state.randint_max) == (2, 8)
    complete_round(state)
    restored = AdaptiveState.from_dict(state.to_dict())
    gain = restored.adjust(1)
    assert gain["delta"] == 3 + reference.randint(2, 8)
    assert restored.round_remaining == restored.round_budget == 50 + gain["delta"]
    assert state.adjust(1) == gain
    assert state.to_dict() == restored.to_dict()


@pytest.mark.parametrize("direction,threshold", [(-1, 10), (1, 90)])
def test_exact_threshold_both_directions_resets_epoch_and_budget(direction, threshold):
    state = AdaptiveState.initial(seed=8)
    delta = 3 + random.Random(8).randint(1, 7)
    state.round_budget = threshold - direction * delta
    state.round_remaining = 0
    decision = state.adjust(direction)
    assert decision["proposed_budget"] == threshold
    assert decision["threshold_reset"]
    assert state.adaptive_epoch == 2 and state.round_budget == state.round_remaining == 50
    assert (state.randint_min, state.randint_max) == (1, 7)


def test_thousand_rounds_keep_global_refresh_and_budget_invariants():
    state = AdaptiveState.initial(seed=39)
    for comparison in [1] * 300 + [-1] * 300 + [0, -1, 1, 1] * 100:
        complete_round(state)
        old_global, old_refresh = state.global_successful_updates, state.next_refresh_at
        state.adjust(comparison)
        assert 11 <= state.round_budget <= 89
        assert state.global_successful_updates == old_global and state.next_refresh_at == old_refresh
        assert state.round_remaining == state.round_budget
        state.validate()


def test_round_refresh_and_comparison_preconditions():
    state = AdaptiveState.initial(seed=1)
    with pytest.raises(ValueError, match="unfinished"):
        state.adjust(1)
    with pytest.raises(ValueError, match="Refresh may"):
        state.consume_refresh()
    with pytest.raises(ValueError, match="cannot be crossed"):
        state.consume_updates(51)
    state.consume_updates(50)
    with pytest.raises(ValueError, match="same dev"):
        state.adjust(1, same_dev_revision=False)
    with pytest.raises(ValueError, match="Comparison"):
        state.adjust(True)
    assert state.global_successful_updates == state.next_refresh_at == 50
    state.consume_refresh()
    assert state.next_refresh_at == 100 and state.round_remaining == 0


def checkpoint(tmp_path, *, state=None, world_size=1, revision="a" * 64):
    path = tmp_path / "checkpoint"
    path.mkdir()
    state = state or AdaptiveState.initial(seed=1)
    for name in ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "training_metrics.json"):
        (path / name).write_bytes(b"native-owned-state")
    (path / "trainer_state.json").write_text(json.dumps({"global_step": 5, "epoch": 0.25}))
    rng_names = ["rng_state.pth"] if world_size == 1 else [f"rng_state_{rank}.pth" for rank in range(world_size)]
    for name in rng_names:
        (path / name).write_bytes(b"owned-rng")
    digest = commit_checkpoint(path, {"state": state.to_dict(), "world_size": world_size,
                                      "data_revision": revision, "scaler_required": False})
    return path, digest


def test_checkpoint_requires_complete_state_and_detects_tamper(tmp_path):
    path, digest = checkpoint(tmp_path)
    assert verify_checkpoint(path, digest)["data_revision"] == "a" * 64
    with pytest.raises(ValueError, match="immutable"):
        commit_checkpoint(path, {})
    (path / "optimizer.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="file changed"):
        verify_checkpoint(path, digest)


def test_manifest_digest_inventory_and_symlinks_are_checked(tmp_path):
    path, digest = checkpoint(tmp_path)
    with pytest.raises(ValueError, match="manifest changed"):
        verify_checkpoint(path, "0" * 64)
    (path / "unexpected").write_bytes(b"extra")
    with pytest.raises(ValueError, match="inventory"):
        verify_checkpoint(path, digest)
    (path / "unexpected").unlink()
    (path / "optimizer.pt").unlink()
    (path / "optimizer.pt").symlink_to(path / "scheduler.pt")
    with pytest.raises(ValueError, match="escaped"):
        verify_checkpoint(path, digest)


def test_data_revision_reset_preserves_native_optimizer_and_original_cursor(tmp_path):
    state = AdaptiveState.initial(seed=3)
    state.consume_updates(5)
    path, digest = checkpoint(tmp_path, state=state)
    native, epoch, _, reset = prepare_segment_resume(path, tmp_path / "unused", expected_sha256=digest,
        data_revision="a" * 64, world_size=1, state=state)
    assert native == path and epoch == .25 and not reset
    view, epoch, _, reset = prepare_segment_resume(path, tmp_path / "new-view", expected_sha256=digest,
        data_revision="b" * 64, world_size=1, state=state)
    assert reset and epoch == 0
    assert json.loads((view / "trainer_state.json").read_text())["global_step"] == 0
    assert json.loads((path / "trainer_state.json").read_text())["global_step"] == 5
    assert file_sha256(view / "optimizer.pt") == file_sha256(path / "optimizer.pt")
    assert verify_checkpoint(path, digest)


def test_topology_change_requires_committed_round_boundary(tmp_path):
    state = AdaptiveState.initial(seed=2)
    state.consume_updates(5)
    path, digest = checkpoint(tmp_path, state=state)
    with pytest.raises(ValueError, match="round boundary"):
        prepare_segment_resume(path, tmp_path / "denied", expected_sha256=digest,
                               data_revision="a" * 64, world_size=2, state=state)


def test_topology_change_at_round_boundary_resets_sampler(tmp_path):
    state = AdaptiveState.initial(seed=2)
    state.consume_updates(50)
    path, digest = checkpoint(tmp_path, state=state, world_size=2)
    state.consume_refresh()
    state.adjust(1)
    view, epoch, _, reset = prepare_segment_resume(path, tmp_path / "new-topology", expected_sha256=digest,
        data_revision="a" * 64, world_size=1, state=state)
    assert reset and epoch == 0 and (view / "rng_state.pth").exists()
    assert not (view / "rng_state_1.pth").exists()
    assert state.global_successful_updates == 50


def test_checkpoint_manifest_missing_scheduler_is_not_committed(tmp_path):
    path, digest = checkpoint(tmp_path)
    (path / "checkpoint_manifest.json").unlink()
    (path / "scheduler.pt").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        commit_checkpoint(path, {"state": AdaptiveState.initial().to_dict(), "world_size": 1})
    assert not (path / "checkpoint_manifest.json").exists()


def test_every_round_grows_range_and_later_deltas_can_exceed_initial_ten():
    state = AdaptiveState.initial(seed=15)
    saved_rng = copy.deepcopy(state.rng_state)
    for _ in range(11):
        complete_round(state)
        decision = state.adjust(0)
        assert decision["delta"] == 0
    assert (state.randint_min, state.randint_max) == (12, 18)
    assert state.rng_state == saved_rng
    complete_round(state)
    decision = state.adjust(-1)
    assert 15 <= decision["delta"] <= 21
    assert (state.randint_min, state.randint_max) == (13, 19)


def test_midround_refresh_does_not_grow_range_or_change_rng():
    state = AdaptiveState.initial(seed=15)
    before = copy.deepcopy(state.rng_state)
    state.consume_updates(50)
    state.consume_refresh()
    assert (state.randint_min, state.randint_max) == (1, 7)
    assert state.rng_state == before


def test_grown_range_resets_on_both_budget_threshold_directions():
    for comparison in (-1, 1):
        state = AdaptiveState.initial(seed=18)
        state.randint_min, state.randint_max = 100, 106
        state.round_remaining = 0
        decision = state.adjust(comparison)
        assert decision["threshold_reset"]
        assert 103 <= decision["delta"] <= 109
        assert state.adaptive_epoch == 2 and state.round_budget == 50
        assert (state.randint_min, state.randint_max) == (1, 7)


def test_legacy_rng_state_migration_and_invalid_width():
    payload = AdaptiveState.initial(seed=4).to_dict()
    payload["schema_version"] = 1
    del payload["randint_min"], payload["randint_max"]
    restored = AdaptiveState.from_dict(payload)
    assert restored.to_dict()["schema_version"] == 2
    assert (restored.randint_min, restored.randint_max) == (1, 7)
    invalid = restored.to_dict()
    invalid["randint_max"] = 8
    with pytest.raises(ValueError, match="seven-integer"):
        AdaptiveState.from_dict(invalid)


def test_new_dataset_resume_retains_native_callback_save_keys_and_clears_stop_flags(tmp_path):
    state = AdaptiveState.initial(seed=4)
    state.consume_updates(5)
    path, _ = checkpoint(tmp_path, state=state)
    (path / "checkpoint_manifest.json").unlink()
    native_path = path / "trainer_state.json"
    native = json.loads(native_path.read_text())
    native["stateful_callbacks"] = {"TrainerControl": {"args": {
        "should_training_stop": True, "should_epoch_stop": True,
        "should_save": True, "should_evaluate": False, "should_log": False}, "attributes": {}}}
    native_path.write_text(json.dumps(native))
    manifest = commit_checkpoint(path, {"state": state.to_dict(), "world_size": 1,
                                       "data_revision": "a" * 64, "scaler_required": False})
    view, _, _, reset = prepare_segment_resume(path, tmp_path / "new-dataset", expected_sha256=manifest,
        data_revision="b" * 64, world_size=1, state=state)
    assert reset
    resumed = json.loads((view / "trainer_state.json").read_text())
    assert "TrainerControl" in resumed["stateful_callbacks"]
    assert not any(resumed["stateful_callbacks"]["TrainerControl"]["args"].values())
    assert json.loads(native_path.read_text())["stateful_callbacks"]["TrainerControl"]["args"]["should_training_stop"]
    assert verify_checkpoint(path, manifest)
