"""Successful optimizer counters cannot be fabricated from native steps/text."""
import json

import pytest

from lucifer.adaptive import AdaptiveState
from lucifer.adaptive_progress import progress_events


def start_state():
    state = AdaptiveState.initial(seed=4)
    state.consume_updates(18)
    return state


def event(global_step=19, **changes):
    return {"event": "adaptive_progress", "segment_id": "owned-segment",
            "global_successful_updates": global_step, "segment_successful_updates": global_step - 18,
            "round_remaining": 50 - global_step, "randint_min": 1, "randint_max": 7,
            "next_refresh_at": 50, "native_epoch": 3.25, "optimizer_skipped": False, **changes}


def parse(line, *, last=18, state=None):
    return progress_events(line, segment_id="owned-segment", state=state or start_state(), last_successful=last)


def test_embedded_json_and_amp_skips_advance_only_successful_cumulative_steps():
    physical = "\r 25%|███|\x1b[0m" + json.dumps(event(19))
    physical += "\r " + json.dumps(event(19, optimizer_skipped=True, native_epoch=3.5))
    physical += "\r " + json.dumps(event(20, native_epoch=3.75))
    results = parse(physical)
    assert [row["cumulative_optimizer_steps"] for row in results] == [19, 19, 20]
    assert [row["optimizer_skipped"] for row in results] == [False, True, False]
    assert [row["round_remaining"] for row in results] == [31, 31, 30]
    assert all(row["optimizer_step"] == row["cumulative_optimizer_steps"] for row in results)
    assert all(row["committed_optimizer_steps"] == 18 for row in results)
    assert parse(json.dumps(event(21)), last=results[-1]["cumulative_optimizer_steps"])[0]["optimizer_step"] == 21


def test_native_trainer_steps_nested_diagnostics_and_unowned_events_do_not_advance():
    line = json.dumps({"event": "training_progress", "step": 900, "total_steps": 1000})
    line += json.dumps({"diagnostic": event(19)})
    line += json.dumps(event(19, segment_id="another-segment"))
    assert parse(line) == []


@pytest.mark.parametrize("changes", [
    {"global_successful_updates": 20}, {"global_successful_updates": 18},
    {"optimizer_skipped": True}, {"optimizer_skipped": 0},
    {"segment_successful_updates": 99}, {"round_remaining": 32},
    {"next_refresh_at": 100}, {"randint_min": 2}, {"randint_max": 8},
    {"native_epoch": float("nan")}, {"native_epoch": float("inf")}, {"native_epoch": -1},
    {"native_epoch": True}, {"global_successful_updates": True},
    {"segment_successful_updates": 1.0}, {"round_remaining": 31.0},
])
def test_invalid_counters_flags_bounds_or_epoch_are_ignored(changes):
    assert parse(json.dumps(event(**changes))) == []


def test_duplicate_keys_nonfinite_records_and_missing_fields_are_ignored():
    duplicate = json.dumps(event(19)).replace('"global_successful_updates": 19',
        '"global_successful_updates": 19, "global_successful_updates": 20')
    missing = event(19)
    del missing["segment_successful_updates"]
    assert parse(duplicate + json.dumps(missing)) == []
    # Recover after malformed braces/TQDM noise without retaining its text.
    assert parse("not json { private junk " + json.dumps(event(19)))[0]["optimizer_step"] == 19


def test_extra_text_fields_never_enter_clean_heartbeat():
    rows = parse(json.dumps(event(19, answer="PRIVATE_ANSWER", reasoning="PRIVATE_REASONING", loss="PRIVATE")))
    assert len(rows) == 1 and "PRIVATE" not in json.dumps(rows)
    assert set(rows[0]) == {"optimizer_step", "cumulative_optimizer_steps", "committed_optimizer_steps",
                            "round_remaining", "randint_min", "randint_max", "next_refresh_optimizer_step",
                            "native_epoch", "optimizer_skipped"}


def test_new_data_native_epoch_reset_does_not_reset_successful_updates():
    state = start_state()
    state.randint_min, state.randint_max = 4, 10
    state.round_budget, state.round_remaining = 57, 39
    row = event(19, native_epoch=.05, round_remaining=38, randint_min=4, randint_max=10)
    results = parse(json.dumps(row), state=state)
    assert results[0]["optimizer_step"] == 19
    assert results[0]["committed_optimizer_steps"] == 18
    assert results[0]["native_epoch"] == .05


def test_cannot_cross_refresh_or_round_boundary_and_initial_skip_is_honest():
    state = start_state()
    state.consume_updates(31)  # global49, remaining1
    valid = event(50, segment_successful_updates=1, round_remaining=0)
    invalid = event(51, segment_successful_updates=2, round_remaining=-1)
    rows = progress_events(json.dumps(valid) + json.dumps(invalid), segment_id="owned-segment", state=state,
                           last_successful=49)
    assert len(rows) == 1 and rows[0]["optimizer_step"] == 50
    skipped = event(18, optimizer_skipped=True, native_epoch=.1)
    assert parse(json.dumps(skipped))[0]["optimizer_step"] == 18


def test_dictionary_state_is_supported_and_invalid_previous_count_is_ignored():
    state = start_state()
    assert progress_events(json.dumps(event()), segment_id="owned-segment", state=state.to_dict(),
                           last_successful=18)[0]["optimizer_step"] == 19
    assert parse(json.dumps(event()), last=17) == []
    assert parse(json.dumps(event()), last=True) == []
