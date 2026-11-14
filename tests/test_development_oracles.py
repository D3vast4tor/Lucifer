"""Independent checks for widened practice spaces; frozen v1 probes stay intact."""
from fractions import Fraction
import itertools

import pytest

from lucifer.evaluation import generate_training_variants, validate_training_variant


QUESTIONS = {
    "binary_strings": "Count binary strings with no adjacent ones.",
    "markov_hitting": "Compute a Markov chain hitting time.",
    "coin_runs": "Find a waiting time for consecutive heads in coin tosses.",
    "snapshot_guard": "Analyze snapshot isolation and shared guard rows.",
}


def rows(family):
    parent = {"fingerprint": family, "source_group_id": family, "conversation_id": family,
              "source_message_id": family + "u", "messages": [{"role": "user", "content": QUESTIONS[family]},
                                                               {"role": "assistant", "content": "Reviewed final."}]}
    return generate_training_variants([parent], maximum=8, cursors={}, partition_version=2, target_families=[family])


def count_strings(n, k, circular):
    # Enumerate state transitions instead of using the generator's binomial formula.
    result = 0
    for first in (0, 1):
        states = {(first, first): 1}
        for _ in range(n - 1):
            following = {}
            for (last, ones), ways in states.items():
                for bit in (0, 1):
                    if last and bit or ones + bit > k:
                        continue
                    key = bit, ones + bit
                    following[key] = following.get(key, 0) + ways
            states = following
        result += sum(ways for (last, ones), ways in states.items()
                      if ones == k and not (circular and first and last))
    return result


def test_widened_binary_oracle_matches_independent_state_enumeration():
    generated = rows("binary_strings")
    assert len(generated) == 8
    assert any(row["validation"]["parameters"]["n"] > 25 for row in generated)
    for row in generated:
        n, k = (row["validation"]["parameters"][key] for key in ("n", "k"))
        assert row["validation"]["expected_json"] == {"linear": count_strings(n, k, False),
                                                       "circular": count_strings(n, k, True)}
        assert row["provenance"]["generator_version"] == "training_variant_v2"
        assert validate_training_variant(row)


def test_widened_markov_oracle_satisfies_exact_first_step_equations():
    generated = rows("markov_hitting")
    assert len(generated) == 8
    for row in generated:
        p = row["validation"]["parameters"]
        a, b, c = (Fraction(p[key]) for key in ("a", "b", "c"))
        expected = row["validation"]["expected_json"]
        first = Fraction(expected["numerator"], expected["denominator"])
        second = first - 1 / (1 - a)
        assert 0 < a < 1 and 0 < b + c < 1
        assert first == 1 + a * first + (1 - a) * second
        assert second == 1 + b * first + c * second
        assert validate_training_variant(row)


def test_widened_coin_oracle_matches_run_states_and_finite_outcome_enumeration():
    generated = rows("coin_runs")
    assert len(generated) == 8
    for row in generated:
        params, expected = row["validation"]["parameters"], row["validation"]["expected_json"]
        p, run, horizon = Fraction(params["probability"]), params["run"], params["horizon"]
        waiting = Fraction(expected["expected_numerator"], expected["expected_denominator"])
        tail = Fraction(0)
        for _ in range(run):
            tail = 1 + (1 - p) * waiting + p * tail
        assert tail == waiting  # Independent first-step system with the terminal streak at zero.
        occurrence = Fraction(0)
        for tosses in itertools.product((0, 1), repeat=horizon):
            if any(all(tosses[start:start + run]) for start in range(horizon - run + 1)):
                occurrence += p ** sum(tosses) * (1 - p) ** (horizon - sum(tosses))
        assert occurrence == Fraction(expected["by_horizon_numerator"], expected["by_horizon_denominator"])
        assert validate_training_variant(row)


def test_widened_snapshot_oracle_matches_explicit_commit_simulation():
    generated = rows("snapshot_guard")
    assert len(generated) == 8
    for row in generated:
        params = row["validation"]["parameters"]
        active = [True] * params["replicas"]
        committed = aborted = 0
        for index in range(len(active)):
            if params["mode"] == "shared_guard_write" and committed:
                aborted += 1
                continue
            visible = [True] * len(active) if params["mode"] == "snapshot" else active
            if any(value for other, value in enumerate(visible) if other != index):
                active[index] = False
                committed += 1
        assert row["validation"]["expected_json"] == {"active_rows": sum(active), "disabling_commits": committed,
            "aborted": aborted, "invariant_preserved": any(active)}
        assert validate_training_variant(row)


@pytest.mark.parametrize("family", QUESTIONS)
def test_legacy_training_version_remains_the_default(family):
    parent = {"fingerprint": family, "source_group_id": family, "conversation_id": family,
              "source_message_id": family + "u", "messages": [{"role": "user", "content": QUESTIONS[family]}]}
    legacy = generate_training_variants([parent], maximum=1)
    assert legacy[0]["provenance"]["generator_version"] == "training_variant_v1"
    assert validate_training_variant(legacy[0])
