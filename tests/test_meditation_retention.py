import copy

import pytest

from lucifer.meditation_retention import CONTRACT, retention_receipt, validate_receipt


def measured(mean=0.001, maximum=0.002):
    return {"head_sha256": "candidate", "reference_head_sha256": "baseline", "datasets": {"development": {
        "tokens": 32, "loss": 0.5, "forward_kl": {"tokens": 32, "mean": mean, "maximum": maximum,
        "direction": "untrained_bf16_reference_to_candidate"}}}}


def receipt(value):
    return retention_receipt(value, contract=CONTRACT, head_sha256="candidate", reference_sha256="baseline")


def test_loss_improvement_does_not_override_distribution_drift_or_missing_evidence():
    good = receipt(measured())
    assert validate_receipt(good, head_sha256="candidate", reference_sha256="baseline")
    for value in (measured(0.01, 0.01), measured(0.001, 0.03)):
        bad = receipt(value)
        assert not bad["passed"]
        with pytest.raises(ValueError, match="passing distribution"):
            validate_receipt(bad, head_sha256="candidate", reference_sha256="baseline")
    missing = measured()
    del missing["datasets"]["development"]["forward_kl"]
    with pytest.raises(ValueError, match="complete"):
        receipt(missing)
    tampered = copy.deepcopy(good)
    tampered["heldout_used"] = True
    with pytest.raises(ValueError):
        validate_receipt(tampered, head_sha256="candidate", reference_sha256="baseline")
    with pytest.raises(ValueError, match="another candidate"):
        validate_receipt(good, head_sha256="different", reference_sha256="baseline")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.001])
def test_invalid_or_nonfinite_divergence_never_passes(value):
    with pytest.raises(ValueError, match="finite"):
        receipt(measured(value))
