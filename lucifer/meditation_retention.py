"""Fixed development-only distribution drift contract and bound receipts."""
from __future__ import annotations

import math


CONTRACT = {"version": 1, "maximum_mean_forward_kl": 0.005, "maximum_token_forward_kl": 0.02}


def retention_receipt(measured, *, contract, head_sha256, reference_sha256):
    if contract != CONTRACT:
        raise ValueError("Unknown or changed head distribution retention contract")
    if measured.get("head_sha256") != head_sha256 or measured.get("reference_head_sha256") != reference_sha256:
        raise ValueError("Distribution retention measured another candidate or reference head")
    development = measured.get("datasets", {}).get("development", {})
    kl = development.get("forward_kl") or {}
    if (type(kl.get("tokens")) is not int or kl["tokens"] <= 0
            or kl.get("tokens") != development.get("tokens")
            or kl.get("direction") != "untrained_bf16_reference_to_candidate"):
        raise ValueError("Missing complete development distribution retention evidence")
    for key in ("mean", "maximum"):
        value = kl.get(key)
        if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid finite distribution retention measurement")
    if kl["mean"] > kl["maximum"] + 1e-12:
        raise ValueError("Distribution retention mean exceeds its maximum")
    return {"contract": dict(contract), "head_sha256": head_sha256, "reference_head_sha256": reference_sha256,
            "tokens": kl["tokens"], "mean_forward_kl": kl["mean"], "maximum_token_forward_kl": kl["maximum"],
            "passed": kl["mean"] <= contract["maximum_mean_forward_kl"]
                      and kl["maximum"] <= contract["maximum_token_forward_kl"],
            "scope": "fixed_development_target_features_only", "heldout_used": False}


def validate_receipt(receipt, *, head_sha256, reference_sha256):
    if not isinstance(receipt, dict):
        raise ValueError("Missing bound distribution retention receipt")
    measured = {"head_sha256": receipt.get("head_sha256"),
                "reference_head_sha256": receipt.get("reference_head_sha256"),
                "datasets": {"development": {"tokens": receipt.get("tokens"), "forward_kl": {
                    "tokens": receipt.get("tokens"), "mean": receipt.get("mean_forward_kl"),
                    "maximum": receipt.get("maximum_token_forward_kl"),
                    "direction": "untrained_bf16_reference_to_candidate"}}}}
    expected = retention_receipt(measured, contract=receipt.get("contract"),
        head_sha256=head_sha256, reference_sha256=reference_sha256)
    if expected != receipt or expected["passed"] is not True:
        raise ValueError("Candidate has no matching passing distribution retention evidence")
    return True
