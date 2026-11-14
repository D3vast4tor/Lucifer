import pytest

from lucifer.meditation_policy import MeditationPolicy, paired_improvement
from lucifer.model_compatibility import adapter_compatibility


def test_development_updates_parameters_and_resume_without_admission_feedback():
    policy = MeditationPolicy()
    result = policy.observe(development_loss=2.0, successful_updates=25, retention_passed=True)
    assert result["action"] == "keep"
    assert policy.segment_updates == 32
    policy = MeditationPolicy.restore(policy.snapshot())
    result = policy.observe(development_loss=2.2, successful_updates=32, retention_passed=True)
    assert result["action"] == "restore_best"
    assert policy.learning_rate == 5e-6
    assert policy.successful_updates == 57
    assert policy.segment_updates == 16
    with pytest.raises(ValueError, match="boundary"):
        policy.observe(development_loss=1.0, successful_updates=17, retention_passed=True)


def test_repeated_regression_requests_fresh_evidence_without_a_global_update_limit():
    policy = MeditationPolicy(best_development_loss=1.0)
    for index in range(3):
        decision = policy.observe(development_loss=0.9, successful_updates=policy.segment_updates,
                                  retention_passed=False)
        assert decision["action"] == "restore_best"
        assert decision["refresh_required"] is (index == 2)
    # A meaningful valid development improvement clears the unproductive streak;
    # successful segments remain unlimited rather than hitting a run-wide cap.
    decision = policy.observe(development_loss=0.9, successful_updates=policy.segment_updates, retention_passed=True)
    assert not decision["refresh_required"] and policy.plateau_segments == 0
    for index in range(100):
        decision = policy.observe(development_loss=0.9 / (index + 2), successful_updates=policy.segment_updates,
                                  retention_passed=True)
        assert not decision["refresh_required"]
    assert policy.successful_updates > 9000


def test_admission_spends_error_budget_and_requires_independent_evidence():
    policy = MeditationPolicy()
    alphas = [policy.next_admission_alpha() for _ in range(1000)]
    assert sum(alphas) < 0.05
    assert paired_improvement([0.1] * 7, alpha=alphas[0])["status"] == "awaiting_evidence"
    result = paired_improvement([0.1] * 8, alpha=alphas[0])
    assert result["passed"]
    assert result["p_value"] == 1 / 256
    assert not paired_improvement([0.1, -0.2] * 4, alpha=alphas[0])["passed"]
    assert not paired_improvement([0.001] * 8, alpha=alphas[0])["passed"]
    with pytest.raises(ValueError):
        paired_improvement([float("nan")] * 8, alpha=0.025)


def test_native_revision_does_not_inherit_old_expert_compatibility():
    old = {"id": "base", "backend": "vllm", "base_revision": "before", "enable_lora": True}
    adapter = {"base_model_id": "base", "base_revision": "before", "format": "peft"}
    assert adapter_compatibility(old, adapter)["compatible"]
    assert not adapter_compatibility({**old, "base_revision": "after"}, adapter)["compatible"]
    assert not adapter_compatibility(old, {"base_model_id": "base"})["compatible"]
    assert not adapter_compatibility({**old, "backend": "strata"}, adapter)["compatible"]
    assert not adapter_compatibility(old, {**adapter, "source_revoked": True})["compatible"]
