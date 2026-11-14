"""Durable development-only control of finite native-head training segments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass
class MeditationPolicy:
    schema_version: int = 1
    learning_rate: float = 1e-5
    minimum_rate: float = 1e-8
    maximum_rate: float = 1e-4
    batch_size: int = 32
    accumulation: int = 4
    segment_updates: int = 25
    successful_updates: int = 0
    segment: int = 0
    best_development_loss: float | None = None
    plateau_segments: int = 0
    admission_attempts: int = 0

    def validate(self):
        if self.schema_version != 1:
            raise ValueError("Unsupported Meditation policy version")
        for name in ("learning_rate", "minimum_rate", "maximum_rate"):
            value = getattr(self, name)
            if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
                raise ValueError("Meditation learning rates must be finite and positive")
        if not self.minimum_rate <= self.learning_rate <= self.maximum_rate:
            raise ValueError("Meditation learning rate exceeds its pinned range")
        for name in ("batch_size", "accumulation", "segment_updates", "successful_updates", "segment",
                     "plateau_segments", "admission_attempts"):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name in {"batch_size", "accumulation", "segment_updates"} else 0):
                raise ValueError("Invalid Meditation optimizer counter")
        if not 10 <= self.segment_updates <= 100:
            raise ValueError("Meditation segment must contain 10..100 successful updates")
        if self.best_development_loss is not None and not math.isfinite(self.best_development_loss):
            raise ValueError("Nonfinite development loss")

    def snapshot(self):
        self.validate()
        return asdict(self)

    @classmethod
    def restore(cls, value):
        if set(value) != set(cls.__dataclass_fields__):
            raise ValueError("Meditation policy checkpoint fields changed")
        policy = cls(**value)
        policy.validate()
        return policy

    def observe(self, *, development_loss, successful_updates, retention_passed, finite=True):
        """Admission scores are deliberately not an input to parameter selection."""
        self.validate()
        if type(successful_updates) is not int or not 0 <= successful_updates <= self.segment_updates:
            raise ValueError("Worker successful updates crossed a segment boundary")
        self.successful_updates += successful_updates
        self.segment += 1
        good = finite and math.isfinite(development_loss) and retention_passed
        previous = self.best_development_loss
        meaningful = good and (previous is None or development_loss < previous - max(1e-5, abs(previous) * 1e-4))
        regression = not good or previous is not None and development_loss > previous * 1.01 + 1e-5
        if meaningful:
            self.best_development_loss = development_loss
            self.plateau_segments = 0
            self.segment_updates = min(100, max(10, math.ceil(self.segment_updates * 1.25)))
            action, reason = "keep", "Measured development improvement"
        elif regression:
            self.learning_rate = max(self.minimum_rate, self.learning_rate * 0.5)
            self.segment_updates = max(10, self.segment_updates // 2)
            self.plateau_segments += 1
            action, reason = "restore_best", "Development regression or invalid numerical/retention evidence"
        else:
            self.plateau_segments += 1
            if self.plateau_segments % 2 == 0:
                self.learning_rate = max(self.minimum_rate, self.learning_rate * 0.5)
            action, reason = "refresh" if self.plateau_segments >= 3 else "keep", "Development plateau"
        return {"action": action, "reason": reason, "meaningful_improvement": meaningful,
                "refresh_required": self.plateau_segments >= 3,
                "state": self.snapshot()}

    def next_admission_alpha(self):
        self.admission_attempts += 1
        k = self.admission_attempts
        return 0.05 / (k * (k + 1))


def paired_improvement(differences, *, alpha, bound=20.0, minimum_groups=8, minimum_delta=0.001):
    """Exact one-sided sign test on fresh independent source-group loss deltas.

    Test whether the median group improves by more than the pinned delta.
    Positive delta means the candidate's clipped per-token loss is smaller;
    ties are conservatively counted against improvement. Require a positive
    average change as well; this does not claim a confidence bound on its mean.
    Fresh source groups must be reserved before each candidate measurement.
    """
    if not 0 < alpha < 1 or not math.isfinite(bound) or bound <= 0:
        raise ValueError("Invalid admission confidence parameters")
    values = list(differences)
    if any(type(x) not in {int, float} or not math.isfinite(x) or abs(x) > bound for x in values):
        raise ValueError("Admission requires finite bounded paired source-group measurements")
    if len(values) < minimum_groups:
        return {"passed": False, "status": "awaiting_evidence", "groups": len(values),
                "required_groups": minimum_groups, "alpha": alpha}
    mean = sum(values) / len(values)
    positive = sum(value > minimum_delta for value in values)
    # Exact binomial tail under P(group improvement > delta) <= 1/2.
    p_value = sum(math.comb(len(values), k) for k in range(positive, len(values) + 1)) / (2 ** len(values))
    return {"passed": p_value <= alpha and mean > minimum_delta, "status": "measured", "groups": len(values),
            "alpha": alpha, "mean_clipped_loss_improvement": mean, "positive_groups": positive,
            "p_value": p_value, "test": "one_sided_exact_group_sign_test",
            "minimum_delta": minimum_delta, "loss_clip": bound}
