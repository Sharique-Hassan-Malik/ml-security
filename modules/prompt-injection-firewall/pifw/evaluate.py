"""Scoring the detector, including the number that decides whether it is usable.

Precision and recall on a balanced corpus are the easy half. The half that
decides deployability is **base rate**: if 0.5% of traffic is an attack, a
detector with 90% recall and 90% precision *on a balanced set* flags roughly
eleven false positives for every true one, because there are two hundred times
more benign requests to be wrong about.

`operating_point()` computes that explicitly, so the choice of threshold is made
against the traffic mix the firewall will actually see rather than against the
corpus it was tuned on. This is the same correction `drift-detector` (#7) needed
for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .corpus import Sample, all_samples
from .firewall import Action, Firewall


@dataclass
class Confusion:
    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0

    @property
    def total(self) -> int:
        return self.true_positive + self.false_positive + self.true_negative + self.false_negative

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 1.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def false_positive_rate(self) -> float:
        denominator = self.false_positive + self.true_negative
        return self.false_positive / denominator if denominator else 0.0

    @property
    def accuracy(self) -> float:
        return (self.true_positive + self.true_negative) / self.total if self.total else 0.0


@dataclass
class Evaluation:
    threshold: float
    overall: Confusion
    by_technique: dict[str, Confusion] = field(default_factory=dict)
    misses: list[Sample] = field(default_factory=list)
    false_alarms: list[Sample] = field(default_factory=list)


def evaluate(firewall: Firewall, samples: list[Sample] | None = None,
             *, treat_flag_as_positive: bool = True) -> Evaluation:
    samples = samples if samples is not None else all_samples()
    overall = Confusion()
    by_technique: dict[str, Confusion] = {}
    misses: list[Sample] = []
    false_alarms: list[Sample] = []

    for sample in samples:
        decision = firewall.check(sample.text, source=sample.channel)
        positive_actions = (Action.BLOCK, Action.FLAG) if treat_flag_as_positive else (Action.BLOCK,)
        predicted = decision.action in positive_actions
        bucket = by_technique.setdefault(sample.technique, Confusion())

        if sample.label == 1 and predicted:
            overall.true_positive += 1
            bucket.true_positive += 1
        elif sample.label == 1:
            overall.false_negative += 1
            bucket.false_negative += 1
            misses.append(sample)
        elif predicted:
            overall.false_positive += 1
            bucket.false_positive += 1
            false_alarms.append(sample)
        else:
            overall.true_negative += 1
            bucket.true_negative += 1

    return Evaluation(firewall.block_threshold, overall, by_technique, misses, false_alarms)


def sweep(thresholds: list[float], samples: list[Sample] | None = None) -> list[Evaluation]:
    out = []
    for threshold in thresholds:
        firewall = Firewall(block_threshold=threshold, flag_threshold=threshold * 0.5)
        out.append(evaluate(firewall, samples))
    return out


@dataclass
class OperatingPoint:
    base_rate: float
    recall: float
    false_positive_rate: float
    requests_per_day: int

    @property
    def true_positives_per_day(self) -> float:
        return self.requests_per_day * self.base_rate * self.recall

    @property
    def false_positives_per_day(self) -> float:
        return self.requests_per_day * (1 - self.base_rate) * self.false_positive_rate

    @property
    def precision_in_production(self) -> float:
        """Precision at the real base rate, which is the only one that matters."""
        total = self.true_positives_per_day + self.false_positives_per_day
        return self.true_positives_per_day / total if total else 1.0

    @property
    def alerts_per_true_attack(self) -> float:
        if self.true_positives_per_day <= 0:
            return float("inf")
        return (self.true_positives_per_day + self.false_positives_per_day) / self.true_positives_per_day


def operating_point(evaluation: Evaluation, base_rate: float,
                    requests_per_day: int = 1_000_000) -> OperatingPoint:
    """Project corpus performance onto real traffic.

    The corpus is roughly balanced; production is not. A detector that looks
    excellent at 50% prevalence can be unusable at 0.5%, and that is a property
    of arithmetic rather than of the detector.
    """
    return OperatingPoint(
        base_rate=base_rate,
        recall=evaluation.overall.recall,
        false_positive_rate=evaluation.overall.false_positive_rate,
        requests_per_day=requests_per_day,
    )
