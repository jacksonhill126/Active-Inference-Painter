from __future__ import annotations

from dataclasses import dataclass

import active_painter.learning_curves as curves


@dataclass
class _Shard:
    trajectory_id: str
    labels: tuple[str, ...]

    @property
    def transition_count(self) -> int:
        return len(self.labels)


def test_fraction_count_is_nested_and_never_empty() -> None:
    assert curves.fraction_count(10, 0.01) == 1
    assert curves.fraction_count(10, 0.3) == 3
    assert curves.fraction_count(10, 0.6) == 6
    assert curves.fraction_count(10, 1.0) == 10


def test_nested_training_order_prioritizes_rare_condition_coverage(monkeypatch) -> None:
    shards = [
        _Shard("common-a", ("common", "common")),
        _Shard("rare", ("dynamic_roll",)),
        _Shard("common-b", ("common",)),
    ]

    def labels(shard: _Shard, index: int) -> dict[str, str]:
        return {"motor": shard.labels[index]}

    monkeypatch.setattr(curves, "transition_condition_labels", labels)
    ordered = curves.nested_training_order(shards)  # type: ignore[arg-type]

    assert ordered[0].trajectory_id == "rare"
    assert {item.trajectory_id for item in ordered} == {
        "common-a",
        "common-b",
        "rare",
    }


def test_diagnosis_requires_change_larger_than_seed_variation() -> None:
    def aggregate(family: str, axis: str, level: str, mean: float, std: float):
        return {
            "family": family,
            "axis": axis,
            "level": level,
            "train_nll": {"mean": mean - 0.1, "std": std},
            "test_nll": {"mean": mean, "std": std},
            "test_coverage_90": {"mean": 0.9, "std": 0.0},
        }

    diagnosis = curves.diagnose(
        [
            aggregate("cnn", "data", "small", -3.0, 0.02),
            aggregate("cnn", "data", "full", -3.2, 0.03),
            aggregate("mixture", "family", "full", -3.22, 0.04),
        ]
    )

    assert diagnosis["data_sensitivity"]["material_improvement"] is True
    assert diagnosis["mixture_family_vs_cnn"]["material_improvement"] is False
