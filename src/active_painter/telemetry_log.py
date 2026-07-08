from __future__ import annotations

import csv
import io
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .arm_sim import ArmPainterSim, JOINT_NAMES


BASE_COLUMNS = (
    "sim_time_s",
    "painting_count",
    "phase",
    "agent_enabled",
    "paint_enabled",
    "contact_on_canvas",
    "contact_pressure",
    "contact_force_n",
    "canvas_coverage",
    "tip_x",
    "tip_y",
    "tip_z",
    "target_tip_x",
    "target_tip_y",
    "target_tip_z",
    "brush_x",
    "brush_y",
    "brush_z",
    "max_joint_target_error_deg",
)

JOINT_COLUMNS = tuple(
    column
    for name in JOINT_NAMES
    for column in (
        f"position_{name}_deg",
        f"target_{name}_deg",
        f"velocity_{name}_rad_s",
        f"velocity_{name}_deg_s",
        f"current_{name}_a",
        f"torque_{name}_nm",
        f"voltage_{name}_v",
    )
)

TELEMETRY_COLUMNS = BASE_COLUMNS + JOINT_COLUMNS


@dataclass(slots=True)
class ArmTelemetryLog:
    max_samples: int = 18_000
    _samples: deque[dict[str, Any]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._samples = deque(maxlen=max(1, int(self.max_samples)))

    def append_from_sim(
        self,
        sim_time: float,
        sim: ArmPainterSim,
        *,
        phase: str,
        painting_count: int,
        agent_enabled: bool,
    ) -> None:
        pose = sim.actual_pose
        target = sim.target_pose
        tip = sim.kinematics.tip(pose)
        target_tip = sim.kinematics.tip(target)
        telemetry = sim.plant.telemetry
        max_target_error = max(abs(float(getattr(target, name) - getattr(pose, name))) for name in JOINT_NAMES)
        row: dict[str, Any] = {
            "sim_time_s": float(sim_time),
            "painting_count": int(painting_count),
            "phase": str(phase),
            "agent_enabled": bool(agent_enabled),
            "paint_enabled": bool(sim.paint_enabled),
            "contact_on_canvas": bool(sim.contact.on_canvas),
            "contact_pressure": float(sim.contact.pressure),
            "contact_force_n": float(sim.contact.force),
            "canvas_coverage": float(sim.canvas.material_coverage()),
            "tip_x": float(tip[0]),
            "tip_y": float(tip[1]),
            "tip_z": float(tip[2]),
            "target_tip_x": float(target_tip[0]),
            "target_tip_y": float(target_tip[1]),
            "target_tip_z": float(target_tip[2]),
            "brush_x": float(sim.contact.brush_world[0]),
            "brush_y": float(sim.contact.brush_world[1]),
            "brush_z": float(sim.contact.brush_world[2]),
            "max_joint_target_error_deg": float(max_target_error),
        }
        for name in JOINT_NAMES:
            velocity = float(sim.plant.velocity[name])
            row[f"position_{name}_deg"] = float(getattr(pose, name))
            row[f"target_{name}_deg"] = float(getattr(target, name))
            row[f"velocity_{name}_rad_s"] = velocity
            row[f"velocity_{name}_deg_s"] = float(np.rad2deg(velocity))
            row[f"current_{name}_a"] = float(telemetry.current[name])
            row[f"torque_{name}_nm"] = float(telemetry.torque[name])
            row[f"voltage_{name}_v"] = float(telemetry.voltage[name])
        self._samples.append(row)

    def clear(self) -> None:
        self._samples.clear()

    def recent(self, limit: int = 240) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        return list(self._samples)[-limit:]

    def summary(self, sim_time: float) -> dict[str, Any]:
        first = self._samples[0]["sim_time_s"] if self._samples else None
        last = self._samples[-1]["sim_time_s"] if self._samples else None
        return {
            "sampleCount": len(self._samples),
            "maxSamples": self._samples.maxlen,
            "firstSampleTime": first,
            "lastSampleTime": last,
            "windowSeconds": float(last - first) if first is not None and last is not None else 0.0,
            "currentSimTime": float(sim_time),
            "csvEndpoint": "/api/telemetry.csv",
        }

    def to_csv(self) -> str:
        out = io.StringIO(newline="")
        writer = csv.DictWriter(out, fieldnames=TELEMETRY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(self._samples)
        return out.getvalue()
