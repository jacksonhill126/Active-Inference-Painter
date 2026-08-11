from __future__ import annotations

import argparse
import math
import time

import mujoco
import mujoco.viewer

from .wrist_roll_design import AngledWristRollSpec, build_wrist_roll_exploration_model


def demo_targets(time_s: float, sweep_degrees: float = 32.0) -> dict[str, float]:
    """Joint targets for a slow, unloaded mechanism-visibility demonstration."""

    phase = 2.0 * math.pi * float(time_s) / 5.0
    return {
        "yaw": math.radians(4.0 * math.sin(0.5 * phase)),
        "pitch": math.radians(-50.0 + 3.0 * math.sin(0.5 * phase)),
        "roll": math.radians(float(sweep_degrees) * math.sin(phase)),
        "elbow": math.radians(100.0 + 4.0 * math.cos(0.5 * phase)),
    }


def run_viewer(brush_angle_deg: float = 15.0, sweep_degrees: float = 32.0) -> None:
    spec = AngledWristRollSpec(brush_angle_deg=brush_angle_deg)
    model = build_wrist_roll_exploration_model(spec)
    data = mujoco.MjData(model)

    initial = demo_targets(0.0, sweep_degrees)
    for name, value in initial.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[int(model.jnt_qposadr[joint_id])] = value
        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            f"{name}_position",
        )
        data.ctrl[actuator_id] = value
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as window:
        window.cam.lookat[:] = (0.075, 0.31, 0.34)
        window.cam.distance = 0.78
        window.cam.azimuth = 150.0
        window.cam.elevation = -12.0
        window.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
        substeps_per_frame = max(1, int(round((1.0 / 90.0) / model.opt.timestep)))
        wall_start = time.perf_counter()
        while window.is_running():
            frame_start = time.perf_counter()
            targets = demo_targets(time.perf_counter() - wall_start, sweep_degrees)
            for name, value in targets.items():
                actuator_id = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    f"{name}_position",
                )
                data.ctrl[actuator_id] = value
            for _ in range(substeps_per_frame):
                mujoco.mj_step(model, data)
            window.sync()
            remaining = 1.0 / 90.0 - (time.perf_counter() - frame_start)
            if remaining > 0.0:
                time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "View the non-canonical angled wrist-roll mechanism. This is an "
            "unloaded design demonstration, not the active-inference painter."
        )
    )
    parser.add_argument("--brush-angle-deg", type=float, default=15.0)
    parser.add_argument("--sweep-deg", type=float, default=32.0)
    args = parser.parse_args()
    run_viewer(args.brush_angle_deg, args.sweep_deg)


if __name__ == "__main__":
    main()
