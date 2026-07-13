from __future__ import annotations

import numpy as np

from .arm_sim import ArmPose, clip_scalar


def ik_pose_for_canvas_point(x: float, z: float, depth: float, elbow_up: bool = True) -> ArmPose:
    """Conventional 2-link IK that realizes a Cartesian canvas contact target."""
    l1 = 13.0
    l2 = 13.0
    yaw = np.rad2deg(np.arctan2(-x, depth))
    radial = float(np.hypot(x, depth))
    dist2 = radial * radial + z * z
    cos_elbow = clip_scalar((dist2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2), -1.0, 1.0)
    elbow = float(np.arccos(cos_elbow))
    if not elbow_up:
        elbow = -elbow
    pitch = float(np.arctan2(z, radial) - np.arctan2(l2 * np.sin(elbow), l1 + l2 * np.cos(elbow)))
    return ArmPose(
        yaw=clip_scalar(yaw, -90.0, 90.0),
        pitch=clip_scalar(np.rad2deg(pitch), -90.0, 90.0),
        roll=0.0,
        elbow=clip_scalar(np.rad2deg(elbow), 0.0, 150.0),
    )


def scripted_pose(t: float) -> ArmPose:
    """Conventional IK stroke script with lift, press, paint, and lift phases."""
    strokes = (
        ((-7.2, -4.5), (6.5, 2.8)),
        ((-5.4, 4.7), (7.2, -2.8)),
        ((-7.8, 0.2), (5.8, 5.9)),
        ((-2.8, -6.6), (3.8, 6.8)),
        ((7.4, 4.8), (-6.8, -1.6)),
    )
    cycle = 5.8
    index = int(t // cycle) % len(strokes)
    u = (t % cycle) / cycle
    start, end = strokes[index]

    if u < 0.16:
        p = u / 0.16
        x = start[0]
        z = start[1]
        depth = 16.25 + 0.45 * p
    elif u < 0.26:
        p = (u - 0.16) / 0.10
        x = start[0]
        z = start[1]
        depth = 16.70 + 0.42 * p
    elif u < 0.84:
        p = (u - 0.26) / 0.58
        ease = p * p * (3.0 - 2.0 * p)
        x = (1.0 - ease) * start[0] + ease * end[0]
        z = (1.0 - ease) * start[1] + ease * end[1]
        pressure = 0.10 + 0.34 * np.sin(np.pi * p) ** 2 + 0.08 * np.sin(5.0 * np.pi * p + index)
        depth = 17.0
    else:
        p = (u - 0.84) / 0.16
        x = end[0]
        z = end[1]
        depth = 17.10 - 0.85 * p
    return ik_pose_for_canvas_point(x, z, depth)


def scripted_contact_pressure(t: float) -> float:
    """Fallback contact intent for the non-agent scripted demo path."""
    cycle = 5.8
    u = (t % cycle) / cycle
    index = int(t // cycle) % 5
    if u < 0.16:
        return 0.0
    if u < 0.26:
        p = (u - 0.16) / 0.10
        return float(0.36 * p * p * (3.0 - 2.0 * p))
    if u < 0.84:
        p = (u - 0.26) / 0.58
        pressure = 0.10 + 0.34 * np.sin(np.pi * p) ** 2 + 0.08 * np.sin(5.0 * np.pi * p + index)
        return clip_scalar(pressure, 0.03, 0.52)
    p = (u - 0.84) / 0.16
    return float(max(0.0, 0.36 * (1.0 - p)))
