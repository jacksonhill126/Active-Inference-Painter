from __future__ import annotations

import numpy as np
import torch

from .config import PainterConfig
from .env import StrokeAction
from .policies import MotorPrimitiveLatent


BASE_STROKE_ACTION_DIM = 7
BASE_SPATIAL_ACTION_CHANNELS = 6
DEFAULT_MOTOR_KINDS = (
    "cartesian_ik",
    "joint_spline",
    "elbow_pivot",
    "upper_arm_roll_positive",
    "upper_arm_roll_negative",
)


def motor_kind(motor_primitive: MotorPrimitiveLatent | None) -> str:
    return "cartesian_ik" if motor_primitive is None else str(motor_primitive.kind or "cartesian_ik")


def motor_kind_labels(config: PainterConfig | None = None) -> tuple[str, ...]:
    if config is None:
        return DEFAULT_MOTOR_KINDS
    return tuple(config.motor_realization_kinds) or DEFAULT_MOTOR_KINDS


def motor_condition_vector(
    config: PainterConfig,
    motor_primitive: MotorPrimitiveLatent | None = None,
    *,
    dim: int | None = None,
) -> np.ndarray:
    """One-hot motor realization condition for transition likelihoods."""

    size = max(0, int(config.action_dim - BASE_STROKE_ACTION_DIM if dim is None else dim))
    vector = np.zeros(size, dtype=np.float32)
    if size == 0:
        return vector
    labels = motor_kind_labels(config)[:size]
    kind = motor_kind(motor_primitive)
    if kind in labels:
        vector[labels.index(kind)] = 1.0
    elif "cartesian_ik" in labels:
        vector[labels.index("cartesian_ik")] = 1.0
    return vector


def encoded_action_vector(
    action: StrokeAction,
    config: PainterConfig,
    motor_primitive: MotorPrimitiveLatent | None = None,
) -> np.ndarray:
    """Summary-model action vector p(s_next | s, stroke, motor_realization)."""

    if action.stop:
        return np.zeros(int(config.action_dim), dtype=np.float32)
    base = action.vector()
    if config.action_dim <= BASE_STROKE_ACTION_DIM:
        return base[: config.action_dim].astype(np.float32)
    return np.concatenate(
        [base, motor_condition_vector(config, motor_primitive)],
        axis=0,
    ).astype(np.float32)


def coerce_action_tensor(action: torch.Tensor, action_dim: int) -> torch.Tensor:
    """Pad legacy 7-D actions to the model's configured action dimension."""

    current = int(action.shape[-1])
    target = int(action_dim)
    if current == target:
        return action
    if current > target:
        return action[..., :target]
    padding_shape = (*action.shape[:-1], target - current)
    padding = torch.zeros(padding_shape, device=action.device, dtype=action.dtype)
    return torch.cat([action, padding], dim=-1)


def coerce_action_raster(action_raster: torch.Tensor, action_channels: int) -> torch.Tensor:
    """Pad legacy 6-channel spatial actions to configured motor-conditioned channels."""

    current = int(action_raster.shape[1])
    target = int(action_channels)
    if current == target:
        return action_raster
    if current > target:
        return action_raster[:, :target]
    padding_shape = (
        action_raster.shape[0],
        target - current,
        *action_raster.shape[2:],
    )
    padding = torch.zeros(padding_shape, device=action_raster.device, dtype=action_raster.dtype)
    return torch.cat([action_raster, padding], dim=1)


def motor_condition_raster(
    grid_size: int,
    channel_count: int,
    config: PainterConfig | None = None,
    motor_primitive: MotorPrimitiveLatent | None = None,
    *,
    stop: bool = False,
) -> np.ndarray:
    """Constant spatial action-conditioning fields for motor realization."""

    channels = max(0, int(channel_count))
    raster = np.zeros((channels, grid_size, grid_size), dtype=np.float32)
    if stop or channels == 0:
        return raster
    labels = motor_kind_labels(config)[:channels]
    kind = motor_kind(motor_primitive)
    if kind in labels:
        raster[labels.index(kind), :, :] = 1.0
    elif "cartesian_ik" in labels:
        raster[labels.index("cartesian_ik"), :, :] = 1.0
    return raster
