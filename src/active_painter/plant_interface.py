from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, runtime_checkable

from .env import StrokeAction


PLANT_INTERFACE_VERSION = "plant-interface-v1"


def _validate_joint_vector(name: str, values: tuple[float, ...], joint_count: int) -> None:
    if len(values) != joint_count:
        raise ValueError(f"{name} must contain one value per declared joint.")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain only finite values.")


def _validate_optional_joint_vector(
    name: str,
    values: tuple[float, ...] | None,
    joint_count: int,
) -> None:
    if values is not None:
        _validate_joint_vector(name, values, joint_count)


@dataclass(frozen=True, slots=True)
class PlantCapabilities:
    """Static facts needed to configure a backend-neutral execution loop."""

    backend_id: str
    joint_names: tuple[str, ...]
    position_limits_rad: tuple[tuple[float, float], ...]
    command_modes: tuple[str, ...]
    physical_sensor_fields: tuple[str, ...]
    nominal_control_period_s: float
    supports_deterministic_step: bool
    supports_evaluation_truth: bool
    interface_version: str = PLANT_INTERFACE_VERSION

    def __post_init__(self) -> None:
        if not self.backend_id:
            raise ValueError("backend_id must be non-empty.")
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be non-empty and unique.")
        if len(self.position_limits_rad) != len(self.joint_names):
            raise ValueError("position_limits_rad must contain one pair per joint.")
        if any(not (math.isfinite(low) and math.isfinite(high) and low < high) for low, high in self.position_limits_rad):
            raise ValueError("Each position limit must be a finite increasing pair.")
        if self.nominal_control_period_s <= 0.0 or not math.isfinite(self.nominal_control_period_s):
            raise ValueError("nominal_control_period_s must be finite and positive.")


@dataclass(frozen=True, slots=True)
class ToolExecutionCommand:
    """Tool command produced after policy selection by the execution layer.

    Contact force and paint delivery are consequences needed to realize a
    selected mark intent. They are not globally preferred painting outcomes.
    """

    contact_enabled: bool = False
    deposition_enabled: bool = False
    paint_tone: float | None = None
    load_fraction: float | None = None
    normal_force_target_n: float | None = None

    def __post_init__(self) -> None:
        if self.paint_tone is not None and not 0.0 <= self.paint_tone <= 1.0:
            raise ValueError("paint_tone must lie in [0, 1].")
        if self.load_fraction is not None and not 0.0 <= self.load_fraction <= 1.0:
            raise ValueError("load_fraction must lie in [0, 1].")
        if self.normal_force_target_n is not None and (
            not math.isfinite(self.normal_force_target_n) or self.normal_force_target_n < 0.0
        ):
            raise ValueError("normal_force_target_n must be finite and non-negative.")


@dataclass(frozen=True, slots=True)
class PlantCommand:
    """One timestamped low-level actuator command in SI units."""

    sequence: int
    monotonic_time_s: float
    position_target_rad: tuple[float, ...]
    velocity_target_rad_s: tuple[float, ...] | None = None
    feedforward_torque_nm: tuple[float, ...] | None = None
    control_mode: str = "position"
    tool: ToolExecutionCommand = ToolExecutionCommand()

    def validate_for(self, capabilities: PlantCapabilities) -> None:
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative.")
        if not math.isfinite(self.monotonic_time_s):
            raise ValueError("monotonic_time_s must be finite.")
        if self.control_mode not in capabilities.command_modes:
            raise ValueError(f"Unsupported control mode: {self.control_mode}.")
        joint_count = len(capabilities.joint_names)
        _validate_joint_vector("position_target_rad", self.position_target_rad, joint_count)
        _validate_optional_joint_vector("velocity_target_rad_s", self.velocity_target_rad_s, joint_count)
        _validate_optional_joint_vector("feedforward_torque_nm", self.feedforward_torque_nm, joint_count)


@dataclass(frozen=True, slots=True)
class PhysicalSensorPacket:
    """Only physical or physically realizable sensor samples.

    Tuple fields follow `joint_names`. Exact simulator pose, contact state,
    material state, process parameters, and RNG state are intentionally absent.
    """

    sequence: int
    monotonic_time_s: float
    joint_names: tuple[str, ...]
    encoder_position_rad: tuple[float, ...]
    encoder_velocity_rad_s: tuple[float, ...]
    motor_current_a: tuple[float, ...]
    bus_voltage_v: tuple[float, ...]
    motor_temperature_c: tuple[float, ...] | None = None
    contact_force_n: float | None = None
    tool_deflection_m: float | None = None
    contact_switch: bool | None = None
    fault_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative.")
        if not math.isfinite(self.monotonic_time_s):
            raise ValueError("monotonic_time_s must be finite.")
        joint_count = len(self.joint_names)
        if joint_count == 0 or len(set(self.joint_names)) != joint_count:
            raise ValueError("joint_names must be non-empty and unique.")
        _validate_joint_vector("encoder_position_rad", self.encoder_position_rad, joint_count)
        _validate_joint_vector("encoder_velocity_rad_s", self.encoder_velocity_rad_s, joint_count)
        _validate_joint_vector("motor_current_a", self.motor_current_a, joint_count)
        _validate_joint_vector("bus_voltage_v", self.bus_voltage_v, joint_count)
        _validate_optional_joint_vector("motor_temperature_c", self.motor_temperature_c, joint_count)
        for name, value in (
            ("contact_force_n", self.contact_force_n),
            ("tool_deflection_m", self.tool_deflection_m),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite when present.")


@dataclass(frozen=True, slots=True)
class BodyBeliefSnapshot:
    """Agent posterior supplied to control and counterfactual prediction."""

    monotonic_time_s: float
    joint_names: tuple[str, ...]
    joint_position_mean_rad: tuple[float, ...]
    joint_position_variance_rad2: tuple[float, ...]
    joint_velocity_mean_rad_s: tuple[float, ...]
    joint_velocity_variance_rad2_s2: tuple[float, ...]
    contact_probability: float
    contact_force_mean_n: float | None
    contact_force_variance_n2: float | None
    posterior_revision: int
    inference_model_id: str = "unversioned-body-inference"
    calibration_status: str = "unspecified"

    def __post_init__(self) -> None:
        joint_count = len(self.joint_names)
        if joint_count == 0 or len(set(self.joint_names)) != joint_count:
            raise ValueError("joint_names must be non-empty and unique.")
        _validate_joint_vector("joint_position_mean_rad", self.joint_position_mean_rad, joint_count)
        _validate_joint_vector("joint_position_variance_rad2", self.joint_position_variance_rad2, joint_count)
        _validate_joint_vector("joint_velocity_mean_rad_s", self.joint_velocity_mean_rad_s, joint_count)
        _validate_joint_vector(
            "joint_velocity_variance_rad2_s2",
            self.joint_velocity_variance_rad2_s2,
            joint_count,
        )
        if any(value < 0.0 for value in self.joint_position_variance_rad2):
            raise ValueError("Joint position variances must be non-negative.")
        if any(value < 0.0 for value in self.joint_velocity_variance_rad2_s2):
            raise ValueError("Joint velocity variances must be non-negative.")
        if not 0.0 <= self.contact_probability <= 1.0:
            raise ValueError("contact_probability must lie in [0, 1].")
        if (self.contact_force_mean_n is None) != (self.contact_force_variance_n2 is None):
            raise ValueError("Contact force mean and variance must either both be present or both be absent.")
        if self.contact_force_variance_n2 is not None and self.contact_force_variance_n2 < 0.0:
            raise ValueError("contact_force_variance_n2 must be non-negative.")
        if self.posterior_revision < 0:
            raise ValueError("posterior_revision must be non-negative.")
        if not self.inference_model_id or not self.calibration_status:
            raise ValueError(
                "inference_model_id and calibration_status must be non-empty."
            )


@dataclass(frozen=True, slots=True)
class CounterfactualRolloutRequest:
    """Belief-conditioned motor prediction request with independent noise."""

    request_id: str
    initial_belief: BodyBeliefSnapshot
    stroke_intent: StrokeAction
    motor_primitive_kind: str
    model_snapshot_id: str
    sample_count: int
    independent_noise_seed: int

    def __post_init__(self) -> None:
        if not self.request_id or not self.model_snapshot_id:
            raise ValueError("request_id and model_snapshot_id must be non-empty.")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive.")
        if self.independent_noise_seed < 0:
            raise ValueError("independent_noise_seed must be non-negative.")


@dataclass(frozen=True, slots=True)
class CounterfactualRolloutResult:
    """Predicted execution outcomes returned to active-inference planning."""

    request_id: str
    final_joint_mean_rad: tuple[float, ...]
    final_joint_variance_rad2: tuple[float, ...]
    expected_energy_j: float
    contact_loss_probability: float
    execution_uncertainty: float
    model_snapshot_id: str
    approximation: str


@dataclass(frozen=True, slots=True)
class SimulatorEvaluationTruth:
    """Privileged simulator labels available only to evaluation tooling."""

    monotonic_time_s: float
    joint_names: tuple[str, ...]
    actual_position_rad: tuple[float, ...]
    actual_velocity_rad_s: tuple[float, ...]
    applied_torque_nm: tuple[float, ...]
    exact_contact: bool
    exact_contact_force_n: float
    hidden_state_revision: int


@runtime_checkable
class PlantBackend(Protocol):
    @property
    def capabilities(self) -> PlantCapabilities: ...

    def send_command(self, command: PlantCommand) -> None: ...

    def read_sensors(self) -> PhysicalSensorPacket: ...

    def close(self) -> None: ...


@runtime_checkable
class SteppablePlantBackend(PlantBackend, Protocol):
    def step(self, duration_s: float) -> None: ...


@runtime_checkable
class ExecutionForecaster(Protocol):
    def forecast(self, request: CounterfactualRolloutRequest) -> CounterfactualRolloutResult: ...


@runtime_checkable
class EvaluationTruthProvider(Protocol):
    """Optional simulator/debug interface that must never be an agent input."""

    def read_evaluation_truth(self) -> SimulatorEvaluationTruth: ...
