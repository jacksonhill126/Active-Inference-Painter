from active_painter.arm_sim import ArmPainterSim, ArmPose
from active_painter.config import PainterConfig
from active_painter.telemetry_log import ArmTelemetryLog, TELEMETRY_COLUMNS


def test_arm_telemetry_log_records_pose_velocity_current_and_torque() -> None:
    sim = ArmPainterSim(PainterConfig(canvas_size=24))
    sim.set_target(ArmPose(yaw=12.0, pitch=-42.0, roll=8.0, elbow=92.0))
    for _ in range(8):
        sim.step(1.0 / 120.0)
    log = ArmTelemetryLog(max_samples=8)

    log.append_from_sim(
        0.125,
        sim,
        phase="paint",
        painting_count=2,
        agent_enabled=True,
    )
    sample = log.recent(1)[0]

    assert sample["phase"] == "paint"
    assert sample["painting_count"] == 2
    for name in ("yaw", "pitch", "roll", "elbow"):
        assert f"position_{name}_deg" in sample
        assert f"target_{name}_deg" in sample
        assert f"velocity_{name}_rad_s" in sample
        assert f"velocity_{name}_deg_s" in sample
        assert f"current_{name}_a" in sample
        assert f"torque_{name}_nm" in sample
        assert f"actuator_position_{name}_deg" in sample
        assert f"encoder_position_{name}_deg" in sample
        assert f"position_error_{name}_deg" in sample
        assert f"elastic_deflection_{name}_deg" in sample
        assert f"encoder_std_{name}_deg" in sample


def test_arm_telemetry_log_exports_stable_csv_header() -> None:
    sim = ArmPainterSim(PainterConfig(canvas_size=24))
    log = ArmTelemetryLog(max_samples=8)

    log.append_from_sim(0.0, sim, phase="planning", painting_count=0, agent_enabled=True)
    csv_text = log.to_csv()

    header = csv_text.splitlines()[0].split(",")
    assert header == list(TELEMETRY_COLUMNS)
    assert "position_yaw_deg" in csv_text
    assert "current_elbow_a" in csv_text
    assert "torque_pitch_nm" in csv_text
    assert "encoder_position_yaw_deg" in csv_text
    assert "position_error_roll_deg" in csv_text
    assert "elastic_deflection_pitch_deg" in csv_text
    assert "encoder_std_elbow_deg" in csv_text


def test_arm_telemetry_summary_reports_rolling_retention_and_estimated_rate() -> None:
    sim = ArmPainterSim(PainterConfig(canvas_size=24))
    log = ArmTelemetryLog(max_samples=3)

    for index in range(5):
        log.append_from_sim(float(index), sim, phase="planning", painting_count=0, agent_enabled=True)

    summary = log.summary(4.0)

    assert summary["sampleCount"] == 3
    assert summary["maxSamples"] == 3
    assert summary["firstSampleTime"] == 2.0
    assert summary["lastSampleTime"] == 4.0
    assert summary["windowSeconds"] == 2.0
    assert summary["estimatedSampleHz"] == 1.0
    assert "rolling overwrite" in summary["retentionPolicy"]
