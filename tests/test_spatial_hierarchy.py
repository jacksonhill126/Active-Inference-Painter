import numpy as np

from active_painter.arm_sim import ArmPainterSim
from active_painter.config import PainterConfig
from active_painter.spatial_hierarchy import infer_mark_event_belief
from active_painter.spatial_state import spatial_canvas_state


def test_mark_event_belief_extracts_separated_spatial_material_regions() -> None:
    cfg = PainterConfig(canvas_size=32, spatial_grid_size=16, mark_slot_count=4, mark_activation_coverage=0.2)
    sim = ArmPainterSim(cfg)
    sim.canvas.thickness[4:10, 4:10] = 0.03
    sim.canvas.thickness[22:28, 22:28] = 0.03
    state = spatial_canvas_state(sim, cfg)

    belief = infer_mark_event_belief(state, cfg)
    active_slots = [slot for slot in belief.slots if slot.active_probability > 0.5]

    assert belief.active_count == 2
    assert len(active_slots) == 2
    centers = sorted((slot.center_x, slot.center_y) for slot in active_slots)
    assert centers[0][0] < 0.35
    assert centers[0][1] < 0.35
    assert centers[1][0] > 0.65
    assert centers[1][1] > 0.65
    assert belief.feature_matrix().shape == (cfg.mark_slot_count, 12)
    assert "not a policy preference" in belief.diagnostics()["approximation"]


def test_mark_event_belief_represents_white_material_as_contrast_to_gray_ground() -> None:
    cfg = PainterConfig(canvas_size=32, spatial_grid_size=16, mark_slot_count=4, mark_activation_coverage=0.1)
    sim = ArmPainterSim(cfg)
    sim.canvas.paint_at(
        np.asarray([0.0, sim.canvas.distance, 0.0]),
        pressure=0.8,
        tone=0.0,
        dt=0.2,
    )
    state = spatial_canvas_state(sim, cfg)

    belief = infer_mark_event_belief(state, cfg)
    slot = belief.slots[0]

    assert belief.active_count >= 1
    assert slot.mass > 0.0
    assert slot.mean_thickness > 0.0
    assert slot.mean_material_coverage > 0.0
    assert slot.mean_observed_tone < cfg.canvas_ground_tone
    assert slot.mean_ground_contrast > 0.0
