# Rough Gantt Chart

Assumption: schedule starts Monday, August 3, 2026. This is a rough planning
chart for a single-investigator pace. It should be revised after each milestone
gate.

```mermaid
gantt
    title Active-Inference Painter Robotics Roadmap
    dateFormat  YYYY-MM-DD
    axisFormat  %b %Y

    section Operating System
    Project tracker, manifests, versioning        :m0, 2026-08-03, 2w
    Failure-mode log and validation gates         :m0b, after m0, 2w

    section Baseline Lock
    Python sim and canvas invariants              :m1, 2026-08-17, 3w
    Planner/controller boundary validation        :m1b, 2026-08-24, 3w

    section MuJoCo Simulation
    Abstract MuJoCo arm clone                     :m2, 2026-09-07, 3w
    MuJoCo backend adapter                        :m3, after m2, 4w
    JS digital twin with painted canvas           :m4, after m3, 3w

    section Geometry and CAD
    Calibration-ready geometry spec               :m5, 2026-09-21, 4w
    CAD frame and joint model                     :m6, after m5, 6w
    Prototype design revision 1                   :m6b, after m6, 4w

    section Control and Safety
    Safety envelope and watchdog design           :m7, 2026-10-19, 4w
    Bench validation protocol                     :m7b, after m7, 4w

    section Hardware
    Single-joint and two-link bring-up            :h1, 2026-12-14, 6w
    Brush/contact calibration rig                 :h2, 2027-01-11, 6w
    Full-arm dry strokes                          :h3, after h2, 4w
    Full-arm wet painting                         :h4, after h3, 4w

    section Research Experiments
    Sim ablations and predictive validation       :r1, 2026-11-09, 8w
    Sim-to-real comparison protocol               :r2, 2027-02-08, 6w
    Hardware research runs                        :r3, after h4, 8w
```

## Phase Reading

- August-September 2026: organize the project and lock the current simulator baseline.
- September-November 2026: get MuJoCo to clone the current arm and drive the existing paint/controller loop.
- October-December 2026: define measured geometry, CAD conventions, and safety systems in parallel.
- December 2026-March 2027: hardware bring-up from single joint to wet painting.
- November 2026 onward: run research validation in simulation first, then compare against hardware.

## Dependency Rule

Do not let hardware/CAD detail block MuJoCo integration. The first MuJoCo target
is the abstract Python arm clone; calibrated physical geometry comes after
measurement.

