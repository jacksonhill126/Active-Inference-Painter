# Angled Wrist-Roll Design Exploration — 2026-08-07

## Decision status

This is a non-canonical design exploration. It does **not** replace the accepted
`mujoco-robstride-electromechanical-v4` plant or its selected actuator
assignment. The current hardware-oriented draft still places RobStride 02 at
upper-arm `roll` and `elbow`.

The exploration asks a narrower question: if the four-actuator constraint is
retained, is a RobStride 02 wrist-roll axis carrying a fixed-angle brush a more
useful painting coordinate than upper-arm roll?

The executable branch is generated in memory by
`src/active_painter/wrist_roll_design.py` and is identified as
`mujoco-robstride-angled-wrist-roll-exploration-v0`, with status
`noncanonical_design_exploration_not_hardware_selected`. Generating the branch
rather than copying the full MJCF keeps the accepted camera, canvas, actuator,
sensor, and contact declarations synchronized while making the altered
kinematic hierarchy explicit.

## Branched mechanism

The canonical chain is:

```text
yaw -> pitch -> upper-arm roll -> elbow -> straight brush
```

The exploration chain is:

```text
yaw -> pitch -> elbow -> wrist roll -> brush fixed 15 degrees off wrist axis
```

Yaw, pitch, and elbow position the wrist center. The relocated `roll` joint
rotates the inclined brush around the distal forearm axis. Its rotation changes
the brush lean direction and makes the tip follow a compact circular arc. The
same stable joint, actuator, and sensor names are retained for controlled
comparison, but the model is not registered as a runtime plant backend.

## First quantitative comparison

The default branch uses a 15-degree brush angle, a wrist axis 247.2 mm from the
elbow, the selected 405 g RobStride 02, and a provisional 80/20 stator/rotor
mass split. A roll sweep from -32 to +32 degrees was evaluated at a fixed
representative arm pose.

| Quantity | Upper-arm roll | Angled wrist roll |
| --- | ---: | ---: |
| Uncompensated tip-path length | 363.23 mm | 24.00 mm |
| Tip chord across sweep | not decision-relevant here | 22.77 mm |
| Wrist tip-orbit radius | n/a | 21.48 mm |
| Brush-direction change | not isolated | 15.77 deg |
| Horizontal-pose elbow gravity bias | 1.326 N m | 2.307 N m |

The upper-arm figure is deliberately uncompensated: existing fixed-roll IK can
coordinate the other joints to preserve a selected Cartesian path. The useful
comparison is scale. Upper-arm roll strongly reorganizes the whole arm, whereas
the angled wrist produces a localized mark-scale motion that can plausibly be
learned as a direct brush consequence.

Relocating the RS02 distally increases the modeled static elbow gravity demand
by 0.981 N m, about 74% at the horizontal reference pose. The resulting 2.307
N m remains below the selected RS02's 6 N m rated torque, but this is not a
dynamic, thermal, emergency-stop, bearing-load, or fatigue validation.

### Brush-angle sensitivity

| Fixed brush angle | Orbit radius | Tip path over +/-32 deg | Brush-direction change |
| ---: | ---: | ---: | ---: |
| 8 deg | 11.55 mm | 12.90 mm | 8.46 deg |
| 10 deg | 14.41 mm | 16.10 mm | 10.56 deg |
| 12 deg | 17.26 mm | 19.28 mm | 12.65 deg |
| 15 deg | 21.48 mm | 24.00 mm | 15.77 deg |
| 18 deg | 25.65 mm | 28.65 mm | 18.85 deg |
| 22 deg | 31.09 mm | 34.73 mm | 22.90 deg |

This makes brush angle a real mechanical scale parameter rather than a cosmetic
choice. The 12-15 degree range is the most credible starting band: it provides
roughly 17-21 mm of local orbit radius without making wrist motion dominate a
large fraction of the 508 mm canvas.

## Bounded reachability check

A damped least-squares position IK solved yaw, pitch, and elbow while holding
roll at -32, 0, or +32 degrees. The grid contained 25 canvas points spanning a
380 x 380 mm square, producing 75 targets per design. A target counted as
reachable below 2 mm Cartesian residual.

| Result | Upper-arm roll | Angled wrist roll |
| --- | ---: | ---: |
| Reachable targets | 75 / 75 | 75 / 75 |
| Median ideal IK residual | 0.0082 mm | 0.0080 mm |

This establishes bounded geometric reachability only. It does not establish
collision clearance, motor tracking, stable contact, useful bristle loading,
camera visibility, or physical buildability.

## Active-inference meaning

If adopted, wrist-roll trajectories would be motor-realization latents below
the selected painting policy. They would predict conditional canvas,
proprioceptive, contact, and brush-orientation outcomes. Wrist roll must not be
introduced as an aesthetic score or an ordinary controller renamed as active
inference.

The current likelihood cannot yet recognize the principal advantage of the
mechanism: anisotropic bristle orientation and its effect on edge morphology.
That requires an explicit brush-orientation/contact likelihood and balanced
camera-derived transition evidence across wrist direction, brush angle, stroke
direction, speed, curvature, load, and local material state.

## Approximation ledger

- Canonical aggregate inertias do not identify motor stator, rotor, housing,
  handle, and bracket contributions separately.
- The branch relocates the vendor-listed 405 g RS02 mass using an explicit
  provisional 80/20 stator/rotor split.
- The angled handle, bracket, cable routing, bearings, sealing, and brush-change
  mechanism do not have CAD-derived masses or collision geometry.
- The comparison is simulation-only and is not hardware calibrated.
- Workspace results are numerical position-IK checks, not executable contact
  strokes.

## Recommendation and next gate

The branch is promising enough to continue. It retains the tested canvas
workspace while producing a compact, directly observable brush motion. It
should not yet replace upper-arm roll in the hardware record.

The next gate is a matched dynamic experiment at 12 and 15 degrees:

1. add roll-compensated straight and curved contact trajectories;
2. measure target error, contact loss, force residual, elbow current, and camera
   visibility for both designs;
3. add a provisional round-brush incidence likelihood so footprint consequences
   are represented rather than merely rendered;
4. perform distal-mass, cable-twist, sealing, and brush-change CAD review;
5. only then decide whether to retain upper-arm roll, replace it with wrist
   roll, or justify a fifth axis.

## Reproduction

```powershell
python -m active_painter.wrist_roll_design
python -m pytest tests/test_wrist_roll_design.py -q
python -m active_painter.wrist_roll_viewer
```

The viewer is an unloaded mechanism-visibility demonstration. It exercises the
branched MuJoCo actuators but does not run painting policy inference, camera
updates, or material deposition.
