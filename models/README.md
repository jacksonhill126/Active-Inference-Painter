# Models

The accepted native simulator reference is documented in
[`docs/NATIVE_PLANT_REFERENCE.md`](../docs/NATIVE_PLANT_REFERENCE.md) as
`native-abstract-v0`.

## Current MuJoCo Draft

[`active_inference_painter.xml`](active_inference_painter.xml) is the
hardware-oriented `mujoco-robstride-electromechanical-v4` draft. It is
intentionally not named `mujoco-abstract-v0`: the yaw and pitch axes are
separated and the roll actuator is translated down the upper-arm axis, so it
does not reproduce the co-located shoulder origin of `native-abstract-v0`.

Stable controller-facing names are:

- joints in order: `yaw`, `pitch`, `roll`, `elbow`;
- position actuators: `<joint>_position`;
- state sensors: `<joint>_position_sensor` and `<joint>_velocity_sensor`;
- brush pose/contact: `tip`, `tip_position_sensor`, `brush_touch_sensor`,
  `brush_force_sensor`, and `brush_compression_sensor`;
- home keyframe: `safe_home`.

MuJoCo supplies the generative process for joint motion, rigid-body geometry,
brush compliance, and realized contact. It does **not** simulate paint.
`VerticalCanvas` remains authoritative for paint thickness, material coverage,
wetness, pigment, surface tone, and deposition. When selected, the MuJoCo
backend feeds its realized `tip`, brush deflection, and contact observations
into that existing material process. The JavaScript scene mirrors the same XML
geometry while remaining the primary visualizer.

The current web viewer now loads its body hierarchy, geometry, materials,
joint ranges, canvas pose, brush dimensions, and RobStride metadata from
`/api/robot-model`, which is derived from this XML. The browser continues to
place `/api/canvas.png` on the physical canvas face; no paint state is stored
in MuJoCo or Three.js.

With the default native plant, the web runtime exposes a clearly labeled
`legacy_canvas_cartesian_retarget` robot state. Conventional visualization-only
IK maps the legacy Cartesian brush point onto the physical canvas.

With `--plant-backend mujoco`, the backend instead exposes `mujoco_direct`:
joint position, velocity, dcmotor winding-current state, actuator force, tip
position, brush compression, and brush/canvas contact come from MuJoCo. The
telemetry voltage is the internal controller's applied voltage request; the
plant packet separately retains the 48 V bus voltage. A compatibility facade
maps the current legacy Cartesian controller into the physical workspace and
maps realized tip motion back into the existing canvas coordinates. Policy
selection and the material process are unchanged.

Counterfactual motor forecasts still deep-copy `native-abstract-v0`, rather
than running MuJoCo per policy particle. This is a named transitional
approximation, not a claim that the two plants have identical dynamics.

## Direct-Mount Actuator Assignment

| Joint | Actuator | Rated / peak torque | Rated / no-load speed | Envelope | Mass |
| --- | --- | ---: | ---: | ---: | ---: |
| `yaw` | RobStride 03 | 20 / 60 N m | 100 / 195 rpm | 106 x 106 x 56 mm | 0.880 kg |
| `pitch` | RobStride 03 | 20 / 60 N m | 100 / 195 rpm | 106 x 106 x 56 mm | 0.880 kg |
| `roll` | RobStride 02 | 6 / 17 N m | 100 / 410 rpm | 78.5 x 78.5 x 45.5 mm | 0.405 kg |
| `elbow` | RobStride 02 | 6 / 17 N m | 100 / 410 rpm | 78.5 x 78.5 x 45.5 mm | 0.405 kg |

Both models use two magnetic encoders. The RS03 is specified for 48 V rated
(15-60 V range), 380 W rated output, 12 A peak-phase rated current, and 43 A
peak phase current. The RS02 is specified for 48 V rated (24-60 V range),
170 W rated output, 7 A peak-phase rated current, and 23 A peak phase current.
These values are stored as `custom/numeric` arrays in joint order for the
backend and future safety configuration; rated and no-load speed are not
treated as permission to operate the assembled arm at those speeds.

The XML uses `gear="1"` because each RobStride unit already includes its
planetary reduction and the joint is direct-mounted at the output. There are no
external belts.

## Electromechanical Drive Model

Each joint uses MuJoCo's `dcmotor` actuator with position input rather than an
ideal position servo. The model is a lumped, output-side equivalent of the
RobStride FOC drive, motor, planetary reduction, and position loop:

- the manufacturer torque constant is paired with output-side back-EMF,
  resistance, and viscous-loss equivalents jointly solved to pass through the
  published 48 V no-load speed/current point and peak-torque stall point;
- MuJoCo's power-balanced effective constant is `sqrt(Kt * Ke)`;
- equivalent terminal resistance preserves peak stall torque, while the
  per-joint viscous loss consumes the published no-load torque at the published
  no-load speed;
- the electrical current-state time constant is the listed line inductance
  divided by listed line resistance; the RS02 inductance uses the midpoint of
  its published 187-339 uH range;
- controller gains are voltage-space approximations selected to retain the
  preceding draft's joint stiffness and damping, not RobStride firmware gains;
- 48 V controller saturation and 60/17 N m peak torque saturation are active.

This adds back-EMF, finite current rise, voltage saturation, dynamic simulated
current, and torque saturation. The MuJoCo `saturation` clamp is deliberately
identified in this draft as an absolute peak envelope, not as proof that peak
torque can be sustained. Rated current is retained as a continuous-duty
diagnostic, but does not yet derate torque: thermal resistance/capacitance and
winding-temperature data were not found in the public specifications. The
unclipped current state remains observable and raises
`model_peak_current_exceeded` if regenerative or externally driven motion
crosses the modeled peak-current envelope. Likewise, the listed no-load current
is used only to fit a one-point equivalent viscous loss; that does not identify
the real split among bearing/gear friction, iron loss, and drive overhead.
Cogging, LuGre friction, controller delay/noise, gearbox compliance, backlash,
and thermal dynamics remain disabled until measured or otherwise supported.
The MuJoCo joint position/velocity sensors are still ideal: they do not yet
model magnetic-encoder quantization, bias, noise, sampling delay, packet loss,
or the distinction between the motor-side and output-side encoders.

The conversion is deliberately identified as an **equivalent integrated-drive
model**, not a phase-accurate inverter model. Manufacturer phase current and
line resistance cannot be inserted directly on the joint output side while
also preserving the published output torque and speed envelope. Hard current,
temperature, force, workspace, and watchdog limits stay outside
painting-policy inference.

At the fully horizontal zero pose, MuJoCo's gravity-bias terms for the current
approximate mass distribution are 6.65 N m at `pitch` and 1.33 N m at `elbow`
(zero at `yaw` and `roll`). That is below the respective 20 and 6 N m rated
torques, but it is only a static model check. It does not yet validate
acceleration, emergency stopping, brush impact/contact, bearing loads, thermal
duty cycle, or the eventual measured assembly mass distribution.

Vendor sources:

- [RobStride 03 product specification](https://robstride.com/products/robStride03)
- [RobStride 02 product specification](https://robstride.com/products/robStride02)
- [MuJoCo `dcmotor` actuator reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-dcmotor)
- [MuJoCo DC motor technical note](https://mujoco.readthedocs.io/en/3.7.0/_static/dcmotor.pdf)

## Geometry And Contact

The model preserves the existing joint sign convention
`Rz(yaw) Rx(pitch) Ry(roll) Rx(elbow)`. Its physical elbow range extends below
the current native controller range:

| Joint | Axis in its parent frame | Range |
| --- | --- | ---: |
| `yaw` | `+z` | -90 to +90 deg |
| `pitch` | `+x` after yaw | -90 to +90 deg |
| `roll` | `+y` along the upper arm | -180 to +180 deg |
| `elbow` | `+x` after roll | -85 to +150 deg |

The yaw axis is 285 mm above the base plate. The pitch axis is another 106 mm
up and 75 mm to the side of the post, giving 391 mm total pitch-axis height and
129.85 mm yaw-to-pitch center distance. This side offset gives the upper arm
clearance to swing down beside the post rather than through it.

The roll axis is 83 mm from the pitch axis along the upper arm; because that
translation is along the roll axis, it preserves the abstract rotation order
while avoiding co-located joint bodies. Pitch-to-elbow and
elbow-to-uncompressed-tip lengths remain 0.3302 m (13 in).

The arm lengths and canvas size are carried over exactly from the Python
reference: two 13 in links and a 20 in square canvas. The hardware-oriented
canvas pose is deliberately revised from the native reference:

| Field | Native Python reference | MuJoCo electromechanical v4 |
| --- | ---: | ---: |
| Upper-arm length | 0.3302 m (13 in) | 0.3302 m (13 in) |
| Elbow-to-tip length | 0.3302 m (13 in) | 0.3302 m (13 in) |
| Canvas width / height | 0.508 m (20 in) | 0.508 m (20 in) |
| Canvas contact plane | 0.4318 m (17 in) | 0.4826 m (19 in) |
| Canvas horizontal center | 0 m | 0.075 m |
| Canvas center height | shoulder-centered | 0.350 m |
| Canvas bottom / top | -0.254 / +0.254 m | 0.096 / 0.604 m |

The extra 50.8 mm of stand-off gives the direct-drive links and brush more
clearance from the board. Raising the center by 50 mm relative to v1 centers
the board better under the 391 mm pitch axis and leaves its lower edge 96 mm
above the floor. Analytic IK plus MuJoCo FK tests cover the center, every edge
midpoint, and all four corners.

The canvas center is also shifted 75 mm laterally to match the pitch-axis
offset beside the post. Neutral `yaw=0` now points at the canvas center instead
of requiring about +8.9 degrees of yaw. The left/right canvas-edge yaw values
are approximately +28.7 and -26.9 degrees, rather than the previous asymmetric
+35.7 and -19.9 degrees.

`native-abstract-v0`, `ArmPose.clipped()`, and the current IK still use an elbow
range of 0 to 150 degrees. That command range is a safe subset of the physical
MJCF range. The model exposes the additional downward travel now without
silently changing the accepted Python reference; extending the controller/IK
will require an explicit later reference revision and its own reachability and
collision tests. The MuJoCo backend reads the physical canvas plane from the
XML and converts realized tip/contact positions back into the existing
`VerticalCanvas` coordinate frame; it does not reuse
`VerticalCanvas.distance` as a physical SI coordinate.

The round bristle bundle is 12.7 mm (0.5 in) in diameter and 35 mm long. It has
12 mm of axial slide travel plus two passive tangential bend coordinates at the
ferrule. Each bend coordinate is limited to +/-20 degrees and currently uses an
approximate 1.2 N m/rad spring with 0.01 N m s/rad damping. The complete bundle
is still one rigid cylinder: this is a lumped root flexure, not a simulation of
individual bending bristles. Frictional contact torque deflects the bundle, the
spring returns it after lift-off, and the resulting physical `tip` position
drives the external paint process and the Three.js geometry.

`brush_touch_sensor` reports realized normal contact; paint pressure remains a
conditional inferred/predicted consequence, not a globally preferred scalar.

The contact pair currently uses equal `0.85` coefficients in both tangential
directions plus approximate torsional friction, so MuJoCo sliding friction is
isotropic in the canvas plane. These coefficients are placeholders pending
brush/canvas measurements; Python wet-paint state does not yet modify them.
The bend stiffness/damping values are likewise provisional and should
eventually come from a simple lateral tip load-deflection/free-decay test.

## Exact, Vendor-Backed, And Approximate Fields

- Exact interface contract: joint names/order/signs, `safe_home`, `tip`, SI
  units, and controller/sensor naming.
- Vendor-backed: RobStride outer envelopes, masses, reductions, rated torque,
  peak torque, voltage and speed envelopes, torque constants, currents,
  resistance, and inductance values listed above.
- Derived equivalent-drive fields: output-side back-EMF constant, effective
  MuJoCo motor constant, terminal resistance, one-point viscous loss, current
  limits, and electrical time constant. These preserve the published
  joint-output torque/speed/current envelope but are not raw phase parameters.
- Approximate: all brackets and links, body mass distribution and inertia,
  post height, physical joint ranges, shoulder/roll offsets, actuator
  armature/friction, position-loop gains, brush stiffness/damping/friction, and
  canvas contact parameters. Encoder observations are ideal and deterministic.
- Required measurements: CAD joint centers, assembled body masses/COM/inertia,
  output friction/backlash, safe cable-limited ranges, torque/current/thermal
  curves, motor step response and firmware configuration, brush geometry and
  load-deflection curve, and canvas/brush friction.

Expected future model lineages:

- `mujoco-abstract-v0`: native kinematic clone;
- `mujoco-calibrated-v0`: parameterized from physical measurements;
- `mujoco-safety-v0`: conservative hardware operating-envelope model.

## Validation

Load and inspect the model with:

```powershell
python -m mujoco.viewer --mjcf models\active_inference_painter.xml
python -m pytest tests\test_mujoco_model.py
```

The `safe_home` keyframe remains clear of the canvas. `contact_probe` creates
bristle contact near the canvas center for contact/sensor inspection.
`lower_arm_down` holds the shoulder pitch at zero and the elbow at -80 degrees
so the near-vertical downward forearm sweep can be checked.
`bottom_edge_probe` contacts 10 mm above the lower edge using a negative elbow
configuration.
