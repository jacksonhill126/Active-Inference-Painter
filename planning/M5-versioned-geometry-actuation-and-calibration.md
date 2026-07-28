# M5: Versioned Geometry, Actuation, And Calibration

## Summary

M5 creates the common physical-description layer used by the native simulator,
MuJoCo, CAD, controller, calibration tools, and eventual hardware. It replaces
scattered constants with versioned geometry, actuator, sensor, tool, and
canvas-station specifications that distinguish measured values, engineering
estimates, and unresolved uncertainty.

M5 is not a final mechanical design freeze. Preliminary schema, workspace,
actuator, and sensitivity work can proceed alongside M1-M3 and S1/S2. Values
are revised as CAD and prototypes produce evidence.

## Relationship To Active Inference

M5 is conventional robotics engineering. It supports active inference by
defining the generative process and the observations available from a physical
body. It does not turn CAD dimensions or estimated motor constants into
posterior beliefs automatically.

Parameter uncertainty has two roles that must remain distinct:

- engineering uncertainty used for sizing, sensitivity, and calibration;
- probabilistic uncertainty represented inside a generative model.

The second role requires an explicit model and inference procedure. Merely
storing a tolerance does not make it an inferred precision.

## Scope

### Included

- Frame graph and geometry schema.
- Actuator, transmission, encoder, and sensor interface schema.
- Provenance and uncertainty for every physical parameter.
- Workspace, singularity, and collision envelopes.
- Preliminary direct-drive versus belt-reduction trade study.
- Mass, inertia, torque, speed, stiffness, and thermal budgets.
- Brush, canvas, camera, paint, and cleaning-station interfaces.
- Calibration and system-identification plan.
- Export adapters for native simulation, MuJoCo, CAD, and manifests.
- Sensitivity and identifiability analysis.

### Deferred

- Final vendor selection and purchasing.
- Production drawings and fabrication release; M6 owns them.
- Safety certification and hardware commissioning; M7 owns them.
- Online Bayesian identification of every physical parameter.
- Full paint rheology and bristle finite-element modeling.
- Final camera calibration data before hardware exists.

## Tasks

### T-501 Define `RobotGeometrySpec`

Status: `Active`
Track: Geometry/Architecture  
Depends on: T-101, T-103  
Owner: Jackson/Codex  
Estimate: 2 days

Acceptance:

- Cover joint order, axes, signs, limits, home pose, link transforms, brush
  mount, canvas frame, base frame, and optional camera frames.
- Cover link mass, center of mass, inertia tensor, collision envelope, and
  visual geometry references.
- Declare SI units internally and conversion rules at interfaces.
- Support provisional, measured, and calibrated revisions.
- Avoid controller-specific correction offsets in the geometry schema.

Notes:

- Work began in the versioned MJCF and XML-derived web model with separated
  shoulder anchors, direct-drive joints, canvas frame, brush dimensions, and
  SI transforms.
- Acceptance remains open until these fields move into an authoritative
  backend-neutral `RobotGeometrySpec` with mass, collision, and camera
  coverage.

### T-502 Define the frame graph and naming convention

Status: `Blocked`  
Track: Geometry  
Depends on: T-501  
Owner: Jackson/Codex  
Estimate: 1 day

Acceptance:

- Name base, yaw, shoulder, roll, elbow, brush mount, brush tip, canvas,
  camera, paint, and cleaning-station frames.
- Declare transform direction, handedness, axis convention, and timestamp
  semantics.
- Provide canonical test poses with expected transforms.
- Distinguish design frames from measured calibration frames.
- Make CAD, MuJoCo, telemetry, and experiment manifests use the same names.

### T-503 Define actuator, transmission, and sensor specifications

Status: `Blocked`  
Track: Actuation/Control Interface  
Depends on: T-103, T-501  
Owner: Jackson/Codex  
Estimate: 2-3 days

Acceptance:

- Represent motor torque constant, resistance, voltage, speed, continuous and
  peak current, rotor inertia, thermal limits, and driver capabilities.
- Represent direct drive, belt reduction, compliance, backlash, and efficiency.
- Represent motor-side and joint-side encoders, current sensing, temperature,
  contact, and force sensing.
- Declare command modes actually exposed by the driver: torque/current,
  velocity, position, or impedance.
- Keep actuator choice independent for each joint.

### T-504 Add parameter provenance and uncertainty

Status: `Blocked`  
Track: Calibration/Research Ops  
Depends on: T-501, T-502, T-503  
Owner: Jackson/Codex  
Estimate: 1-2 days

Acceptance:

- Every parameter records source, revision, units, timestamp, and status:
  placeholder, datasheet, CAD estimate, measured, fitted, or validated.
- Store tolerances or uncertainty intervals where known.
- Distinguish manufacturing tolerance, measurement error, operating variation,
  and model discrepancy.
- Prevent missing values from becoming meaningful zeros.
- Record correlations when independent tolerances would be misleading.

### T-505 Define the preliminary design and operating envelope

Status: `Blocked`  
Track: Requirements/Mechanical  
Depends on: T-501, T-503  
Owner: Jackson/Codex  
Estimate: 2 days

Acceptance:

- Define canvas size and pose range, reachable painting area, desired tip
  speed, acceptable contact force, brush payload, and duty cycle.
- Define practical joint ranges, stowed/retracted poses, and service access.
- Define expected paint and solvent station locations.
- Separate desired capability, hard requirement, and provisional assumption.
- Include a smaller credible build envelope if the full workspace drives cost
  or inertia excessively.

### T-506 Run workspace, singularity, and collision studies

Status: `Blocked`  
Track: Kinematics/Mechanical  
Depends on: T-502, T-505  
Owner: Jackson/Codex  
Estimate: 2-3 days

Acceptance:

- Map reachable brush position and orientation over the canvas.
- Identify joint-limit, self-collision, base, easel, and station constraints.
- Quantify manipulability and poor-conditioning regions.
- Test retract, paint loading, cleaning, and return paths.
- Export representative cases for MuJoCo and controller validation.

### T-507 Compare direct drive and belt-reduction architectures

Status: `Blocked`  
Track: Actuation/Mechanical Decision  
Depends on: T-503, T-505, T-506  
Owner: Jackson/Codex  
Estimate: 3-4 days

Acceptance:

- Compare joint-local quasi-direct drive, modest belt reduction, and mixed
  architectures at yaw, shoulder, roll, and elbow.
- Compare reflected inertia, torque density, speed, backdrivability, backlash,
  compliance, packaging, encoder placement, cost, serviceability, and risk.
- Include gravity, contact, acceleration, and fault load cases.
- Identify where direct drive is plausible and where reduction is justified.
- Record a reversible preliminary decision rather than tuning the simulator to
  defend a preferred mechanism.

### T-508 Build mass, inertia, torque, stiffness, and thermal budgets

Status: `Blocked`  
Track: Dynamics/Mechanical  
Depends on: T-503, T-505, T-507  
Owner: Jackson/Codex  
Estimate: 3-4 days

Acceptance:

- Estimate link, motor, transmission, cable, brush, and mount masses and
  centers of mass.
- Compute static gravity, representative acceleration, contact, and emergency
  loads with explicit margins.
- Estimate joint and structural stiffness requirements relevant to brush-tip
  error and controller bandwidth.
- Estimate continuous and intermittent thermal duty.
- Report sensitivity to distal mass and uncertain dimensions.

### T-509 Define brush, canvas, camera, paint, and cleaning interfaces

Status: `Blocked`  
Track: End Effector/Workcell  
Depends on: T-502, T-505  
Owner: Jackson/Codex  
Estimate: 2-3 days

Acceptance:

- Define brush clamp, replaceable tool datum, nominal compliance, tip length,
  and allowable paint load.
- Define canvas registration, stiffness assumptions, and replacement procedure.
- Define fixed-camera mounting envelope and unobstructed field of view.
- Define secured black/white paint vessels and solvent-cleaning location.
- Separate manual initial loading/cleaning from later automation.

### T-510 Define calibration and system-identification procedures

Status: `Blocked`  
Track: Calibration  
Depends on: T-502, T-504, T-509  
Owner: Jackson/Codex  
Estimate: 3-4 days

Acceptance:

- Define encoder zeroing, joint-axis, link-transform, brush-tip, canvas-plane,
  camera, current/torque, friction, compliance, and contact calibration.
- Identify required fixtures and measurements.
- Define train/fit data separately from validation motions.
- State which parameters are directly measured and which are jointly fitted.
- Define residuals, uncertainty estimates, and acceptance thresholds.

### T-511 Implement parameter export and validation adapters

Status: `Blocked`  
Track: Tooling/Architecture  
Depends on: T-501, T-502, T-503, T-504  
Owner: Jackson/Codex  
Estimate: 3-5 days

Acceptance:

- Generate or validate native-sim, MJCF, CAD, controller, and manifest inputs
  from one versioned specification.
- Reject unit, frame, sign, missing-value, and inconsistent-limit errors.
- Record intentional backend approximations.
- Add round-trip or canonical-pose tests where possible.
- Avoid generating controller gains from geometry without a declared design
  method.

### T-512 Run sensitivity and identifiability analysis

Status: `Blocked`  
Track: Validation/Modeling  
Depends on: T-506, T-508, T-510, T-511  
Owner: Jackson/Codex  
Estimate: 3-4 days

Acceptance:

- Rank geometry, inertia, friction, compliance, motor, and sensor parameters by
  influence on tip path, contact, current, and paint consequences.
- Identify parameter combinations that available measurements cannot
  distinguish.
- Avoid spending fabrication or calibration effort on insensitive parameters.
- Flag parameters whose uncertainty could invalidate embodied policy
  comparisons.
- Produce a prioritized measurement and prototype list.

### T-513 Select a preliminary geometry and actuation baseline

Status: `Blocked`  
Track: Mechanical Decision  
Depends on: T-507, T-508, T-509, T-512  
Owner: Jackson  
Estimate: 1 day

Acceptance:

- Select provisional link dimensions, joint architecture, actuator class,
  encoder topology, brush interface, and workcell envelope.
- Record alternatives rejected, unresolved risks, and reversal cost.
- Verify that the selection fits the current cost and fabrication envelope.
- Preserve margins for wiring, housings, bearings, and iteration.
- Assign a new version when later evidence changes the baseline.

### T-514 M5 calibration-readiness gate

Status: `Blocked`  
Track: Validation  
Depends on: T-511, T-513  
Owner: Jackson  
Estimate: 0.5 day

Acceptance:

- Geometry, frames, actuation, sensors, tools, and workcell schemas are
  versioned.
- Parameter provenance and uncertainty are explicit.
- Preliminary sizing and architecture decisions have traceable calculations.
- Calibration and export paths are testable.
- M6 receives bounded fabrication questions rather than hidden simulator
  assumptions.

## Concurrency

- T-501 through T-505 can begin as soon as the S0 interfaces are known.
- T-506 through T-509 can proceed alongside M1-M3 and S1/S2.
- Calibration execution waits for hardware, but T-510 planning does not.
- M6 can begin parametric CAD after T-502 and T-505; it need not wait for the
  M5 gate.
- Geometry revisions must not silently alter research baselines.

## Feasibility

- Estimated effort: 28-41 focused workdays.
- Solo calendar estimate: 7-10 weeks, interleaved with simulation and CAD.
- Compute cost: negligible beyond local simulation.
- Main risks: premature component selection, underestimated distal inertia,
  unclear driver control authority, and treating estimated parameters as
  measurements.

## Outputs

- Versioned geometry, actuator, sensor, tool, and workcell specifications.
- Frame graph and canonical transform fixtures.
- Workspace and collision report.
- Direct-drive/belt trade study.
- Dynamic and thermal budgets.
- Calibration and identification plan.
- Backend/CAD export adapters.
- Sensitivity and preliminary architecture decision.
- M5 gate note.
