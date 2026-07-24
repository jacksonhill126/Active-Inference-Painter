# M7: Safety And Staged Hardware Bring-Up

## Summary

M7 defines and executes the staged path from individual powered joints to
supervised wet painting and finally bounded autonomous research runs. Safety,
fault handling, limits, and recovery remain conventional external engineering
constraints. They are never represented as painting preferences and cannot be
overridden by policy inference.

Safety design begins before the integrated arm is fabricated. Physical stages
advance only when the preceding stage has recorded tests and accepted
residual risks.

## Governing Boundary

Active inference may predict observable current, temperature, contact,
tracking, and energetic consequences and may hold prior preferences over
viable outcomes. It does not own:

- emergency stop;
- power isolation;
- hard current, temperature, force, speed, or joint limits;
- workspace and collision interlocks;
- communication watchdogs;
- non-finite-state handling;
- supervised operating-mode selection.

These mechanisms can veto or stop a selected policy. The resulting interruption
is logged as an external safety event, not silently rewritten as a policy
choice.

## Staged Operating Modes

1. Unpowered inspection and manual motion.
2. Single-joint powered bench mode.
3. Two-link powered mode.
4. Full-arm dry commissioning.
5. Controlled brush-contact mode.
6. Supervised wet painting.
7. Sensor-equivalent shadow inference.
8. Bounded autonomous research mode.

Each mode has its own command authority, limits, supervision, and recovery
procedure.

## Tasks

### T-701 Perform hazard analysis and define operating modes

Status: `Blocked`  
Track: Safety  
Depends on: T-503, T-505  
Owner: Jackson  
Estimate: 2-3 days

Acceptance:

- Identify crushing, impact, pinch, entanglement, electrical, thermal,
  unexpected motion, falling structure, sharp-tool, paint, and solvent hazards.
- Identify hazards during startup, calibration, normal motion, contact, paint
  handling, cleaning, maintenance, and faults.
- Define allowed personnel, command authority, and supervision for each mode.
- Record severity, likelihood, detection, mitigation, and residual risk.
- Treat the analysis as a living document updated by prototypes and incidents.

### T-702 Design the independent safety architecture

Status: `Blocked`  
Track: Safety/Electrical  
Depends on: T-701  
Owner: Jackson/Codex  
Estimate: 2-4 days

Acceptance:

- Define power isolation, enable chain, E-stop, driver enable, brake or
  gravity-safe behavior, and manual reset.
- Keep the safety path independent of the web UI and painting-policy process.
- Define safe behavior on computer crash, communication loss, sensor loss,
  driver fault, and power interruption.
- State what remains energized after each stop class.
- Identify hardware versus software enforcement explicitly.

### T-703 Specify emergency stop, recovery, and restart

Status: `Blocked`  
Track: Safety/Operations  
Depends on: T-702  
Owner: Jackson  
Estimate: 1-2 days

Acceptance:

- Define accessible E-stop locations and tested stopping behavior.
- Define controlled stop, emergency stop, power removal, and fault-latched
  states.
- Require inspection and deliberate reset before motion resumes.
- Define recovery from canvas contact without uncontrolled retraction.
- Prevent automatic continuation of a partially executed painting policy after
  an ambiguous reset.

### T-704 Define hard motion, electrical, thermal, and contact limits

Status: `Blocked`  
Track: Safety/Control  
Depends on: T-508, T-701, T-702  
Owner: Jackson/Codex  
Estimate: 3-4 days

Acceptance:

- Define hard and soft joint, speed, acceleration, current, voltage,
  temperature, force/contact, and workspace limits.
- Define limit sources and update rates.
- Include canvas, easel, base, paint, cleaning, and human exclusion zones.
- Use conservative commissioning limits that may expand only through evidence.
- Keep these limits outside EFE and policy posterior calculations.

### T-705 Implement watchdog and state-validity rules

Status: `Blocked`  
Track: Safety/Software  
Depends on: T-702, T-704  
Owner: Jackson/Codex  
Estimate: 2-3 days

Acceptance:

- Stop motion on stale commands, stale sensors, non-finite state, revision
  mismatch, timing overrun, or lost backend connection.
- Define monotonic timestamps and heartbeat requirements.
- Reject commands outside the active operating mode or calibration version.
- Log the exact cause, state, command, and recovery action.
- Test watchdog logic without relying on active-inference code.

### T-706 Define hardware calibration and encoder-zero procedure

Status: `Blocked`  
Track: Calibration/Operations  
Depends on: T-510, T-703, T-704  
Owner: Jackson/Codex  
Estimate: 2-3 days

Acceptance:

- Define safe homing or referencing without assuming absolute truth from an
  unvalidated encoder.
- Validate joint direction, limits, zero, current polarity, and sensor units at
  restricted power.
- Version the resulting calibration.
- Define checks required at every assembly change or brush replacement.
- Prevent motion under an incompatible calibration manifest.

### T-707 Commission and validate one powered joint

Status: `Blocked`  
Track: Hardware Bring-Up  
Depends on: T-611, T-702, T-704, T-705, T-706  
Owner: Jackson  
Estimate: 3-5 days

Acceptance:

- Verify enable, stop, current, temperature, encoder, direction, limits, and
  watchdog behavior.
- Measure tracking, hold, damping, friction, backlash, and thermal response.
- Test representative disturbances at conservative energy.
- Compare measured telemetry with M5/MuJoCo predictions.
- Update limits and model parameters from evidence.

### T-708 Commission and validate a two-link chain

Status: `Blocked`  
Track: Hardware Bring-Up/Dynamics  
Depends on: T-707, T-613  
Owner: Jackson  
Estimate: 4-7 days

Acceptance:

- Validate coupled gravity, inertia, tracking, damping, and stop behavior.
- Test coordinated slow trajectories and retraction without canvas contact.
- Compare predicted and realized joint/current consequences.
- Identify oscillation, structural mode, encoder, and cable effects.
- Do not advance if coupling invalidates safe single-joint limits.

### T-709 Commission the integrated arm in dry free space

Status: `Blocked`  
Track: Hardware Bring-Up  
Depends on: T-614, T-708  
Owner: Jackson  
Estimate: 5-8 days

Acceptance:

- Verify every joint, cable, stop, workspace zone, retracted pose, and
  emergency behavior.
- Execute canonical poses and low-speed paths away from the canvas.
- Compare kinematics, current, and timing against the versioned model.
- Validate startup, pause, manual jog, controlled stop, and shutdown.
- Record a dry commissioning artifact bundle.

### T-710 Validate brush approach, contact, release, and retract

Status: `Blocked`  
Track: Contact/Safety  
Depends on: T-612, T-709  
Owner: Jackson  
Estimate: 4-7 days

Acceptance:

- Approach a replaceable test surface at restricted speed and force.
- Validate contact detection, pressure proxy, release, normal retraction, and
  emergency recovery.
- Test edge, shallow, broad, and missed-contact cases.
- Measure brush and surface compliance and update contact models.
- Confirm planning or pause states cannot leave sustained unintended contact.

### T-711 Define paint and solvent operating procedures

Status: `Blocked`  
Track: Workcell/Safety  
Depends on: T-608, T-701  
Owner: Jackson  
Estimate: 2-3 days

Acceptance:

- Define manual filling, brush loading, transfer, cleaning, spill response,
  ventilation, storage, waste, and shutdown procedures.
- Secure paint and solvent containers and keep electronics outside likely
  spill paths.
- Review material safety information for the selected oil paint and solvent.
- Begin with supervised manual loading and cleaning.
- Treat later automated dipping or sloshing as motion-task expansion requiring
  its own collision and contamination tests, not a software-only feature.

### T-712 Validate physical sensors and fixed-camera observations

Status: `Blocked`  
Track: Sensors/Calibration  
Depends on: AI-201, T-509, T-706, T-709  
Owner: Jackson/Codex  
Estimate: 4-6 days

Acceptance:

- Validate joint, current, temperature, contact, and fixed-camera channels,
  units, rates, noise, delay, and dropout behavior.
- Calibrate camera-to-canvas geometry and record residuals.
- Produce blank, black, white, overlap, wet-paint, and motion fixtures.
- Ensure inference receives only the declared sensor-equivalent package.
- Keep hidden measurement references available only for calibration and
  evaluation.

### T-713 Implement the hardware backend and provenance path

Status: `Blocked`  
Track: Runtime/Hardware  
Depends on: T-103, T-401, T-705, T-709, T-712  
Owner: Jackson/Codex  
Estimate: 5-8 days

Acceptance:

- Implement the backend-neutral command, observation, telemetry, capability,
  timestamp, and calibration contract.
- Keep operating-mode and safety vetoes external to painting policy inference.
- Expose hardware, geometry, calibration, driver, sensor, and code versions.
- Represent unsupported simulator-only fields explicitly.
- Add recorded-data replay before permitting live autonomous commands.

### T-714 Run fault-injection and recovery tests

Status: `Blocked`  
Track: Safety Validation  
Depends on: T-703, T-705, T-709, T-713  
Owner: Jackson  
Estimate: 3-5 days

Acceptance:

- Test communication loss, stale sensor, invalid command, non-finite state,
  encoder disagreement, current/temperature limit, contact fault, and backend
  crash.
- Verify stop class, power state, telemetry, fault latch, and recovery.
- Use restricted energy or simulation where destructive testing is not
  justified.
- Record failures without weakening limits to make tests pass.
- Repeat critical cases after relevant hardware or controller changes.

### T-715 Conduct supervised wet painting trials

Status: `Blocked`  
Track: Hardware Validation/Painting  
Depends on: T-710, T-711, T-712, T-714  
Owner: Jackson  
Estimate: 4-7 days

Acceptance:

- Execute scripted black, white, overlap, curved, broad, edge, and cleaning
  cases under direct supervision.
- Measure contact, current, path, observation, deposition, and cleanup
  consequences.
- Validate that paint handling does not defeat sensors or safety mechanisms.
- Compare physical observations and consequences with simulation.
- Keep painting policies scripted for this commissioning stage.

### T-716 Run sensor-equivalent inference in shadow mode

Status: `Blocked`  
Track: Active-Inference Transfer Validation  
Depends on: M2, T-713, T-715  
Owner: Jackson/Codex  
Estimate: 4-6 days

Acceptance:

- Feed recorded or live physical sensor observations to the M2 inference path
  without granting command authority.
- Compare posterior predictions, VFE, uncertainty, and proposed policies with
  realized observations and safe scripted actions.
- Identify simulator-to-hardware likelihood and transition mismatch.
- Prevent hidden calibration truth from entering inference.
- Require calibration evidence before any precision is increased.

### T-717 Conduct bounded autonomous research runs

Status: `Blocked`  
Track: Hardware Research/Safety  
Depends on: AI-315, T-714, T-716  
Owner: Jackson  
Estimate: 3-5 days initially

Acceptance:

- Define bounded duration, workspace, current, speed, contact, paint, and
  supervision conditions.
- Require immediate manual stop authority and automatic external safety vetoes.
- Record complete observation, belief, VFE, EFE, policy, controller, telemetry,
  canvas, and fault traces.
- Stop rather than improvise after unsupported capability or model failure.
- Treat the first runs as transfer experiments, not demonstrations of general
  autonomy.

### T-718 M7 hardware-readiness gate

Status: `Blocked`  
Track: Validation  
Depends on: T-714, T-715, T-716, T-717  
Owner: Jackson  
Estimate: 1 day

Acceptance:

- Safety architecture and staged procedures are accepted.
- Dry, contact, wet, fault, and shadow-inference evidence exists.
- Hardware and calibration provenance is complete.
- Residual risks and unsupported autonomous capabilities are explicit.
- Bounded research operation is permitted under a documented mode; broader
  autonomy is not implied.

## Concurrency

- T-701 through T-706 begin during M5/M6.
- T-711 can proceed while the arm is fabricated.
- T-712 observation fixtures can begin with the fixed camera and canvas before
  full-arm commissioning.
- Powered stages wait on their exact mechanical and safety dependencies.
- M8 simulation experiments continue while hardware is unavailable.

## Feasibility

- Safety design effort: 12-20 focused workdays before integrated hardware.
- Physical bring-up effort: 35-60 focused workdays after hardware is available.
- Solo calendar estimate: 3-6 months, strongly dependent on faults and rework.
- Smaller credible endpoint: stop at scripted supervised wet painting and
  shadow inference; that is already scientifically useful transfer evidence.

## Outputs

- Hazard analysis, operating modes, and safety architecture.
- Limits, watchdogs, calibration, recovery, and fault-test records.
- Single-joint, two-link, dry-arm, contact, and wet-paint commissioning bundles.
- Physical sensor and camera characterization.
- Hardware backend and recorded-data replay.
- Shadow-inference and bounded-autonomy reports.
- M7 hardware-readiness gate note.
