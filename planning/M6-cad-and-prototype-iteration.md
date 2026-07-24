# M6: CAD And Prototype Iteration

## Summary

M6 converts the versioned M5 physical description into an iterative mechanical
development process. It uses parametric CAD, targeted joint and linkage rigs,
measured feedback, and explicit revision records rather than attempting one
complete fabricated arm from unvalidated assumptions.

M6 is conventional mechanical engineering. Its scientific value is that
embodiment claims can later be tied to a documented body whose compliance,
backlash, inertia, sensing, and workcell constraints were measured rather than
invented after observing behavior.

## Development Strategy

The critical principle is to retire expensive uncertainty before committing to
expensive fabrication:

1. Resolve frame and requirement inconsistencies in a CAD skeleton.
2. Prototype the riskiest joint/transmission and brush-contact interfaces.
3. Measure them.
4. Feed measurements back into M5, MuJoCo, and controller versions.
5. Release a bounded integrated prototype only after the risky interfaces pass.

The first arm is a research prototype, not a production machine.

## Scope

### Included

- Requirements-to-CAD traceability.
- Parametric skeleton and frame mapping.
- Joint, transmission, bearing, structure, and brush-interface design.
- Cable routing, hard stops, serviceability, and assembly access.
- Base, easel, paint, cleaning, and camera fixtures.
- Tolerance, stiffness, load, manufacturing, and cost reviews.
- Single-joint and brush-contact prototypes.
- Measured update loop into M5 and MuJoCo.
- Integrated prototype release package.

### Deferred

- Cosmetic industrial design.
- Production tooling and volume manufacture.
- Automated brush exchange or a large paint-handling system.
- Final sealed solvent-compatible workcell.
- High-fidelity bristle or paint-flow finite elements.
- Unnecessary custom machining where adjustable or off-the-shelf structures
  can answer the research question.

## Tasks

### T-601 Define CAD traceability and revision policy

Status: `Blocked`  
Track: CAD/Research Ops  
Depends on: T-501, T-504, T-505  
Owner: Jackson/Codex  
Estimate: 1 day

Acceptance:

- Map CAD parameters to M5 geometry and actuation fields.
- Define CAD assembly, drawing, export, prototype, and hardware revision names.
- Require a reason and affected-interface list for each revision.
- Define how measured as-built values supersede nominal CAD values.
- Prevent silent edits to geometry used by a recorded experiment.

### T-602 Build the parametric CAD skeleton and frame map

Status: `Blocked`  
Track: CAD/Geometry  
Depends on: T-502, T-505, T-601  
Owner: Jackson/Codex  
Estimate: 3-5 days

Acceptance:

- Represent base, joint axes, link lengths, brush datum, canvas, camera, paint,
  and cleaning frames.
- Drive key dimensions from versioned parameters.
- Export canonical poses and transforms for M5 validation.
- Show adjustment ranges separately from nominal geometry.
- Avoid detailed housings until axis placement and workspace are accepted.

### T-603 Analyze structural load paths and stiffness

Status: `Blocked`  
Track: Mechanical Analysis  
Depends on: T-508, T-602  
Owner: Jackson/Codex  
Estimate: 3-5 days

Acceptance:

- Trace gravity, acceleration, contact, belt, bearing, and fault loads.
- Estimate link, joint, base, and easel deflection at the brush tip.
- Identify modes or compliance likely to interact with controller bandwidth.
- Apply explicit load factors without pretending rough estimates are certified
  structural analysis.
- Prioritize geometry changes by predicted consequence and fabrication cost.

### T-604 Design modular joint and transmission concepts

Status: `Blocked`  
Track: Mechanical/Actuation  
Depends on: T-507, T-603  
Owner: Jackson/Codex  
Estimate: 4-7 days

Acceptance:

- Develop joint-local concepts for the selected direct-drive, belt, or mixed
  architecture.
- Define motor, bearing, shaft, pulley, tensioning, encoder, and hard-stop
  interfaces.
- Preserve access for adjustment and replacement.
- Estimate backlash, compliance, reflected inertia, and assembly sensitivity.
- Identify one low-cost testable concept for each unresolved high-risk joint.

### T-605 Integrate actuators, encoders, and drivers mechanically

Status: `Blocked`  
Track: Mechatronics  
Depends on: T-503, T-604  
Owner: Jackson/Codex  
Estimate: 3-5 days

Acceptance:

- Define mounting datums, thermal paths, connectors, strain relief, and
  encoder alignment.
- Distinguish motor-side from joint-side measurement.
- Provide access for current, temperature, and calibration checks.
- Avoid routing loads through fragile encoder or motor housings.
- Record driver and sensor changes back into the M5 specification.

### T-606 Design cable routing, stops, covers, and service access

Status: `Blocked`  
Track: Mechanical/Safety  
Depends on: T-604, T-605  
Owner: Jackson/Codex  
Estimate: 2-4 days

Acceptance:

- Define cable bend radii, loops, clamps, flex cycles, and joint-range
  clearances.
- Add mechanical hard stops independent of software limits where practical.
- Identify pinch, snag, solvent, and paint-exposure risks.
- Permit belt tension, bearing, encoder, and brush maintenance without major
  disassembly.
- Include provisional covers only where they protect a concrete hazard.

### T-607 Design the brush mount and contact-compliance module

Status: `Blocked`  
Track: End Effector  
Depends on: T-509, T-602, T-603  
Owner: Jackson/Codex  
Estimate: 3-5 days

Acceptance:

- Define repeatable brush datum, replacement, paint loading, and cleaning.
- Provide intentional compliance or force sensing if needed without adding an
  undeclared painting-policy DOF.
- Estimate tip-location variation from brush replacement and bending.
- Support thick wet oil paint and a fully loaded brush.
- Define a simple bench contact test before full-arm integration.

### T-608 Design the base, easel, camera, paint, and cleaning fixtures

Status: `Blocked`  
Track: Workcell  
Depends on: T-509, T-602, T-603  
Owner: Jackson/Codex  
Estimate: 3-5 days

Acceptance:

- Provide a rigid, adjustable base and registered canvas fixture.
- Keep camera and canvas geometry calibratable.
- Secure black paint, white paint, and solvent vessels against tipping.
- Keep the thinking/retracted pose inside the workcell without requiring a
  large lateral sweep.
- Provide manual access and spill containment for early supervised operation.

### T-609 Define tolerances, manufacturing processes, and inspection

Status: `Blocked`  
Track: Manufacturing  
Depends on: T-604, T-606, T-607, T-608  
Owner: Jackson/Codex  
Estimate: 3-5 days

Acceptance:

- Assign tolerances according to sensitivity rather than uniformly tight
  machining.
- Select off-the-shelf extrusion, plate, printed, laser-cut, or machined
  processes by function.
- Define critical datums and inspection measurements.
- Use adjustable joints or fixtures where likely iteration is cheaper than
  precision fabrication.
- Identify parts whose failure would force expensive redesign.

### T-610 Maintain BOM, fabrication, and iteration risk estimates

Status: `Blocked`  
Track: Cost/Risk  
Depends on: T-604, T-605, T-608, T-609  
Owner: Jackson  
Estimate: 2 days

Acceptance:

- Track motors, drivers, encoders, bearings, structure, fabrication,
  electronics, wiring, safety, tools, and painting fixtures separately.
- Include shipping, rework, spare parts, and at least one mechanical iteration.
- Classify make/buy decisions and lead-time risks.
- Identify where additional spending reduces research risk and where low-cost
  parts are adequate.
- Maintain a target compatible with the approximately USD 4,000 prototype
  objective, with explicit contingency rather than optimistic omission.

### T-611 Build and characterize a single-joint prototype

Status: `Blocked`  
Track: Prototype/Actuation  
Depends on: T-604, T-605, T-606, T-609  
Owner: Jackson  
Estimate: 5-8 days plus fabrication lead time

Acceptance:

- Exercise the riskiest representative joint with intended motor, driver,
  encoder, bearing, and transmission topology.
- Measure current, torque estimate, speed, tracking, backlash, compliance,
  temperature, noise, and hold behavior.
- Test controlled stops and power removal.
- Record raw data and update parameter provenance.
- Reject or revise the architecture before multiplying a bad joint design.

### T-612 Build and characterize a linkage/brush-contact rig

Status: `Blocked`  
Track: Prototype/Contact  
Depends on: T-607, T-608, T-609  
Owner: Jackson  
Estimate: 4-7 days plus fabrication lead time

Acceptance:

- Exercise brush approach, contact, pressure change, release, retract, paint
  loading, and cleaning geometry.
- Measure repeatability, compliance, contact-force proxy, brush-tip location,
  and canvas-fixture motion.
- Test thick wet paint on replaceable test surfaces.
- Keep manual supervision and hard motion limits.
- Feed observed contact and workcell constraints into M5 and simulation.

### T-613 Feed prototype measurements back into all models

Status: `Blocked`  
Track: Calibration/Digital Thread  
Depends on: T-511, T-611, T-612  
Owner: Jackson/Codex  
Estimate: 3-5 days

Acceptance:

- Update measured geometry, inertia estimates, friction, backlash, compliance,
  actuator, sensor, brush, and contact parameters.
- Assign new specification, MuJoCo, controller, and CAD revisions.
- Re-run canonical-pose, workspace, load, and sensitivity checks.
- Preserve pre-update models for reproducibility.
- Document residual mismatch rather than forcing every backend to agree.

### T-614 Conduct the integrated prototype design review

Status: `Blocked`  
Track: Mechanical Review  
Depends on: T-610, T-613  
Owner: Jackson  
Estimate: 2 days

Acceptance:

- Review requirements, frames, loads, actuation, wiring, serviceability,
  calibration, safety interfaces, cost, and unresolved risks.
- Verify that high-risk findings from both prototypes are addressed.
- Separate required changes from desirable polish.
- Release a bounded integrated prototype package or explicitly return to a
  named task.
- Record the cost and schedule consequence of remaining uncertainty.

### T-615 M6 prototype-release gate

Status: `Blocked`  
Track: Validation  
Depends on: T-614  
Owner: Jackson  
Estimate: 0.5 day

Acceptance:

- CAD and M5 specifications identify the same versioned body.
- Risky joint and contact assumptions have prototype evidence.
- Manufacturing and inspection requirements are bounded.
- BOM includes iteration and safety costs.
- M7 receives a testable prototype and unresolved-risk register.

## Concurrency

- T-601/T-602 can start before the M5 gate once frames and requirements exist.
- T-603 through T-610 proceed while M1-M3 and S1/S2 continue.
- T-611 and T-612 are independent rigs and may be fabricated in parallel.
- M7 safety architecture can begin before integrated CAD release.
- Physical full-arm bring-up waits for T-614, not for completion of all
  active-inference research.

## Feasibility

- Estimated engineering effort: 42-67 focused workdays plus fabrication lead
  times.
- Solo calendar estimate: 3-5 months with iteration.
- Main financial risk: custom parts repeated before joint and contact rigs
  resolve the important uncertainties.
- Smaller credible alternative: build an adjustable two-link planar arm with
  one brush-contact rig before adding upper-arm roll or automated cleaning.

## Outputs

- Parametric CAD and frame map.
- Joint, transmission, brush, and workcell designs.
- Manufacturing, inspection, and revision policy.
- BOM and iteration-risk register.
- Single-joint and brush-contact prototype reports.
- Updated M5, MuJoCo, and controller parameters.
- Integrated prototype release package and M6 gate note.
