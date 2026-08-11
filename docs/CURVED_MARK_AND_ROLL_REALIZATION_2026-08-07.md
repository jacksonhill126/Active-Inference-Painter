# Curved Mark and Roll Realization — 2026-08-07

## Outcome

The painter can now propose and physically track curved single-mark policies,
and the bounded MuJoCo sensor profile no longer fixes every selected mark to a
neutral-roll Cartesian realization. It compares neutral Cartesian IK with
symmetric +24 and -24 degree fixed upper-arm-roll postures. The broader
research configuration retains both directions of a -32 to +32 degree roll
sweep, but those sweeps are not enabled in the repeated live smoke profile yet.

This is an action-space and generative-model correction, not a claim that the
system has learned to prefer aesthetically pleasing curves.

## Semantic split

Two different latent variables must remain distinct:

- `StrokeAction.curvature` is painting-policy geometry. It declares a signed
  quadratic-Bezier mark consequence. Zero is exactly the historical straight
  segment. Its magnitude is the maximum lateral midpoint deflection divided by
  endpoint chord length.
- `MotorPrimitiveLatent` is bodily realization below the selected painting
  policy. Neutral Cartesian IK and the two fixed-roll postures are alternative
  conditional controllers for the same declared mark. Dynamic roll sweeps are
  additional research alternatives.

There is no curve reward, roll reward, or globally preferred contact pressure.
Curvature enters the proposal distribution and its explicit proposal log
density. Motor alternatives enter the declared motor-policy prior, independent
plant likelihood forecasts, proprioceptive EFE terms, and conditional motor
posterior.

## What changed

1. `StrokeAction` gained signed `curvature`; summary action vectors are now
   eight painting variables and spatial action rasters are seven painting
   channels before motor conditioning.
2. Canvas deposition, spatial action rasters, path length, feasibility checks,
   timing, and controller references all use the same quadratic path.
3. Curvature is projected only as far as necessary to keep the sampled path in
   canvas proposal support. Endpoint geometry is not silently changed.
4. The hand-written proposal is a mixed measure: one straight atom has one
   third probability, while the other two thirds are split evenly between
   continuously distributed positive and negative curvature. The current live
   profile spans shallow through strong bends rather than emitting two fixed
   arc templates. This changes candidate support, not preference.
5. Fixed +24 and -24 degree upper-arm-roll realizations use the existing
   contact-aware Cartesian controller with fixed-roll IK along the entire path.
6. Opposite fixed-roll postures produce distinct predicted proprioceptive
   outcomes in simulation, so roll is decision-relevant rather than a display
   coordinate.
7. The live profile evaluates its three independent motor forecasts
   concurrently. Counterfactual integration is reduced to 30 Hz as an explicit
   simulation-throughput approximation; live process control is separate.
8. The empty initial compact-brush posterior made the old 8% reload prior
   dominate small-mark outcome evidence. The live smoke profile now uses an
   equal preserve/reload policy prior. Reload remains inferred from its expected
   deposition and pigment consequences; it is not forced by a threshold.

## Validation

Focused tests establish that:

- signed curvature has the declared midpoint meaning and longer arclength;
- proposal projection keeps the complete curve inside canvas support;
- the spatial action footprint follows the curve rather than the endpoint
  chord;
- the Cartesian reference reaches the declared curved midpoint;
- straight/positive/negative curvature branch probabilities are symmetric;
  nonzero magnitudes are continuously distributed and their mixed
  atom/density log measure is explicit;
- the proposal's mark `length` means approximate brush travel for both
  straight and curved marks; curved chords are shortened by their unit-curve
  arclength factor so curvature does not receive an undeclared coverage bonus;
- fixed and swept roll primitives remain semantically distinct;
- opposite fixed-roll postures generate different proprioceptive predictions;
- the camera-closed MuJoCo smoke profile completes repeated actions, accepts
  causal camera updates, records transition evidence, and deposits paint.

Straight paths retain exact fast paths for raster distance and servo reference
generation. This avoids charging legacy straight policies for curve sampling
on every control tick.

## Limits and next work

- The current curve family remains one symmetric quadratic arc per mark, but
  its signed magnitude is now continuously proposed from shallow through
  strong bends rather than fixed at one repeated value. Curvature apex timing,
  inflection/S-curves, and continuous learned curvature emission are not yet
  implemented.
- The live profile compares neutral and fixed-roll postures only. Roll sweeps
  exist and have distinct simulated likelihoods, but repeated camera-closed
  execution was not reliable enough to enable them in the smoke profile.
- The brush-contact model does not yet contain a calibrated anisotropic
  bristle-orientation likelihood. Consequently, the system can predict
  different joint/contact dynamics under roll, but cannot yet make a calibrated
  claim that a particular roll avoids the “knife edge” failure mode.
- The leakage-resistant corpus must be stratified by signed curvature and motor
  realization. Otherwise the learned transition model can average away exactly
  the effects added here.
- A matched set of upward, downward, and curved marks should measure path,
  contact loss, pressure residuals, coverage, and brush-edge morphology for
  neutral roll, fixed ±roll, and both sweep directions before enabling sweeps
  in the live profile.

The bounded camera process also uses an identity superficial-canvas composite
for the registered model-facing product. This retains camera geometry,
occlusion, quantization, and acquisition noise, but removes an unmodelled
lighting gain that previously drove a blank canvas to an implausible near-full
material posterior. It is a simulation-only photometric alignment, not a
hardware calibration result.

Likewise, the profile starts with bootstrap disabled and therefore does not use
the random local dynamics network to set a material mean. For the first 64
camera-derived transition samples, the declared action-conditioned transition
prior preserves the preceding material mean and widens uncertainty only inside
the selected brush footprint. The causally subsequent camera likelihood is the
only source of material-change evidence during this warm-up. This prevents an
untrained network from hallucinating canvas-wide paint and prematurely selecting
the otherwise valid terminal `stop` policy; it is not a shortcut around the
camera update or a hardware calibration claim.
