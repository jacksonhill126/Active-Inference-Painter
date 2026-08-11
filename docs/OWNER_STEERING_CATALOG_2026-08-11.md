# Owner Decision And Steering Catalog

Date compiled: 2026-08-11
Coverage: earliest locally recoverable project conversation through 2026-08-11

## Purpose

This is the itemized record requested by the project owner. It is intentionally
less abstract than the companion contribution brief. Each entry records a
specific decision, correction, selection, deferral, diagnostic observation, or
meaningful redirection visible in the surviving project conversations.

This catalog does **not** count every "okay," status request, run request, or
commit request as an intellectual contribution. A short approval is included
only when the adopted proposal is recoverable and consequential. Questions are
included when they changed the architecture, exposed a false claim, or caused
a durable investigation. Ordinary troubleshooting is included only when it
established a lasting requirement.

## Recoverable source boundary

The earliest local project-specific conversation found is a Codex session from
2026-06-29. The source repository already existed at that point, so the catalog
cannot establish the origin of every feature in the initial download. A
ChatGPT project named `Arm` is visible in the application, but its older chats
were not enumerated by the available thread interface. If those chats are
later exported, they should be reviewed as prehistory to this catalog.

Sources used:

| Key | Source | Coverage |
| --- | --- | --- |
| `CX-0629` | Codex session `019f14c4-3a6d-7221-b053-895e04ba6196` | 2026-06-29 |
| `CX-0630` | Codex session `019f18a7-823b-7eb0-8219-8622c84d9363` | 2026-06-30 through 2026-07-06 |
| `CL-0706` | Claude session `8ad2a6a6-a7c4-4562-a682-0533f05c67b0` | 2026-07-06 |
| `CX-0707` | Codex session `019f3cbd-bd7f-7f53-a52a-a1c7334eba90` | 2026-07-07 through 2026-07-24 |
| `CL-0714` | Claude session `e655b4e4-3c9d-42df-ba8b-c994be9d2285` | 2026-07-14 through 2026-07-15 |
| `CX-0727` | Codex session `019fa461-8f55-7f93-99ea-ddbb0064b324` | 2026-07-27 through 2026-08-02 |
| `CL-0803` | Claude session `7ccc29fa-acb4-47b6-b8ff-3a5910bea9b7` | 2026-08-03 through 2026-08-04 |
| `CX-0804` | current Codex session `019fcd79-6504-7960-b99e-11677a30bd72` | 2026-08-04 through 2026-08-11 |

The raw Claude files remain in the user's local Claude project store; this
repository contains only the summarized project-relevant steering, not copies
of the full private transcripts.

Labels:

- **Decision** — a design or scope choice stated by the owner.
- **Correction** — the owner identified a wrong model, claim, or behavior.
- **Requirement** — a constraint the implementation was expected to satisfy.
- **Selection** — the owner chose among alternatives.
- **Deferral** — the owner deliberately postponed or excluded work.
- **Diagnostic steering** — an observation that caused a substantive technical
  investigation or correction.
- **Adopted proposal** — an agent proposed the details and the owner explicitly
  selected the proposal; this is not recorded as sole owner origination.
- **Candidate** — an owner direction not yet accepted as a complete canonical
  design.

## 2026-06-29 — Establishing one real simulator and viewer

### S-001 — Build an executable visual simulation, not only a model library

- **Type:** requirement.
- **Owner steering:** install the required libraries and supply a robot-arm
  simulation/visualizer that the owner can actually run.
- **Effect/status:** established the live simulation and viewer as a first-class
  development surface rather than leaving the project as offline inference
  code.
- **Evidence:** `CX-0629` messages 1–2.

### S-002 — Begin with a Python-native simulator; omit unnecessary UI controls

- **Type:** selection and scope.
- **Owner steering:** initially requested a Python-native visualizer patterned
  on the earlier visual form, while explicitly saying not to spend effort on
  sliders and similar controls.
- **Effect/status:** focused the first integration on seeing the arm and paint,
  not on building a configurable GUI. The renderer was later changed to
  Three.js.
- **Evidence:** `CX-0629` messages 3–4.

### S-003 — Correct the initial arm/canvas/brush scale and orientation

- **Type:** diagnostic steering and correction.
- **Owner steering:** identified an oversized brush, wrong arm orientation,
  missing canvas paint display, transparent canvas, and arm/canvas clipping.
- **Effect/status:** drove the first geometry, opacity, painting-display, and
  collision/penetration corrections.
- **Evidence:** `CX-0629` messages 7–8.

### S-004 — Preserve the approximately 17-inch arm-to-canvas distance

- **Type:** requirement.
- **Owner steering:** rejected a geometry fix that moved the canvas so far away
  that the elbow had to remain fully extended; requested a 17-inch distance.
- **Effect/status:** introduced an explicit reach/geometry constraint instead
  of solving clipping by arbitrarily moving the work surface.
- **Evidence:** `CX-0629` messages 9–10.

### S-005 — Add a faster-than-real-time mode

- **Type:** requirement.
- **Owner steering:** requested a button or mode that lets the simulation run as
  fast as computation allows rather than being locked to wall-clock time.
- **Effect/status:** established accelerated simulation as necessary for
  observation and learning, an idea later extended into headless parallel
  collection.
- **Evidence:** `CX-0629` message 11.

### S-006 — Use Three.js for rendering while Python remains authoritative

- **Type:** architectural selection.
- **Owner steering:** after finding the Python GUI too pixelated, proposed a
  Three.js renderer with Python running active inference and accepted that
  split.
- **Effect/status:** became the enduring web architecture: browser rendering
  and controls, Python cognition and dynamics.
- **Evidence:** `CX-0629` messages 12–14.

### S-007 — Do not build a visually similar but behaviorally fake web planner

- **Type:** governing requirement and correction.
- **Owner steering:** explicitly rejected a parallel web implementation that
  only approximated the desired behavior: the browser must display the actual
  Python-driven model rather than imitate it.
- **Effect/status:** established a single-source execution rule. Later work
  similarly derived web geometry from MJCF rather than maintaining an
  independent robot.
- **Evidence:** `CX-0629` messages 16–17 and especially message 24.

### S-008 — Use conventional CAD-like orbit controls

- **Type:** interface requirement.
- **Owner steering:** requested pan/tilt behavior like the earlier HTML viewer
  and rejected side-to-side dragging that rolled the camera.
- **Effect/status:** set the viewer interaction model used for repeated visual
  diagnosis.
- **Evidence:** `CX-0629` messages 18–19.

### S-009 — Canvas contact must limit the arm; pressure is not deep penetration

- **Type:** physical-model correction.
- **Owner steering:** rejected interpreting pressure as the arm moving many
  inches behind the canvas. A brush may compress modestly, but the canvas must
  physically stop the arm and preserve plausible joint geometry.
- **Effect/status:** drove contact/compliance semantics and remains consistent
  with the later hard-support canvas and compliant brush.
- **Evidence:** `CX-0629` messages 20–22.

### S-010 — Telemetry must explain the actual probabilistic model

- **Type:** diagnostic steering.
- **Owner steering:** questioned zero epistemic value, a unit policy posterior,
  and unexplained scalar telemetry in a supposedly hierarchical probabilistic
  system.
- **Effect/status:** began the recurring requirement to expose VFE/EFE,
  posteriors, uncertainties, and hierarchy state rather than decorative
  numbers.
- **Evidence:** `CX-0629` message 23.

### S-011 — Physics must not freeze between cognitive phases

- **Type:** physical/runtime requirement.
- **Owner steering:** noticed that jitter and inertia vanished perfectly during
  lift/approach/paint gaps and asked whether simulation time was stalling.
- **Effect/status:** initiated the rule that planning and lifecycle phases may
  hold or retract the arm, but physical simulation must continue.
- **Evidence:** `CX-0629` message 25.

## 2026-06-30 through 2026-07-06 — Mathematical rigor, motor-aware realization, and spatial representation

### S-012 — Test that learned uncertainty changes epistemic policy selection

- **Type:** requirement.
- **Owner steering:** requested a deterministic fixture showing that greater
  learned-model uncertainty increases epistemic value and can alter the
  policy preference without adding reward-like terms.
- **Effect/status:** established a test standard for probabilistic claims:
  uncertainty must be decision-relevant, not merely logged.
- **Evidence:** `CX-0630` message 1.

### S-013 — Reject an invalid EFE decomposition and require literature-backed math

- **Type:** mathematical correction.
- **Owner steering:** challenged adding information gain to risk plus ambiguity,
  pointed out that epistemic/pragmatic value is an alternative decomposition,
  and demanded consultation of the established literature.
- **Effect/status:** caused repeated audits and the enduring rule that VFE and
  EFE decompositions are logged separately and information terms are not
  double-counted.
- **Evidence:** `CX-0630` messages 2–7.

### S-014 — Active inference must be mathematically principled, not a resemblance

- **Type:** governing requirement.
- **Owner steering:** stated that the system cannot merely "look kinda like
  active inference" and repeatedly requested complete, independent review.
- **Effect/status:** drove multiple audit passes, the reference-model tests,
  the research charter, and the current `AGENTS.md` governing constraint.
- **Evidence:** `CX-0630` messages 3 and 8–11.

### S-015 — Automatically continue across paintings and save periodic canvases

- **Type:** runtime requirement.
- **Owner steering:** requested automatic restart after stopping and a saved
  canvas image every five paintings.
- **Effect/status:** established long-run lifecycle and qualitative-record
  requirements. Later discussion clarified that immediate-stop bugs and
  restart-on-legitimate-stop are separate problems.
- **Evidence:** `CX-0630` message 14.

### S-016 — Expose the running Python version in the viewer

- **Type:** observability requirement.
- **Owner steering:** requested a version identifier beside the viewer title,
  connected to the actual Python version, and repeatedly rejected a broken
  placeholder.
- **Effect/status:** created a visible protection against diagnosing stale
  server code as if it were the latest implementation.
- **Evidence:** `CX-0630` messages 18 and 21–23.

### S-017 — Treat 80–90% material coverage as the intended terminal band

- **Type:** decision, later refined.
- **Owner steering:** reaffirmed the 80–90% target and proposed temporarily
  preventing stop below 70% because the painter was stopping after one mark.
- **Effect/status:** the strong terminal coverage preference remains canonical,
  conditional on `stop`. The hard no-stop floor was superseded: the immediate
  stop policy must now always remain available, with preference expressed
  probabilistically rather than through an undeclared action veto.
- **Evidence:** `CX-0630` messages 24 and 26.

### S-018 — Learned state must persist between paintings

- **Type:** requirement.
- **Owner steering:** explicitly required training retention when a new canvas
  starts.
- **Effect/status:** separated canvas reset from learned-model reset and later
  motivated checkpoint continuity.
- **Evidence:** `CX-0630` message 27.

### S-019 — Connect stroke planning to bodily controllability

- **Type:** owner-originated architectural requirement.
- **Owner steering:** argued that the stroke planner should consider what the
  motor level can realize predictably, while the motor level should anticipate
  planned strokes. "Easy" meant predictable and controllable, not low muscular
  effort.
- **Effect/status:** became the central painting-policy/motor-realization bridge.
- **Evidence:** `CX-0630` messages 30–33.

### S-020 — Adopt the conditional realized-execution formulation

- **Type:** adopted explicit plan.
- **Owner steering:** approved and pasted the detailed formulation
  `intended stroke -> q(realized execution | stroke, body, canvas) ->
  q(canvas next | realized execution, canvas) -> EFE`.
- **Effect/status:** established `StrokeReference`, execution forecasting,
  contact-aware realization, intended-versus-realized telemetry, and policy
  evaluation through predicted bodily consequences.
- **Evidence:** `CX-0630` messages 34–37.

### S-021 — Do not add a scalar motor-ease or effort reward

- **Type:** methodological requirement within the adopted plan.
- **Owner steering:** controllability may enter through realization bias,
  covariance, contact-loss probability, propagated material outcomes, smooth
  motor priors, and hard safety—not through a painting-level bonus.
- **Effect/status:** remains canonical in the control/plant/policy boundary.
- **Evidence:** `CX-0630` message 37.

### S-022 — Use receding-horizon, contact-aware control instead of waypoint chasing

- **Type:** adopted proposal.
- **Owner steering:** accepted a controller that previews approach, contact,
  paint, and lift; ramps pressure; and reduces overshoot before deposition.
- **Effect/status:** became the conventional realization layer beneath policy
  selection.
- **Evidence:** `CX-0630` messages 15–17 and the plan in message 37.

### S-023 — Keep simulating while planning; retract or hold away from the canvas

- **Type:** requirement and correction.
- **Owner steering:** requested that cognition run while the arm remains in a
  physically simulated holding/retracted state rather than freezing the world.
- **Effect/status:** led to planning holds, contact release, retraction cycles,
  and later passage-local holds.
- **Evidence:** `CX-0630` messages 38–40.

### S-024 — Add durable arm telemetry

- **Type:** requirement.
- **Owner steering:** requested logging of torque/current/position/velocity.
- **Effect/status:** grew into rolling plant and execution telemetry including
  contact, voltage, target error, force, and realization diagnostics.
- **Evidence:** `CX-0630` message 41.

### S-025 — Reject a nonspatial six-summary painting state

- **Type:** major architectural correction.
- **Owner steering:** reacted strongly on learning that the model could not
  represent where marks were placed and rejected calling that a minor
  limitation.
- **Effect/status:** caused the transition from summary-only planning toward a
  spatial material hierarchy; summary mode was later explicitly deprecated.
- **Evidence:** `CX-0630` messages 43–48.

### S-026 — The top latent need not be explicitly spatial, but intermediate layers must represent relations

- **Type:** conceptual decision.
- **Owner steering:** cautioned that forcing the very top level to be a literal
  spatial map might be limiting, while insisting that some layers evaluate
  spatial relationships and composition.
- **Effect/status:** anticipated the later split among pixel material fields,
  relational latents, passage latents, and slower painting-level causes.
- **Evidence:** `CX-0630` messages 49–51.

### S-027 — Specify which hierarchy variables are explicit and which are learned

- **Type:** architecture-planning requirement.
- **Owner steering:** asked for dimensionality at each level, the division
  between declared variables and learned latents, and whether deeper neural
  networks were appropriate.
- **Effect/status:** made representational transparency part of the planning and
  documentation standard.
- **Evidence:** `CX-0630` messages 52–54.

### S-028 — Ground visible tone must affect the actual simulated observation

- **Type:** correction.
- **Owner steering:** rejected a viewer-only gray canvas that the model could
  not perceive; relative contrast and tonal relationships had to exist in the
  simulation/observation path.
- **Effect/status:** forced separation of display background from material
  ground and later explicit surface-tone/ground-contrast state. Exact ground
  treatment evolved afterward.
- **Evidence:** `CX-0630` messages 62–64.

### S-029 — Composition should arise from efficient contextual compression

- **Type:** owner-originated research hypothesis.
- **Owner steering:** proposed that composition come less from explicit content
  priors and more from arrangements where parts of the canvas predict one
  another and high-level structure makes lower-level structure more
  predictable in context.
- **Effect/status:** became the compression-based composition program.
- **Evidence:** `CL-0706`, user UUID
  `bc88e20f-df01-4509-92ac-6ce29fa70821`.

### S-030 — Prefer hierarchical explanatory advantage over raw compressibility

- **Type:** collaborative adopted proposal.
- **Owner steering:** accepted the agent's correction that raw compressibility
  favors blank/uniform canvases and that the useful quantity is the hierarchy's
  explanatory advantage over a weaker baseline.
- **Effect/status:** produced the compression-gap preference and later the
  best-of-baseline family. Its present mathematical form is agent-formalized,
  not solely owner-originated.
- **Evidence:** `CL-0706` conversation following the compression proposal and
  the explicit instruction to implement the changes.

### S-031 — Fractal or periodic nonrepresentational output counts as success

- **Type:** research-scope decision.
- **Owner steering:** stated that fractal-like or periodic texture would be a
  desirable result and that the target was not representational pictures of
  familiar objects.
- **Effect/status:** prevented regular nonsemantic structure from being mislabeled
  as an aesthetic pathology and sharpened the nonrepresentational research aim.
- **Evidence:** `CL-0706`, user UUID
  `df7ee465-26de-493d-a026-7a299f66605f`.

### S-032 — Continue hierarchical planning and broaden brush-width support

- **Type:** selected next work.
- **Owner steering:** prioritized hierarchical planning and requested wider
  strokes or meaningful width variation.
- **Effect/status:** contributed to passage planning and wider/log-distributed
  mark-width proposal support.
- **Evidence:** `CL-0706`, user UUID
  `d8feadfd-02d2-412c-a7b4-5dd3cdf974d5`.

## 2026-07-07 through 2026-07-15 — Sparse multiscale planning, passages, embodiment, and oil paint

### S-033 — Represent passages and phrases, not only independent marks

- **Type:** architecture requirement.
- **Owner steering:** asked whether the agent could make a broader dark section
  through several related marks, then requested a hierarchical passage planner
  without demonstrations or fine-tuning.
- **Effect/status:** led to passage, polyline, passage-plan, and local-passage
  inference layers.
- **Evidence:** `CX-0707` messages 5–6.

### S-034 — Color selection must follow learned consequences, not random choice

- **Type:** correction and requirement.
- **Owner steering:** rejected independent random black/white selection because
  the two pigments have opposite material and visual effects.
- **Effect/status:** color/tone became part of the painting policy and predicted
  material consequence rather than an unrelated display choice.
- **Evidence:** `CX-0707` messages 7–8.

### S-035 — Document priors, likelihoods, dimensions, and precision at every level

- **Type:** transparency requirement.
- **Owner steering:** asked exactly what each prior and likelihood tracked, how
  many dimensions it had, and how precision weighting entered.
- **Effect/status:** led toward the generative-model specification and long-form
  architecture documentation.
- **Evidence:** `CX-0707` message 9.

### S-036 — Bottom-level brush prediction must operate on pixel-derived detail

- **Type:** major architecture correction.
- **Owner steering:** rejected 16×16 as the lowest perceptual/material
  resolution and argued that coarse-graining belongs higher in the hierarchy.
  A pretrained VAE was acceptable if needed, but the model had to retain enough
  local detail to learn brushmarks.
- **Effect/status:** directly motivated sparse pixel-local transition learning.
- **Evidence:** `CX-0707` messages 10–14.

### S-037 — Adopt sparse local dynamics with deterministic coarse-graining

- **Type:** adopted detailed plan.
- **Owner steering:** approved a local-patch transition ensemble at pixel scale,
  unchanged identity outside stroke support, material pyramids above it, and
  coarse composition evaluation derived from the predicted pixel state.
- **Effect/status:** produced the local spatial planner, local replay, sparse
  EFE scaling, and the multiscale material pyramid.
- **Evidence:** `CX-0707` messages 15–17.

### S-038 — Do not add a VAE or discrete AIF library before the pixel path is trustworthy

- **Type:** adopted scope decision.
- **Owner steering:** accepted the sparse-planner plan's explicit deferral of a
  VAE and `pymdp`: first establish correct pixel-local data and dynamics.
- **Effect/status:** explains why the early VAE idea was deferred rather than
  forgotten.
- **Evidence:** implementation plan pasted in `CX-0707` message 17.

### S-039 — Prefer correctness and passage batching before GPU optimization

- **Type:** priority selection.
- **Owner steering:** on a non-CUDA machine, deferred GPU work and chose batching
  plus hierarchical passage planning, explicitly favoring correctness over
  optimization.
- **Effect/status:** planning work proceeded through batching/caching first;
  CUDA was enabled later on a different capable machine.
- **Evidence:** `CX-0707` messages 19–20.

### S-040 — Let embodiment make arcs naturally different from Cartesian lines

- **Type:** owner-originated embodied-planning requirement.
- **Owner steering:** explained that an elbow rotation creates an easy, even arc
  for a painter, whereas forcing a straight Cartesian path may fight the body's
  dynamics.
- **Effect/status:** led to motor-conditioned action channels and ultimately the
  separate curvature/roll repertoire.
- **Evidence:** `CX-0707` messages 21–22 and 26–28.

### S-041 — Increase motor/linkage fidelity with plausible physical values

- **Type:** priority and requirement.
- **Owner steering:** identified unrealistically reliable control as distorting
  mark selection and requested more representative actuator, inertia, friction,
  backlash, and linkage behavior.
- **Effect/status:** produced the richer native plant and later the
  vendor-grounded MuJoCo electromechanical model.
- **Evidence:** `CX-0707` messages 27–30.

### S-042 — Use encoder feedback to prevent unobserved drift

- **Type:** correction and adopted proposal.
- **Owner steering:** challenged large joint drift in motors that should have
  positional encoding and approved closing the loop with encoder feedback.
- **Effect/status:** encoded feedback became part of the servo and body-state
  pathway.
- **Evidence:** `CX-0707` messages 32–35.

### S-043 — Hold position during planning and retract centrally

- **Type:** runtime/control requirement.
- **Owner steering:** diagnosed between-mark drift, requested a hold position,
  and later asked that global planning occur from a simple retracted pose rather
  than an unnecessary motion off to the side.
- **Effect/status:** produced planning holds, contact escape, and a central
  retraction cycle.
- **Evidence:** `CX-0707` messages 36–40, 45–48, and 55–57.

### S-044 — Use different timescales inside a passage and between passages

- **Type:** owner-originated hierarchical temporal model.
- **Owner steering:** described painting a passage with muted global concerns
  and fast local processing between marks, followed by stepping back for slower
  whole-canvas deliberation. Passages should take longer to plan than their
  execution.
- **Effect/status:** became passage-local inference, retraction between passages,
  and slower passage/painting beliefs.
- **Evidence:** `CX-0707` messages 40–41.

### S-045 — Model oil paint as thick, opaque, and wet-into-wet

- **Type:** physical-process requirement.
- **Owner steering:** increased opacity/deposition, corrected white-over-black
  behavior, specified oil rather than acrylic, and intended a fully loaded
  brush during preparation/planning.
- **Effect/status:** changed surface dominance, wet mixing, and brush loading;
  later brush work refined the same ontology.
- **Evidence:** `CX-0707` messages 51–54.

### S-046 — Paint does not dry during a painting session

- **Type:** explicit correction.
- **Owner steering:** repeatedly said not to decay wetness and later rejected a
  finite reservoir/drying interpretation because the medium is oil.
- **Effect/status:** removed within-session drying and preserved persistent wet
  material state.
- **Evidence:** `CX-0707` message 69 and `CL-0714`, user UUID
  `040de321-d4df-4614-8f23-8dfbf44001ca`.

### S-047 — Coverage means painted surface area, not accumulated layers

- **Type:** material/preference correction.
- **Owner steering:** stated that overpainting the same location should not keep
  increasing coverage; coverage is the canvas area carrying black or white
  material.
- **Effect/status:** coverage is now derived from paint thickness support and
  separated from visible tone and total paint mass.
- **Evidence:** `CX-0707` message 75.

### S-048 — Profile before optimizing and do not break sequential learning

- **Type:** development-method decision.
- **Owner steering:** requested a timing audit before selecting speedups and
  warned that skipping or overlapping learning could change the next action,
  which is conditioned on the previous mark and update.
- **Effect/status:** led to planning profiling, safe caching/batching, and
  explicit separation of parallel forecasts from sequential belief updates.
- **Evidence:** `CX-0707` messages 58–59.

### S-049 — Allow checkpoint reuse only when architecture is compatible

- **Type:** persistence requirement.
- **Owner steering:** requested an optional learned-weight file that can be
  loaded into an architecturally unchanged instance.
- **Effect/status:** produced checkpoint save/load and later full training-state
  continuation.
- **Evidence:** `CX-0707` messages 60–61.

### S-050 — Favor quasi-direct joint actuation and keep the distal arm simple

- **Type:** hardware direction, initially exploratory.
- **Owner steering:** questioned belt drives, noted the light payload and lack
  of wrist DOFs, and favored quasi-direct actuation located no farther distal
  than the elbow.
- **Effect/status:** evolved into the later no-belts RobStride direct-mount
  draft. The later wrist-roll study remained explicitly noncanonical.
- **Evidence:** `CX-0707` messages 62–67.

### S-051 — Keep initial paint handling mechanically simple

- **Type:** hardware/scope direction.
- **Owner steering:** proposed dipping into separate black/white containers and
  a fixed gamsol cup rather than treating paint handling as an expensive,
  elaborate mechanism.
- **Effect/status:** informed the later dedicated-brush and instantaneous reload
  approximations; gamsol cleaning was subsequently deferred because of
  variability.
- **Evidence:** `CX-0707` message 67.

### S-052 — Add learned high-level canvas and relational latents

- **Type:** selected architecture work.
- **Owner steering:** explicitly chose the high-level canvas latent and
  relational middle level as the next scaling step.
- **Effect/status:** created the canvas and relational posterior modules and
  their slower transition beliefs.
- **Evidence:** `CX-0707` messages 78–79.

### S-053 — Add upper-arm roll as a fourth DOF

- **Type:** hardware/plant decision.
- **Owner steering:** requested a roll degree of freedom around the upper-arm
  axis.
- **Effect/status:** became the accepted canonical arm topology and later a
  decision-relevant motor realization variable.
- **Evidence:** `CX-0707` message 82.

### S-054 — Enable GPU execution when suitable hardware is available

- **Type:** performance requirement.
- **Owner steering:** checked CUDA availability, requested CUDA-enabled PyTorch,
  and asked that the planner run on GPU.
- **Effect/status:** device selection and CUDA execution were added; profiling
  later showed that CPU motor forecasts, not neural EFE, dominated runtime.
- **Evidence:** `CX-0707` messages 83–85 and `CL-0714`, user UUID
  `4e685b14-4942-4650-ba40-96daf23b686a`.

### S-055 — Build a moderately realistic round-brush process without bristle-level simulation

- **Type:** owner-originated fidelity target.
- **Owner steering:** requested more paint handling, texture, blending, and
  angle-dependent shape for an initially round brush, while explicitly
  excluding expensive individual-bristle fidelity.
- **Effect/status:** initiated the successive brush-process models.
- **Evidence:** `CL-0714`, user UUID
  `72a2efd7-3737-4c63-b534-cec1d0658187` and follow-up approval
  `56eae5e0-6328-4075-97a2-d0c6343e5f1d`.

### S-056 — Prioritize blending and shape dynamics over decorative impasto lighting

- **Type:** priority selection.
- **Owner steering:** asked the agent to study commercial paint engines;
  accepted impasto lighting as optional, but identified blending and brush-shape
  dynamics as the larger problem.
- **Effect/status:** focused work on pickup/release, carried pigment, canvas
  tooth, tip lag, taper, and footprint rather than primarily on rendering.
- **Evidence:** `CL-0714`, user UUID
  `8c252054-999f-438c-90bc-c44e4668fbdd`.

### S-057 — Keep simulated brush physics cheap and calibratable for sim-to-real

- **Type:** methodology requirement.
- **Owner steering:** required enough fidelity for simulated training to help a
  future real robot, but at a cost compatible with many counterfactuals.
- **Effect/status:** favored reduced structural brush models with measurable
  parameters over fluid or individual-bristle simulation.
- **Evidence:** same `CL-0714` prompt as S-056.

### S-058 — Learn which motions are reliable or jittery

- **Type:** owner-originated motor-learning requirement.
- **Owner steering:** did not require a fully learned proprioceptive likelihood,
  but insisted that relative motion reliability be learned because painters
  strongly condition mark choice and realization on controllability.
- **Effect/status:** produced per-realization motion-reliability beliefs and
  decision-relevant forecast inflation.
- **Evidence:** `CL-0714`, user UUID
  `3fe79cb1-bbd4-479d-8161-3bf63e3c41cb`.

### S-059 — Bound spatial replay after the overnight slowdown

- **Type:** diagnostic steering.
- **Owner steering:** reported that a long overnight run became extremely slow
  and suspected a leak.
- **Effect/status:** investigation found a bounded-but-enormous spatial replay
  buffer; the spatial runtime cap was reduced and regression-tested.
- **Evidence:** `CL-0714`, user UUID
  `8bf50801-cc64-46f9-b83f-88c7a03b8573`.

### S-060 — Represent polylines as a higher-level sequence of straight marks

- **Type:** action-hierarchy decision.
- **Owner steering:** requested low-dimensional polylines and clarified that
  they could be planned as a series of straight marks one level above a mark,
  analogous to passage planning.
- **Effect/status:** produced polyline passage support without forcing one mark
  primitive to encode every shape.
- **Evidence:** `CX-0707` messages 92–94.

### S-061 — Maintain an explanatory architecture brief for the owner

- **Type:** documentation requirement.
- **Owner steering:** requested an informal, educational, overexplained brief
  calibrated to undergraduate aerospace engineering and moderate active-
  inference knowledge.
- **Effect/status:** established the dual-audience reporting practice now also
  present in `AGENTS.md`.
- **Evidence:** `CX-0707` message 96.

## 2026-07-16 through 2026-07-24 — Research identity, feasibility, foveation, and program planning

### S-062 — Produce a roadmap toward principled active inference

- **Type:** governing planning decision.
- **Owner steering:** asked where the architecture was unprincipled or
  hand-wavy and requested a staged plan to make it an actual active-inference
  model, with each major item receiving its own later design session.
- **Effect/status:** led to the research charter, milestone plans, reference
  fixtures, and continuing formal audits.
- **Evidence:** `CX-0707` messages 97–98.

### S-063 — Do not train composition as RL-like aesthetic feedback

- **Type:** research-method constraint.
- **Owner steering:** rejected painting-corpus fine-tuning and owner-feedback
  optimization as too close to RL on an objective. Proposed repetition with
  variation and nested structure as consequences of information compression.
- **Effect/status:** excluded aesthetic reward models and imitation as the
  primary composition mechanism.
- **Evidence:** `CX-0707` message 99.

### S-064 — Do not hard-code motif creation and transformation

- **Type:** correction and constraint.
- **Owner steering:** rejected explicitly instructing the system to create and
  vary motifs because that would supply the desired organization
  algorithmically rather than let it be discovered.
- **Effect/status:** motifs, symmetry, balance, and similar named devices remain
  outside the declared preference model.
- **Evidence:** `CX-0707` message 100.

### S-065 — Explore painting as self-investigation, but subject it to criticism

- **Type:** candidate direction and methodological requirement.
- **Owner steering:** preferred a self-investigation framing over motif coding,
  while asking for risks and alternatives before committing.
- **Effect/status:** informed the embodied/foveated research direction but was
  not encoded as a literal self-experiment reward or scripted probe sequence.
- **Evidence:** `CX-0707` messages 101–102.

### S-066 — Center the research on spatial relationships and sensorimotor vision

- **Type:** conceptual steering.
- **Owner steering:** doubted that an untuned agent would systematically use the
  canvas as a laboratory notebook; argued that human painting more plausibly
  explores space, vision, depth, and motion.
- **Effect/status:** shifted emphasis from explicit scientific probe behavior to
  active vision, spatial relations, and embodied prediction.
- **Evidence:** `CX-0707` message 102.

### S-067 — Nonrepresentational composition need not refer to objects

- **Type:** research-scope decision.
- **Owner steering:** used music as the analogy: composition may be structure
  without representing a house, cube, or other observed object.
- **Effect/status:** reinforced relational/nonsemantic composition and the
  project's abstract-painting identity.
- **Evidence:** `CX-0707` messages 103–104.

### S-068 — Foveation may ground spatial self-organization

- **Type:** candidate-to-program direction.
- **Owner steering:** accepted a preference for action continuity as potentially
  hard-coded, explored homeostasis/self-evidencing, and identified foveation as
  a route by which spatial structure might play a role analogous to temporal
  structure in music.
- **Effect/status:** foveation became a central research hypothesis. Action
  continuity remains constrained by the rule that it may be a declared motor
  trajectory prior, not an undeclared aesthetic reward.
- **Evidence:** `CX-0707` messages 105–107.

### S-069 — Active inference must be the central mechanism, with divergences disclosed

- **Type:** governing requirement.
- **Owner steering:** required rigorous active inference rather than
  bookkeeping, allowed ML implementation techniques where justified, and asked
  agents to say explicitly whenever the design diverges from canonical active
  inference.
- **Effect/status:** became the strongest standing rule in `AGENTS.md`.
- **Evidence:** `CX-0707` message 107.

### S-070 — The generative process should approximate the eventual physical platform

- **Type:** embodiment requirement.
- **Owner steering:** required representative mechanics, environment, and sensor
  modalities. Variables unavailable to a physical robot must be inaccessible
  to inference.
- **Effect/status:** established the generative-process/model boundary and the
  sensor-access ledger.
- **Evidence:** `CX-0707` message 109.

### S-071 — Temporarily defer detailed real-camera failure modes

- **Type:** deferral, later narrowed.
- **Owner steering:** initially allowed occlusion, distortion, glare, and related
  camera complications to be ignored while the sensor-modal architecture was
  established.
- **Effect/status:** enabled staged development. Occlusion, calibration, and
  provisional photometry were later reintroduced before hardware claims.
- **Evidence:** `CX-0707` message 109.

### S-072 — Leave the slow-latent/motivation problem open rather than force an answer

- **Type:** explicit open decision.
- **Owner steering:** identified motivation and the useful slow latent as the
  hardest unresolved problem and requested extended discussion rather than a
  quick implementation.
- **Effect/status:** slow canvas, relational, passage, and plan latents were
  introduced provisionally, but their ultimate painting-scale semantics remain
  an open research issue.
- **Evidence:** `CX-0707` messages 109–111.

### S-073 — Consider a minimal external self-view, then put it on hold

- **Type:** candidate and deferral.
- **Owner steering:** suggested a mirror or camera view of the robot in its
  environment as a minimal life outside painting, then explicitly put the idea
  on hold to investigate modern world models.
- **Effect/status:** not part of the current runtime; retained as conceptual
  history rather than silently expanding the world.
- **Evidence:** `CX-0707` messages 112–115.

### S-074 — Permit pretrained components, but expose imported structure

- **Type:** methodological direction.
- **Owner steering:** investigated JEPA/world-model integration, expressed
  concern that the homegrown model and dataset were too small, and considered
  pretrained open models acceptable if their scientific role was explicit.
- **Effect/status:** pretrained perception is permitted by the charter but must
  be ablated against from-scratch learning and cannot silently become the
  preference model.
- **Evidence:** `CX-0707` messages 114–120.

### S-075 — Constrain custom training to roughly $1,000

- **Type:** resource constraint.
- **Owner steering:** stated that any custom model must be very small because
  available training spend was about $1,000; pretrained open-source components
  therefore deserved serious consideration.
- **Effect/status:** feasibility under individual-scale compute became part of
  architecture selection.
- **Evidence:** `CX-0707` message 119.

### S-076 — Evaluate every direction against individual time and budget

- **Type:** development-governance requirement.
- **Owner steering:** asked agents to discuss feasibility regularly under the
  constraints of one person, limited time, and limited money, without
  compromising unnecessarily.
- **Effect/status:** informed the staged simulation/hardware roadmap and reduced
  brush/camera approximations.
- **Evidence:** `CX-0707` message 121.

### S-077 — Define the project as approximately 80% research and 20% conceptual art

- **Type:** project-identity decision.
- **Owner steering:** emphasized scientific value, clear scope, interesting
  results, and ideally isolatable evidence, while retaining a conceptual-art
  dimension.
- **Effect/status:** became the opening identity in the research charter.
- **Evidence:** `CX-0707` messages 122–123.

### S-078 — Use MuJoCo now while keeping Isaac Sim as a future option

- **Type:** platform direction.
- **Owner steering:** explicitly compared MuJoCo and Isaac Sim and investigated
  what Isaac might unlock later rather than switching immediately.
- **Effect/status:** MuJoCo became the current plant/backend; broader parallel
  or perception-heavy simulation remains a future option.
- **Evidence:** `CX-0707` messages 125–128.

### S-079 — Treat parallel training as important for an online-learning agent

- **Type:** early performance/experiment direction.
- **Owner steering:** identified parallel experience generation as likely
  important for a mostly online learner.
- **Effect/status:** the idea was implemented later as isolated parallel
  collection and centralized pretraining.
- **Evidence:** `CX-0707` message 128.

### S-080 — Replace superficial plans with capability-gated M0–M8 work

- **Type:** program-management decision.
- **Owner steering:** judged the plans too superficial, requested deeper M1–M3
  active-inference tasks, allowed cognitive and mechanical work to proceed in
  parallel, and asked for gate reviews.
- **Effect/status:** produced detailed milestone files, validation gates,
  dependency structure, and task tracker.
- **Evidence:** `CX-0707` messages 129–133.

### S-081 — Maintain a usable spreadsheet/Gantt and an $8,000 lifecycle budget

- **Type:** resource and planning requirement.
- **Owner steering:** requested an Excel Gantt, populated tasks across all
  milestones, a more useful workbook organization, and an $8,000 total project
  budget.
- **Effect/status:** created the planning workbook and preliminary budget.
- **Evidence:** `CX-0707` messages 134–139.

### S-082 — The roadmap must include building and testing the physical robot

- **Type:** major scope correction.
- **Owner steering:** objected that physical sensor/actuator integration,
  cameras, and hardware testing were missing even though the intent was to
  build the arm.
- **Effect/status:** expanded M5–M8 to include CAD, calibration, safety,
  staged bring-up, physical perception, and experiment programs.
- **Evidence:** `CX-0707` messages 146–149.

### S-083 — Preserve a rich environment separated by a sparse boundary

- **Type:** research question/steering, not implemented decision.
- **Owner steering:** raised the concern that a generative process fully
  representable through boundary states may not preserve a meaningful
  environment/internal separation and referenced sparse coupling.
- **Effect/status:** recorded as a conceptual test of environmental richness;
  it did not authorize arbitrary hidden-state access.
- **Evidence:** `CX-0707` messages 144–145.

### S-084 — Vendor the active-inference reference implementation and compare against it

- **Type:** methodology/resource decision.
- **Owner steering:** requested cloning the Active Inference Institute
  fundamentals repository into `Reference material`, later asked for direct
  comparisons, and wanted to learn the concepts needed to supervise the work.
- **Effect/status:** enabled independent reference/oracle tests and the chapter
  comparison used in subsequent audits.
- **Evidence:** `CX-0707` messages 150–156 and the separate 2026-07-24 study
  task.

## 2026-07-27 through 2026-08-02 — MuJoCo embodiment, cameras, sensor access, and material beliefs

### S-085 — Use direct-mounted RobStride drives with no belts

- **Type:** hardware selection/reaffirmation.
- **Owner steering:** instructed the model work to assume RobStride drives,
  direct mounting, and no belt transmissions.
- **Effect/status:** became the hardware-oriented actuator draft. Exact axis/SKU
  assignment is canonical, although its earliest selection may predate the
  surviving conversation.
- **Evidence:** `CX-0727` opening prompt.

### S-086 — Use a roughly 0.5-inch brush and noncoincident shoulder axes

- **Type:** hardware geometry requirement.
- **Owner steering:** specified an approximately half-inch brush with contact
  modeling and rejected the simplified co-located shoulder axes.
- **Effect/status:** entered the MJCF geometry and later corrected the round
  brush footprint scale.
- **Evidence:** `CX-0727` opening prompt.

### S-087 — Make MJCF interface with the generative/control stack, while Python renders paint

- **Type:** architecture decision.
- **Owner steering:** asked that MuJoCo supply the physical robot representation
  and integrate with the existing model/control system, while the web/Python
  path remained the paint visualizer rather than asking MuJoCo to simulate the
  full paint image.
- **Effect/status:** became the hybrid MuJoCo plant plus custom material process.
- **Evidence:** `CX-0727` opening follow-up.

### S-088 — Derive the web robot from the same XML as MuJoCo

- **Type:** consistency requirement.
- **Owner steering:** requested that frontend geometry and joint variables come
  from the MJCF source so the two views remain synchronized.
- **Effect/status:** produced the XML-derived web robot model.
- **Evidence:** `CX-0727` messages corresponding to the XML/frontend request.

### S-089 — Correct reach, post height, canvas height/distance, and downward pitch

- **Type:** visual/mechanical correction.
- **Owner steering:** identified inability to reach the lower canvas, requested
  a taller post and lower-arm pitch approaching −90 degrees, and allowed axis
  offset beside the post where necessary.
- **Effect/status:** drove several MJCF geometry/reachability revisions.
- **Evidence:** `CX-0727` owner image feedback on arm and canvas geometry.

### S-090 — Increase MuJoCo motor fidelity before detailed CAD

- **Type:** priority selection.
- **Owner steering:** while beginning linkage design thinking, asked agents to
  improve motor/electrical representation and clarify what fidelity MuJoCo
  actually provided.
- **Effect/status:** produced the vendor/derived electromechanical drive ledger,
  dcmotor actuators, controller limits, and telemetry.
- **Evidence:** `CX-0727` motor-fidelity discussion.

### S-091 — Tangential brush compliance is useful; individual bristle bending can wait

- **Type:** scope and contact-model selection.
- **Owner steering:** requested tangential bending/compliance, accepted leaving
  other details out, and explicitly said individual bristles were unnecessary.
- **Effect/status:** yielded lumped passive brush flexure rather than a bristle
  bundle solver.
- **Evidence:** `CX-0727` prompts on friction and compliance.

### S-092 — Base tangential compliance should not invent a preferred chisel direction

- **Type:** physical correction.
- **Owner steering:** requested isotropic tangential behavior in the then-current
  round-brush/MuJoCo compliance model.
- **Effect/status:** remains compatible with later directional push/pull
  interaction: the tuft has no fixed chisel axis, while handle geometry and
  canvas tooth can still create direction-dependent friction.
- **Evidence:** `CX-0727` tangential-friction prompt.

### S-093 — Deposition should follow contact/pressure rather than a paint-on flag

- **Type:** process correction.
- **Owner steering:** challenged the existence of a separate painting on/off
  flag and asked why deposition was not tied directly to contact pressure.
- **Effect/status:** made incidental and lifecycle contact physically
  consequential; later taper fixes avoided artifacts through realization, not
  by reintroducing a gate.
- **Evidence:** `CX-0727` prompt on the painting flag.

### S-094 — Enumerate accessible sensor state and hidden process state before proceeding

- **Type:** methodology requirement.
- **Owner steering:** asked to proceed step by step and explicitly disambiguate
  sensor-accessible observations from active/hidden simulator variables.
- **Effect/status:** produced the variable/sensor-access ledger and fail-closed
  boundaries.
- **Evidence:** `CX-0727` sensor-state discussion.

### S-095 — Put camera geometry into the simulated scene

- **Type:** embodiment requirement.
- **Owner steering:** requested physical camera geometry in the scene rather
  than treating the camera as an unlocated image function.
- **Effect/status:** enabled occlusion rays, pose sweeps, collision envelopes,
  and later compact-rig design.
- **Evidence:** `CX-0727` camera-geometry prompt.

### S-096 — Select camera positions through measured occlusion/visibility sweeps

- **Type:** experimental-design requirement.
- **Owner steering:** asked whether the camera should be centered or offset and
  proposed tests that minimize occlusion rather than choosing from intuition.
- **Effect/status:** created the reproducible pose-sweep tool and observability
  brief.
- **Evidence:** `CX-0727` camera-location discussion.

### S-097 — The contacting end effector must remain visible

- **Type:** camera observability requirement.
- **Owner steering:** made continuous visibility of the end effector during
  canvas contact the important criterion and accepted two cameras if needed.
- **Effect/status:** became a key camera-sweep metric.
- **Evidence:** `CX-0727` visibility prompts.

### S-098 — Explore head-on and profile/inspection views before selecting the compact rig

- **Type:** exploration, later superseded.
- **Owner steering:** considered a head-on view when the arm is retracted and a
  cheap tangent/profile camera for brush-to-canvas distance.
- **Effect/status:** informed the early four-view proposal. These cameras were
  later removed from the selected initial baseline in favor of the compact
  dual-IMX296 rig.
- **Evidence:** `CX-0727` camera-design prompts.

### S-099 — Use grayscale global observations plus native-resolution foveae

- **Type:** observation architecture decision.
- **Owner steering:** selected grayscale input, questioned inadequate camera
  resolution, and emphasized sampling local high-resolution regions without
  mechanical camera movement.
- **Effect/status:** produced native acquisition, independently derived global
  registration, and requested foveal products.
- **Evidence:** `CX-0727` camera-resolution and pose-sweep discussion.

### S-100 — Normalize multi-camera perspective and model focus limitations

- **Type:** technical steering.
- **Owner steering:** asked how oblique views could be correlated after
  perspective differences and noted depth-of-field/focus variation across an
  angled canvas.
- **Effect/status:** made calibration, rectification, per-camera uncertainty,
  and future optical validation explicit.
- **Evidence:** `CX-0727` perspective and focus prompts.

### S-101 — Represent white and black material, local mobility, contrast, edges, and rhythm at appropriate levels

- **Type:** representational correction.
- **Owner steering:** required explicit black/white material; argued that
  mobility belongs at brushmark/foveal scale; and said global decisions should
  depend more on image-making relations such as pigment location, contrast,
  edges, and rhythm than on global wetness/thickness averages.
- **Effect/status:** influenced the material fields and the later rejection of
  hand-selected summary variables. Contrast/edges/rhythm are not installed as
  heuristic rewards.
- **Evidence:** `CX-0727` owner material-state discussion.

### S-102 — Track brush loading, canvas pickup, and pigment mixture

- **Type:** owner-originated material requirement.
- **Owner steering:** explained that a white brush dragged into black becomes
  gray, that loading changes behavior, and that the relevant local state is the
  superficial paint interaction rather than a timestamped stack of abstract
  brushmarks.
- **Effect/status:** produced persistent brush load/pigment state, pickup,
  depletion, and local transition requirements.
- **Evidence:** `CX-0727` brush-loading sequence.

### S-103 — Make reload selectable and defer cleaning

- **Type:** scope/behavior decision.
- **Owner steering:** accepted instantaneous reload as an inferred preparation
  action, with selected-color reset; proposed dedicated brushes per color; and
  deferred cleaning because gamsol would add variability.
- **Effect/status:** compact brush preparation supports preserve/reload while
  cleaning remains out of the current policy set.
- **Evidence:** `CX-0727` reload/cleaning discussion.

### S-104 — Explicitly deprecate the six-summary mode

- **Type:** major correction and documentation requirement.
- **Owner steering:** again rejected high-level variables such as average
  thickness and requested that summary mode be unmistakably marked obsolete in
  favor of learned predictive features.
- **Effect/status:** summary mode is now an `obsolete_compatibility_fixture`.
- **Evidence:** `CX-0727` turns
  `019faf8f-29fa-7da2-9047-f0aedf04e7e6` and
  `019faf95-925d-7360-9af0-615c61d79078`.

### S-105 — Temporarily prioritize the camera path over perfect paint photometry

- **Type:** priority/deferral decision.
- **Owner steering:** accepted white-on-white as unobservable for the moment and
  asked to focus on camera architecture, with only a quick provisional specular
  term if inexpensive.
- **Effect/status:** enabled progress on the observation boundary without making
  a false hardware-photometry claim.
- **Evidence:** `CX-0727` camera-priority prompt.

### S-106 — Use existing Sony A7R II and Olympus OM-1 as provisional camera starting points

- **Type:** provisional hardware exploration, later superseded.
- **Owner steering:** explored whether existing mirrorless cameras and 25/35 mm
  lenses could seed the camera model.
- **Effect/status:** informed early high-resolution camera assumptions but was
  replaced by the compact Raspberry Pi global-shutter selection.
- **Evidence:** `CX-0727` mirrorless-camera discussion.

### S-107 — Show the delivered fovea and a memory-relevant fading trace

- **Type:** observability requirement.
- **Owner steering:** requested a viewer overlay for current foveal attention
  and a ghost trail, ideally linked to the memory duration that actually affects
  foveation rather than an arbitrary animation.
- **Effect/status:** current/requested fovea trace was added; full learned gaze
  memory remains future work.
- **Evidence:** `CX-0727` turn
  `019fb9cb-0f38-7031-b61f-708838c32075`.

### S-108 — Connect camera observations to inference after establishing the sensor contract

- **Type:** selected sequencing.
- **Owner steering:** first chose the native/global/foveal observation contract,
  then explicitly chose to connect it to the active-inference model.
- **Effect/status:** preserved a staged boundary: acquisition first, likelihood
  and posterior update next.
- **Evidence:** `CX-0727` turns
  `019fb91b-efde-7331-b61f-708838c32075` and the preceding step selection.

## 2026-08-03 through 2026-08-06 — Formal audit, proposal work, and the first sensor-mediated loop

### S-109 — Audit the implementation directly against the reference material

- **Type:** methodology requirement.
- **Owner steering:** asked for an evaluation of adherence to active inference,
  explicit divergences, the hardest unresolved problems, and possible
  solutions.
- **Effect/status:** exposed motor-EFE issues, proposal limitations, unit
  mismatches, and the composition cold-start trap.
- **Evidence:** `CL-0803`, user UUID
  `9185fddd-7ec3-4f1e-812f-d03cb483fb2f`.

### S-110 — Seed early structure from the body's own motion manifold

- **Type:** adopted agent proposal.
- **Owner steering:** explicitly said proposal A matched the owner's first
  thought and requested implementation of A–F.
- **Effect/status:** embodiment-generated bootstrap marks became an allowed
  source of early composition training. This is not a painting corpus or
  aesthetic demonstration.
- **Evidence:** `CL-0803`, user UUID
  `ba7f9638-1a42-4496-85fe-8b82a475382a` and proposal A in the preceding
  assistant response.

### S-111 — Strengthen the compression-gap baseline to isolate longer-range structure

- **Type:** adopted agent proposal.
- **Owner steering:** accepted replacing the too-weak flat baseline with a
  stronger context-free/local baseline so the hierarchy is credited for
  structure beyond local brush smoothness.
- **Effect/status:** the composition layer now compares against more than one
  baseline family. Exact coding choices were agent-designed.
- **Evidence:** same `CL-0803` approval and proposal B.

### S-112 — Investigate coverage precision as an inferred scaffold

- **Type:** adopted proposal, later constrained.
- **Owner steering:** accepted exploring a coverage precision that withdraws as
  structural evidence becomes discriminative, rather than letting fixed
  coverage dominate forever.
- **Effect/status:** precision beliefs were developed, but the terminal
  preference itself remains declared and unlearned; later audits added strict
  safeguards against preferences learning themselves from outcomes.
- **Evidence:** same `CL-0803` approval and proposal C.

### S-113 — Learn an amortized policy proposal without turning it into a preference

- **Type:** adopted agent proposal.
- **Owner steering:** accepted a learned proposal distribution trained from the
  policy posterior while preserving the EFE posterior as the selector.
- **Effect/status:** proposal-network implementation and recovery tests exist;
  emission remains default-off because finite candidate convergence was not
  established.
- **Evidence:** same `CL-0803` approval and proposal D.

### S-114 — Build a small discrete mirror as an independent mathematical oracle

- **Type:** adopted agent proposal.
- **Owner steering:** accepted verifying VFE/EFE and policy inference against a
  tractable discrete reference rather than relying on internal additivity tests.
- **Effect/status:** reference/oracle tests and accepted decomposition records
  were added.
- **Evidence:** same `CL-0803` approval and proposal E.

### S-115 — Explore closing the motor loop with active-inference action

- **Type:** adopted research proposal, not current canonical control.
- **Owner steering:** accepted investigation of using IK as a descending
  prediction and proprioceptive free-energy descent for action.
- **Effect/status:** body inference and posterior-conditioned forecasting were
  strengthened, but the current accepted boundary still permits conventional
  motor control below painting policy; a full Chapter-7 action law is not the
  live controller.
- **Evidence:** same `CL-0803` approval and proposal F.

### S-116 — Assess interrupted work before continuing it

- **Type:** development-process steering.
- **Owner steering:** after Claude exhausted usage, asked Codex to inspect the
  entire repository, determine whether the task ended abruptly, and assess the
  project before editing.
- **Effect/status:** produced the continuation audit, recovered incomplete
  proposal work, and began the current long task.
- **Evidence:** `CX-0804` messages 1–2.

### S-117 — Update documentation before relying on new agents

- **Type:** priority and governance requirement.
- **Owner steering:** prioritized documentation and later emphasized that fresh
  Codex/Claude instances rely on support docs instead of reconstructing the
  codebase.
- **Effect/status:** made documentation alignment part of the project's control
  surface rather than an optional afterthought.
- **Evidence:** `CX-0804` messages 3, 7, and 9.

### S-118 — Correct the claim that no specific motors were selected

- **Type:** factual/design correction.
- **Owner steering:** pointed out that specific motors had already been
  selected and rejected documentation that erased that decision.
- **Effect/status:** canonical plant vocabulary now distinguishes fixed
  RobStride hardware assignment from native/MuJoCo plants and motor
  realization.
- **Evidence:** `CX-0804` message 8.

### S-119 — Align counterfactual motor realization with the selected plant

- **Type:** architecture correction.
- **Owner steering:** clarified that motor realization needed to be corrected
  into alignment with the MuJoCo plant.
- **Effect/status:** MuJoCo execution now forecasts through independent MuJoCo
  data rather than reusing native-plant forecasts.
- **Evidence:** `CX-0804` messages 10–12.

### S-120 — Condition forecasts on frozen body, material, and brush beliefs

- **Type:** adopted sequence of agent-proposed priorities.
- **Owner steering:** repeatedly selected the next highest-priority closure work
  after plant alignment.
- **Effect/status:** body forecasts initialize from the body posterior;
  material forecasts from the spatial posterior; brush forecasts from compact
  load/pigment belief, all with independent future noise.
- **Evidence:** `CX-0804` messages 12–18 and corresponding commits
  `15d44c6`, `749c62a`, and `3dfedd2`. The detailed formulation was agent-led.

### S-121 — Prioritize an end-to-end painter that does not use exact simulator state

- **Type:** owner-selected integration target.
- **Owner steering:** asked how quickly the system could paint in simulation
  without exact process state and explicitly chose getting that loop running.
- **Effect/status:** produced the opt-in provisional sensor-simulation profile,
  causal camera gating, conservative warm-up, and repeated painting cycles.
- **Evidence:** `CX-0804` messages 21–23.

### S-122 — Correct repository author identity

- **Type:** project-governance correction.
- **Owner steering:** stopped a push after noticing the unintended
  `jhill@mechoshade` Git identity.
- **Effect/status:** repository identity was corrected before subsequent
  publishing. This is operational rather than architectural, but it is
  meaningful project stewardship.
- **Evidence:** `CX-0804` messages 24–26.

## 2026-08-07 through 2026-08-11 — Compact cameras, roll/curve repertoire, brush anisotropy, training, and painterly order

### S-123 — Correct simulated camera placement by visual inspection

- **Type:** diagnostic steering.
- **Owner steering:** twice identified that a camera body was mounted on the
  wrong side despite an attempted fix.
- **Effect/status:** corrected scene geometry and reinforced the need to inspect
  rendered physical placement, not only numeric camera transforms.
- **Evidence:** `CX-0804` messages 28–29.

### S-124 — Make the camera installation compact and self-contained

- **Type:** physical-design requirement.
- **Owner steering:** rejected the spatial envelope of the larger camera
  proposal and asked to bring cameras closer because available setup space was
  limited.
- **Effect/status:** drove the compact-rig redesign.
- **Evidence:** `CX-0804` messages 30–31.

### S-125 — Select exactly two Raspberry Pi Global Shutter Cameras

- **Type:** hardware selection.
- **Owner steering:** chose the two-camera Raspberry Pi global-shutter option
  after considering affordable edge-learning cameras.
- **Effect/status:** established the current selected-but-unpurchased dual
  IMX296 baseline; exact lens and mount geometry remain provisional.
- **Evidence:** `CX-0804` messages 33–34.

### S-126 — Make roll and curvature part of the usable repertoire

- **Type:** owner-originated action-space correction.
- **Owner steering:** observed that roll was ignored and marks were straight;
  explained that roll prevents knife-like upward contact and lets bodily
  dynamics support natural curves.
- **Effect/status:** produced continuous signed mark curvature plus neutral and
  symmetric fixed-roll conditional realizations.
- **Evidence:** `CX-0804` message 36.

### S-127 — Treat stopping-at-start separately from restart-after-stop

- **Type:** lifecycle diagnostic correction.
- **Owner steering:** clarified that the bug was not failure to restart after a
  legitimate stop; the painter was selecting/stalling at stop immediately on
  startup.
- **Effect/status:** prevented the symptom from being hidden by unconditional
  auto-restart and redirected debugging toward policy/lifecycle initialization.
- **Evidence:** `CX-0804` messages 40–46.

### S-128 — Verify black/white selection and contrast/edge capacity

- **Type:** diagnostic/conceptual steering.
- **Owner steering:** questioned why white was never selected and whether the
  model could represent contrast and edges.
- **Effect/status:** exposed proposal/prior/observability behavior and kept
  image-making capacity, rather than only material coverage, on the active
  research agenda.
- **Evidence:** `CX-0804` messages 43–44.

### S-129 — Explore an angled distal wrist-roll design as a branch

- **Type:** hardware exploration.
- **Owner steering:** questioned whether a wrist-roll axis with an angled brush
  might be preferable to upper-arm roll and asked for a branched simulation
  study.
- **Effect/status:** produced the noncanonical wrist-roll design generator,
  viewer, comparison, and tests.
- **Evidence:** `CX-0804` messages 48–55.

### S-130 — Put the wrist redesign on the back burner

- **Type:** explicit deferral.
- **Owner steering:** after seeing the branch and integration cost, chose not to
  continue it into the active-inference runtime.
- **Effect/status:** upper-arm roll remains canonical; wrist-roll files are
  clearly labeled exploration only.
- **Evidence:** `CX-0804` message 56.

### S-131 — Model ferrule-leading catch and handle-leading release as physical consequences

- **Type:** owner-originated brush-process requirement.
- **Owner steering:** requested force-sensitive sticking/chatter/irregular
  friction when pushing ferrule-first, so the model could learn the consequences
  of leading a stroke with the handle rather than receiving a direct reward for
  doing so.
- **Effect/status:** produced the reduced directional stick/release brush
  process with canvas tooth and separate normal/tangential force.
- **Evidence:** `CX-0804` message 56.

### S-132 — Research established brush/friction work before inventing a model

- **Type:** methodology requirement.
- **Owner steering:** explicitly asked for prior research and comparable
  resources before an ad hoc implementation.
- **Effect/status:** the process was grounded in Baxter, LuGre, round-brush, and
  paint-transfer literature, with approximations named.
- **Evidence:** `CX-0804` message 57.

### S-133 — Replace process-clone counterfactuals with learned generative approximations eventually

- **Type:** owner-originated long-term architecture correction.
- **Owner steering:** stated that counterfactual rollouts should ultimately be
  uncertain generative approximations, not merely forward runs of the same
  physics implementation as the world.
- **Effect/status:** current independent-process rollouts are now explicitly
  labeled an integration baseline; learned brush/contact likelihood remains
  required work.
- **Evidence:** `CX-0804` message 58.

### S-134 — Reject one identical arc as the curve repertoire

- **Type:** visual/action-space correction.
- **Owner steering:** objected to every stroke having the same curvature and
  robotic appearance.
- **Effect/status:** nonzero curvature magnitude became continuously sampled,
  with symmetric signs and preserved arclength semantics.
- **Evidence:** `CX-0804` message 60.

### S-135 — Correct the brush from flat/chisel behavior to an axisymmetric round tuft

- **Type:** major physical-model correction.
- **Owner steering:** identified that the result looked like a flat brush rather
  than a round brush with angle-dependent contact.
- **Effect/status:** removed the fictional material-width axis and tied
  elongation to the projected round-brush handle direction.
- **Evidence:** `CX-0804` message 61.

### S-136 — Use brush-axis/canvas-plane angle and realistic bundle diameter

- **Type:** geometric correction.
- **Owner steering:** rejected the oversized diameter and reiterated that the
  determinant is the angle between the end effector and canvas.
- **Effect/status:** footprint geometry now uses the 12.7 mm envelope and acute
  axis/plane incidence, circular at 90° and elongated toward grazing contact.
- **Evidence:** `CX-0804` message 62.

### S-137 — Remove the circular terminal stamp after taper

- **Type:** visual/process diagnostic correction.
- **Owner steering:** noticed a circular node after every tapered tail.
- **Effect/status:** investigation found fixed penetration and flow during
  zero-motion/lift frames; exit now unloads pressure/depth and starts lift with
  zero taper flow.
- **Evidence:** `CX-0804` message 63.

### S-138 — Do not infer composition learning from two paintings

- **Type:** interpretation/validation correction.
- **Owner steering:** rejected overinterpreting very early output and said two
  paintings were insufficient experience.
- **Effect/status:** reinforced capability gates and the rule against claiming
  learned composition from isolated runs.
- **Evidence:** `CX-0804` messages 64–65.

### S-139 — Parallelize experience collection and accelerate training

- **Type:** selected implementation priority.
- **Owner steering:** observed that real-time serial training would take too
  long and asked for multiple simultaneous environments or other speedups.
- **Effect/status:** produced isolated headless workers, trajectory-level data
  splits, centralized training, worker scaling benchmarks, and resumable runs.
- **Evidence:** `CX-0804` messages 66–70.

### S-140 — Preserve and visually evaluate trained checkpoints

- **Type:** validation/observability steering.
- **Owner steering:** asked what survived an interrupted run, whether learned
  behavior was visible, and requested loading the checkpoint into a live viewer.
- **Effect/status:** produced checkpoint recovery/viewing and a qualitative
  observation that output appeared more structured while remaining far from a
  composition result.
- **Evidence:** `CX-0804` messages 71–73.

### S-141 — Treat overlapping broken contours as possible local model mismatch

- **Type:** owner-originated conceptual steering.
- **Owner steering:** suggested that patches with overlapping lines, broken
  contours, and chaotic fragments require more information to explain and may
  be candidates for resolution.
- **Effect/status:** opened the present mesoscopic-order research direction.
- **Evidence:** `CX-0804` messages 74–75.

### S-142 — Define painterly order through compressible brushmark explanations, not pixel neatness

- **Type:** owner-originated candidate architecture.
- **Owner steering:** explained that a clean brushmark or recognizable series
  takes fewer parameters to describe than irregular, unrepeating fragments with
  many incompatible edge directions. Explicit contours may be unnecessary if
  a learned model captures this explanatory economy.
- **Effect/status:** recorded as a candidate mesoscopic likelihood/problem, not
  as a hand-coded contour or messiness reward.
- **Evidence:** `CX-0804` message 76, turn
  `019ff126-ae42-7e22-bee8-029121b5e8d4`.

### S-143 — Select locally disordered or globally incompatible patches, then infer a repairing mark

- **Type:** owner-originated candidate policy loop.
- **Owner steering:** proposed selecting patches that are locally
  unpredictable/disordered or incompatible with larger structure, rolling out
  mark candidates, and choosing a mark expected to reduce that mismatch.
- **Effect/status:** not yet implemented. It must be formalized as explicit
  likelihood/preference/EFE factors without becoming an aesthetic score.
- **Evidence:** `CX-0804` message 77, turn
  `019ff129-402a-7d83-9a78-d54fe653cb5d`.

### S-144 — Material consequence prediction is enabling, not the painting objective

- **Type:** owner-originated hierarchy distinction.
- **Owner steering:** said that the model must predict physical mark
  consequences, perhaps through pretraining, but that mere predictability of
  material consequences should not be the primary mark-selection driver.
- **Effect/status:** separates the low-level brush/material transition model
  from the emerging mesoscopic and painting-scale organization model.
- **Evidence:** same `CX-0804` message 77.

### S-145 — Reconsider the project's original pretrained VAE mark predictor

- **Type:** owner-supplied project history and candidate direction.
- **Owner steering:** recalled that the earliest project used a pretrained VAE
  for low-level marks and asked whether it would be useful again.
- **Effect/status:** selected on 2026-08-11 for a conditional local material-
  transition VAE shadow experiment. The implemented model is offline-only,
  has no policy influence, and is not yet the mesoscopic brushmark-order model.
  It records the owner's distinction that material consequence prediction is
  enabling rather than the main painting objective.
- **Evidence:** `CX-0804` message 78, turn
  `019ff12d-e547-7e20-88c5-7b2944e3b43e`.

## Decisions that evolved or were superseded

These are still owner steering and belong in the history, but future agents
must not treat the earliest form as current policy:

1. **Python-native viewer → Three.js rendering.** The owner initially requested
   Python-native visualization, then selected Three.js once visual limitations
   became clear. Python remained authoritative.
2. **Hard no-stop-below-70% idea → always-admissible immediate stop.** The hard
   floor was a response to a lifecycle/policy bug. Current architecture expresses
   the 80–90% target as a terminal preference and always includes immediate stop.
3. **Existing A7/OM-1 and multi-view camera concepts → compact dual IMX296.**
   Earlier inspection/profile views were exploration; exactly two compact
   global-shutter cameras are the selected initial baseline.
4. **Temporarily ignore camera realism → explicit provisional camera model.**
   The initial deferral enabled architecture work; current claims now name
   occlusion, calibration, noise, timing, and photometric approximations.
5. **No wrist DOF → wrist-roll exploration → wrist branch deferred.** Upper-arm
   roll remains canonical.
6. **Isotropic round-brush compliance → direction-dependent push/pull contact.**
   This is not necessarily a contradiction: the tuft has no fixed chisel axis,
   while handle geometry and surface tooth make the interaction directional.
7. **Full brush at each stroke → persistent brush-loading belief.** Initial
   simplification supported opaque oil marks; later owner input required
   depletion, pickup, mixture, and selectable reload across actions.
8. **Composition gap as the primary implemented structure signal → newly
   articulated mesoscopic order.** The recent idea is more specific and is not
   yet implemented by the existing compression-gap layer.
9. **Process-copy forecasts accepted for integration → learned uncertain
   generative forecasts required long-term.** The current baseline is a staged
   compromise, not the final epistemic architecture.

## Meaningful operational steering not counted as separate architectural decisions

The owner also repeatedly:

- requested runnable commands and live demonstrations rather than accepting
  code-only completion;
- caught stale servers through mismatched versions and asked that services be
  restarted after code changes;
- requested commits and pushes at coherent milestones;
- asked for honest status, limitations, cost, and timing rather than optimistic
  descriptions;
- required saved paintings, telemetry, checkpoints, and benchmark reports so
  claims could be inspected later;
- corrected repeated runtime bugs involving frozen physics, canvas contact,
  early stopping, invalid JSON, and lifecycle restart behavior;
- constrained long experiments by the available computer shutdown window and
  asked for restart/resume behavior;
- asked to learn the underlying active-inference mathematics rather than
  outsourcing all conceptual judgment to agents.

These actions materially shaped development quality and reproducibility, but
they are grouped here to keep the main ledger focused on design and research
steering.

## Remaining provenance gap

The catalog is comprehensive for locally recoverable project conversations
from 2026-06-29 onward. It is not yet comprehensive for the creation of the
initial repository or the earliest VAE-based design. The owner has stated that
the original low-level mark predictor was a pretrained VAE, but the transcript
where that architecture was first chosen has not been recovered.

If the ChatGPT `Arm` project or an older Claude export becomes available, add a
new pre-2026-06-29 section and distinguish:

- the initial research prompt;
- why active inference was selected;
- the first VAE/generative-model factorization;
- the origin of the terminal coverage target;
- the first hardware/kinematic concept;
- any early composition or foveation hypotheses.

This catalog should be treated as the detailed historical companion to
[What You Have Contributed To Active-Inference Painter](OWNER_CONTRIBUTION_BRIEF_2026-08-11.md)
and the more compact
[Project Decision Provenance Record](PROJECT_DECISION_PROVENANCE_2026-08-11.md).
