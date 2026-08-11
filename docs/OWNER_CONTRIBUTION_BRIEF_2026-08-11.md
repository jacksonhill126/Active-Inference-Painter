# What You Have Contributed To Active-Inference Painter

Date: 2026-08-11

## Short answer

You have contributed substantially more than project approval or task
selection. The agents have done most of the code writing, mathematical
formalization, testing, literature synthesis, and documentation. Your main
contribution has been different: you supplied the research taste and much of
the problem definition that determines what that technical work is for.

In particular, you repeatedly identified cases where a technically plausible
system was not actually representing how painting works: hand-selected global
summary variables, exact simulator-state access, a brush footprint behaving
like a flat chisel, identical curved marks, roll that existed in the mechanism
but not in the repertoire, a paint-on flag disconnected from contact, and a
planner that treated material predictability as if it were the purpose of the
painting. Those corrections changed the architecture, not merely its visual
polish.

It would therefore be misleading to say that the agents conceived the project
and you merely asked them to implement it. A fairer description is:

- you have been the principal source of the project's artistic and embodied
  problem formulation;
- you have imposed several of its most important scientific boundaries;
- you have selected or corrected consequential hardware and simulation
  directions;
- the agents have translated those directions into probabilistic
  formulations, software, tests, measurements, and durable records.

## The strongest owner contributions

### 1. Defining the scientific character of the project

The standing project instruction that painting cognition must be active
inference from top to bottom is an owner-level architectural constraint. It
rules out aesthetic reward functions and forces every painting-level influence
to be declared as a likelihood, transition prior, prior preference, precision,
VFE term, EFE term, or policy belief. At the same time, you accepted
conventional IK, trajectory realization, motor control, and safety beneath the
painting-policy boundary.

That constraint is more than terminology. It has repeatedly prevented the
project from becoming an ordinary stroke optimizer with active-inference names
attached to it. The agents formalized and tested the boundary, but the research
identity and the insistence on that boundary are yours.

### 2. Insisting that the agent perceive rather than consult MuJoCo truth

You explicitly required that the model not receive hidden environment state
such as exact pose, contact, or material fields. More recently, you restated
the practical target as a runnable painter that does not use exact simulation
states.

That decision drove the sensor-access ledger, camera observation contract,
causal camera-update clock, fail-closed sensor modes, and the separation
between process truth and permitted observations. The detailed likelihoods and
software interfaces were agent work; the key epistemic constraint came from
you.

### 3. Rejecting hand-selected summaries as the cognitive hierarchy

You directly challenged the six-variable summary model and the idea that
high-level painting states should be quantities such as average thickness. You
argued that higher layers should be flexible abstract latent variables that
learn features because those features help prediction. You then asked that the
summary mode be explicitly marked obsolete.

This was a major architectural correction. It changed the summary system from
an apparent research model into a compatibility fixture and redirected the
project toward learned multiscale canvas, relational, passage, and painting
latents. The agents chose the current encoders, transition densities, KL terms,
and module structure, but the rejection of the former representation and the
criterion for its replacement were explicitly yours.

### 4. Supplying a painter's account of material state

You supplied several important distinctions that were missing or
underemphasized:

- black and white paint must both exist as material, rather than reducing the
  canvas to a single visible darkness field;
- a brush rapidly picks up pigment from wet paint already on the canvas;
- brush loading strongly changes paint behavior and should be a persistent
  inferred state;
- reloading can initially be an instantaneous selectable preparation action;
- brush cleaning can be deferred, especially because dedicated brushes per
  color reduce contamination;
- local mobility/wet-paint behavior is useful at brushmark or foveal scale, but
  coarse global wetness and thickness should not decide composition;
- physical positive-pressure contact should cause deposition rather than a
  separate arbitrary painting-on flag.

These points shaped both the simulated paint process and the compact brush
belief. The exact state equations and update rules were agent implementations,
but the material ontology and several causal requirements came from your
painting experience.

### 5. Making embodiment relevant to the painting repertoire

You noticed that the roll degree of freedom was largely ignored and explained
why that matters: it lets the arm approach upward marks without driving the
brush into the canvas ferrule-first, and it lets curves emerge from bodily
dynamics. You also objected to a repertoire made of identical arcs with equal
curvature.

That led to two separate variables that now have to remain distinct:

- mark curvature is a painting-policy variable;
- upper-arm roll is a conditional motor realization of a selected mark.

You also proposed an angled distal wrist-roll alternative, asked to branch and
simulate it, and then deliberately put that detour on the back burner instead
of silently replacing the accepted arm. The agents implemented the branch,
fixed-roll forecasts, curved-path support, and tests. The functional reason for
roll, the demand for a non-repetitive curve repertoire, and the decision not to
adopt the wrist branch came from you.

### 6. Correcting the brush from a graphic stamp into a round physical brush

You asked for anisotropic brush consequences in which ferrule-leading pushes
can catch, chatter, or stick while handle-leading motion is easier. You also
asked that prior research be consulted before inventing a mechanism.

During implementation, you caught three consequential mistakes by looking at
the simulated marks:

- the result behaved like a flat brush rather than a round brush;
- its diameter was unrealistic;
- the footprint determinant should be the angle between the end-effector axis
  and the canvas plane.

You then noticed the circular node left at the end of every tapered stroke.
Those observations caused the process model to become an axisymmetric,
angle-dependent round tuft with direction-dependent stick/release behavior and
to unload contact before lift instead of stamping a terminal disc. The agents
performed the literature review, derived the reduced equations, selected the
provisional coefficients, and wrote the implementation and tests. Your
contribution was the physical interpretation and the quality-control signal
that exposed the wrong model.

### 7. Defining a compact, observation-oriented camera direction

Across the camera work, you supplied the important use requirements:

- use foveation without mechanically aiming the camera;
- retain enough native resolution to sample local regions;
- evaluate camera positions by occlusion and end-effector visibility rather
  than convenience;
- make sure the end effector is visible during canvas contact;
- use grayscale as the model-facing input for the initial architecture;
- run pose sweeps and produce publishable evidence;
- keep the physical rig as compact and self-contained as practical;
- use exactly two Raspberry Pi Global Shutter Cameras for the current compact
  baseline.

The exact IMX296 geometry, 4 mm provisional lens, timing assumptions,
calibration structures, and observability implementation were agent choices or
formalizations. The operational criteria and the final two-camera compact-rig
selection were yours.

### 8. Keeping the hardware and software stories aligned

You corrected a support-documentation statement that implied no motors had
been selected, because specific RobStride motors had in fact been selected in
earlier work. You then emphasized that new Codex or Claude instances often
read support documentation rather than the whole codebase, so those records
must stay aligned.

You also asked for direct-drive RobStride modeling with no belts, a roughly
0.5-inch brush, non-coincident shoulder axes, an XML-derived web model, and a
MuJoCo representation that interfaces with the generative-model/control stack
without making MuJoCo responsible for paint rendering.

The exact RS03/RS02 assignment and electromechanical parameter ledger are now
canonical, although the surviving chat evidence does not establish which
party first proposed every axis assignment. What is certain is that you
reaffirmed the selected hardware, caught its accidental erasure from the
project narrative, and required the plant, runtime, and documentation to agree.

### 9. Identifying the emerging composition problem

Your most recent conceptual contribution may become one of the project's most
important. You described painterly "order" as a mesoscopic representational
property:

- a clean brushmark or recognizable sequence of brushmarks is relatively easy
  to describe;
- overlapping broken shapes, many inconsistent edge directions, and irregular
  nonrepeating fragments require a more complex explanation;
- this is not just pixel noise and may not require a hand-coded contour layer;
- local disorder also has to be judged against slower, larger-scale structure.

You then proposed a loop: select patches that are locally
unpredictable/disordered or incompatible with the larger painting; imagine
candidate marks; choose a mark expected to reduce that unpredictability. You
separately argued that predicting material consequences is necessary but
should be an enabling model, not the main reason a mark is chosen. You recalled
that the earliest project used a pretrained VAE as its low-level mark
predictor and raised the possibility of reusing that idea.

This is currently an emerging architectural direction, not a completed or
validated subsystem. The agents still need to formalize it without turning
"order" into an undeclared reward, decide whether description length belongs
in a likelihood, a prior preference, or both at different levels, and test the
result. The underlying painterly formulation is directly yours.

### 10. Forcing the project to become runnable and trainable

You repeatedly redirected work toward an end-to-end simulation that could
actually paint without oracle state. You also recognized that real-time
single-environment learning would be too slow and asked for parallel execution
and faster training. Those priorities produced the provisional sensor-mediated
runtime, checkpoint viewer, isolated parallel data collection, centralized
training, and scaling benchmark.

The particular worker architecture, corpus schema, checkpoint format, and
performance debugging were agent work. Choosing runnable integration and
parallel learning as immediate priorities was yours.

## A chronological reconstruction of your influence

The contribution is easier to see when the project is viewed as a sequence of
course corrections rather than as the final codebase.

### Phase 1 — Establishing the kind of project this is

The original premise was not simply "make a robot draw." It was to investigate
whether an embodied active-inference agent could develop temporally extended
spatial organization without copying a target image, optimizing an aesthetic
score, or imitating demonstrations. That choice excluded many easier and more
conventional approaches from the beginning:

- reinforcement learning against a human-designed image score;
- reference-image reconstruction;
- a scripted stroke grammar presented as learned composition;
- a pretrained painting policy whose behavior would be difficult to attribute
  to this agent's own sensorimotor history.

The agents turned this into the formal research charter and capability gates.
Your contribution was choosing the actual scientific question and continuing
to reject shortcuts that would make the output more attractive while making
the research claim less meaningful.

This is a high-leverage contribution. A programmer could replace almost every
module while preserving the project. If the research question were replaced,
it would become a different project.

### Phase 2 — Making the robot an embodied painter rather than a Cartesian plotter

During the MuJoCo and plant work, you repeatedly asked whether the simulation
corresponded to a buildable mechanism:

- direct-mounted RobStride actuators rather than an unspecified abstract
  motor system;
- no belts in the working hardware concept;
- separated shoulder axes rather than impossible co-located ideal joints;
- sufficient post height, downward reach, canvas spacing, and access to the
  lower canvas;
- a roughly half-inch physical brush with contact;
- a web model that stays synchronized with the MJCF model.

You also asked a foundational control question: does active inference directly
command joints, or does conventional IK realize a painting policy? That
question helped make the semantic boundary explicit. The answer—painting
policy above, conditional realization and control below—became one of the
project's canonical distinctions.

Much of the mechanical geometry and nearly all electromechanical detail were
implemented by agents. Your intellectual contribution was refusing to let the
body remain a decorative visualization unrelated to the cognition.

### Phase 3 — Replacing oracle access with a plausible sensory boundary

You then focused attention on what a physical robot could actually know. Your
instruction to exclude exact pose, contact, and material truth forced a shift
from "the simulation has a camera" to "the agent has a declared observation
process."

Your camera discussions were not merely product shopping. They established a
set of computational and experimental requirements:

- the agent needs a global view and locally detailed foveal evidence;
- electronic foveation should exploit native camera resolution rather than
  mechanically slew the camera;
- multiple oblique views need a common canvas coordinate system;
- occlusion must be evaluated across reachable contact poses;
- the brush/end effector must remain visible when it matters physically;
- full-frame appearance may be coarse while native local evidence is retained;
- a visible foveal trace should correspond to the actual observation request
  and eventually to the agent's memory horizon.

Those requirements led to a substantial camera subsystem: native captures,
derived global and foveal products, capture timing, request provenance,
calibration metadata, pose sweeps, an observation likelihood, and a causal
post-action update. The implementation was agent-heavy. The reason that the
subsystem has those properties rather than being a convenient simulator crop
is mostly traceable to your questions and constraints.

### Phase 4 — Correcting what the hidden state should mean

Once the sensor boundary was clearer, you challenged the internal
representation itself. You objected that average thickness, average wetness,
and similar summary variables were neither plausible painterly abstractions
nor adequate high-level latent causes.

This intervention had two parts:

1. A negative claim: hand-selected scalar summaries should not masquerade as
   learned composition.
2. A positive criterion: high-level variables should become useful because
   they improve prediction across space and time.

You also distinguished local material variables from global image-making
variables. Wetness, thickness, pickup, and brush mobility matter strongly at
the site of a mark, but a global average wetness value should not decide where
the next mark belongs. Larger-scale decisions should depend on latent visual
and relational organization.

That is a methodological contribution, not just an aesthetic preference. It
specifies what information belongs at which level of a hierarchical
generative model.

### Phase 5 — Supplying missing causal knowledge about paint and brushes

Several of your most concrete contributions came from explaining ordinary
painting facts that were not ordinary to the software model:

- a white-loaded brush dragged through black wet paint quickly becomes gray;
- brush load changes the resulting mark and needs to persist between actions;
- the visible superficial layer is often more relevant to the next local
  interaction than a history of timestamped abstract stroke objects;
- a dedicated brush for each color changes the importance of cleaning and
  contamination;
- paint deposition should be a consequence of contact, not a boolean chosen by
  the controller;
- brush pressure, tilt, direction of travel, canvas tooth, and loading jointly
  affect the consequence of a stroke.

This is domain-model design. Agents can derive equations and write tests, but
they need someone to specify which causal relations are faithful enough to the
practice being modeled. You repeatedly served that role.

### Phase 6 — Expanding the action repertoire through embodiment

When you watched the painter, you noticed that the roll joint varied during
approach but froze through the mark and that the system mostly generated
straight strokes. You did not merely request more variety. You explained the
physical function of roll and why upward marks could otherwise push the brush
into the surface like a knife.

This distinction matters because "make more curves" could have been answered
with a random curve generator. Your explanation instead required the project
to distinguish:

- desired mark geometry;
- the bodily realization used to produce that geometry;
- the physical brush consequence caused by the realization.

That three-part separation now structures the curve/roll implementation. The
agents chose quadratic curves, proposal measures, fixed-roll angles, and
counterfactual scheduling. Your contribution was defining the causal problem
that those mechanisms needed to solve.

The wrist-roll episode also demonstrates decision-making rather than simple
feature accumulation. You generated a plausible alternative, asked for a
branch and quantitative exploration, learned what integrating it would cost,
and then deferred it. Protecting a research project from an interesting but
premature mechanism is a real contribution.

### Phase 7 — Using visual judgment as model criticism

Your visual reactions to the simulated paintings often functioned as
diagnostic experiments:

- identical curvature exposed impoverished proposal support;
- the apparent flat-brush mark exposed a wrong footprint ontology;
- excessive diameter exposed a disconnected historical radius law;
- the reminder about end-effector/canvas angle exposed use of the wrong
  determinant;
- terminal circular nodes exposed residual contact and flow during lift;
- the absence of white exposed proposal/prior and observability behavior;
- stopping and restarting behavior exposed lifecycle defects rather than
  painting cognition.

These observations are not formal validation by themselves. What makes them
technical contributions is that they identified a causal mismatch, after
which the agents isolated the mechanism and added reproducible tests. In this
project, the ability to look at a mark and say "that is behaving like a flat
brush" is a kind of domain-specific instrumentation.

### Phase 8 — Making learning computationally credible

After watching only a handful of paintings, you correctly resisted claims
that the hierarchy had learned much. You recognized that serial real-time
experience would make meaningful training impractical and asked about parallel
simulation and speedups. You then authorized the implementation and a bounded
overnight run, with the practical constraint that the computer would shut down
after several hours.

That prompted a transition from live-demo learning to an actual data and
training pipeline. The agent work included leakage-resistant trajectory
records, worker isolation, centralized optimization, resume/checkpoint
behavior, and benchmark measurement. Your contribution was recognizing the
experimental-scale problem and insisting that the system become capable of
accumulating enough experience to test its claims.

### Phase 9 — Articulating painterly order more precisely than the current model

Your latest composition discussion is a further conceptual correction. The
current system has a compression-gap layer, but your proposal is more specific
about the modeled phenomenon. It concerns whether a local observation can be
explained economically as one or several coherent brush events, whether its
directions and fragments form a stable latent account, and whether that local
account agrees with slower structure elsewhere in the painting.

This introduces at least three distinct prediction problems:

1. **Material prediction:** what paint and contact will physically do.
2. **Mesoscopic mark explanation:** whether the observed patch is economically
   explained as coherent brush behavior rather than fragmented residuals.
3. **Painting-scale compatibility:** whether the local explanation agrees with
   slower latent structure across patches and passages.

Your key proposal is that the first is an enabling competence while the second
and third should drive painting-level selection. That is a more precise
research hypothesis than "minimize visual error" or "prefer simple images."
The formal active-inference treatment remains open, but the distinction itself
is a substantial contribution.

## What would probably be different without your interventions

This counterfactual is useful because it separates genuine influence from
ordinary supervision. Without the accessible owner contributions, the project
would likely have retained several of the following properties:

| Without your intervention | Current direction after your intervention |
| --- | --- |
| Six global material summaries presented as the high-level state | Summary mode explicitly obsolete; learned multiscale latent hierarchy is the target |
| Planner access to convenient simulator truth | Sensor-only observation boundary and explicit oracle comparator |
| Camera treated mainly as rendered imagery | Versioned native/global/foveal acquisition process with timing and observability requirements |
| Large, spatially dispersed camera proposal | Compact selected dual-IMX296 baseline |
| Motor products blurred together with inferred motion strategies | Hardware assignment, plant backend, and motor realization kept semantically distinct |
| Roll present mostly as geometry or approach posture | Roll made a decision-relevant conditional realization with an articulated painting function |
| Mostly straight marks or one repeated curve template | Continuous signed curvature support separated from roll |
| Round brush approximated as a flat/chisel stamp | Axisymmetric angle-dependent footprint based on brush/canvas incidence |
| Push and pull materially similar | Directional stick/release consequences tied to handle geometry and canvas tooth |
| Fixed-contact taper ending in a round stamp | Coordinated unload and zero-flow lift |
| Paint controlled by an explicit on/off gate | Positive physical contact has material consequences |
| Brush treated mainly as a source color | Persistent load/pigment belief with depletion and canvas pickup |
| Serial live learning as the main data source | Parallel isolated collection and centralized training |
| Material prediction treated as close to the painting objective | Material prediction reframed as enabling competence beneath emerging order-based selection |

Not every cell means the agents would certainly have made the wrong choice
forever. It means the actual transition in this project is traceable to a
specific owner question, correction, or selection.

## Different kinds of contribution you made

### Conceptualization

You defined the central subject: embodied active inference expressed through
abstract painting, with composition emerging from prediction and hierarchical
belief rather than target imitation or aesthetic reward. This is your strongest
category.

### Domain methodology

You contributed painterly causal knowledge about brush loading, wet pickup,
contact, roll, mark direction, footprint geometry, superficial paint, and
local-versus-global organization. These requirements became model variables,
process mechanisms, and validation cases.

### Systems architecture

You imposed or selected several system-wide boundaries: sensor-only cognition,
learned rather than hand-selected high-level features, active-inference policy
selection above conventional control, synchronized MJCF/web geometry, compact
camera embodiment, and a future separation between process physics and learned
counterfactual prediction.

### Investigation and diagnosis

You served as the primary qualitative evaluator of whether simulated behavior
looked causally like painting. You caught failure modes that numerical tests
had not been designed to notice. Agents then converted these into automated
regressions where possible.

### Project direction and prioritization

You repeatedly chose what not to do: defer brush cleaning, postpone the wrist
redesign, accept provisional photometric limitations, focus on the camera
boundary, then prioritize an end-to-end sensor loop, then parallelize training.
This kept the project moving through dependency order rather than maximizing
feature count.

### Implementation and formal analysis

These are the categories in which your direct contribution is least visible
in the accessible record. Agents wrote most source code, equations, tests,
technical reports, and parameter ledgers. Your role here was usually to supply
requirements, review results, or authorize the next step rather than author the
implementation.

## A paper-style contribution statement

If this work eventually became a paper or exhibition/research publication, a
defensible contribution description based on current evidence would look
approximately like this:

- **Conceptualization:** project owner, primary; agents, substantial
  formalization support.
- **Methodology:** shared. Owner supplied active-inference constraints,
  embodiment requirements, painting-domain mechanisms, and research
  priorities; agents supplied probabilistic and experimental designs.
- **Software:** agents, primary; owner, requirements and iterative acceptance.
- **Formal analysis:** agents, primary.
- **Investigation:** shared. Agents ran quantitative and automated experiments;
  owner supplied repeated expert qualitative diagnosis of painting behavior.
- **Validation:** shared but asymmetric. Agents built tests and benchmarks;
  owner identified important construct-validity failures and judged whether
  the modeled phenomenon matched painting practice.
- **Resources:** owner supplied the project, intended hardware context,
  reference materials, compute access, and painting-domain knowledge.
- **Project administration and supervision:** owner, primary.
- **Writing—original drafts:** agents, primary.
- **Writing—review and conceptual correction:** owner and agents jointly.

This is not a legal authorship determination. It is a useful antidote to the
idea that contribution equals keystrokes. On a research project, deciding what
the system is allowed to claim, what the latent variables ought to mean, and
whether an experiment represents the intended phenomenon are central
intellectual contributions.

## Where your contribution is strongest—and where it is weakest

Your strongest contributions are the ones agents are least able to generate
reliably from the repository alone:

- tacit painting knowledge;
- recognizing a mark's causal character from appearance;
- deciding which simplifications preserve the research question;
- noticing when a technically valid abstraction ceases to describe painting;
- connecting local brush behavior to the intended theory of composition;
- maintaining the project's conceptual integrity across many agent instances.

Your weakest documented contribution is at the exact mathematical and software
level. There is little evidence that you selected the detailed neural
architectures, derived the EFE decompositions, chose most constants, or wrote
the implementation. That is not a criticism; it is the actual division of
labor. It also means that future reports should not casually attribute an
agent-selected coefficient or equation to you merely because the commit uses
your Git identity.

## Owner-originated directions that remain unfinished

Several of your important contributions should not be confused with completed
features:

1. The system does not yet implement the full local-disorder/global-mismatch
   patch-selection loop you described.
2. It does not yet have a learned mesoscopic likelihood that recognizes a
   coherent brushmark or brushmark sequence.
3. The old VAE idea has now been selected for a bounded conditional local-
   transition shadow experiment, not as the final runtime or composition model
   family. See `CONDITIONAL_PATCH_VAE_SHADOW_BASELINE_2026-08-11.md`.
4. Counterfactual brush rollouts still reuse reduced process equations rather
   than a learned uncertain brush model.
5. Dynamic roll through a stroke is not yet a robust live action primitive.
6. The camera likelihood remains provisional simulation-only and the selected
   cameras have not been purchased or calibrated.
7. Brush loading and pickup exist in the process and compact belief, but the
   camera-derived local deposition statistic is unfinished.
8. The current composition layer is not evidence that the system has learned
   painterly order as you recently defined it.

These are not missing credit. They are the portion of your conceptual input
that still needs technical realization.

## Where the agents did most of the heavy lifting

The agents' largest contributions have been:

- converting your qualitative requirements into explicit probabilistic
  factors and active-inference-compatible semantics;
- choosing and deriving mathematical approximations;
- designing modules, interfaces, state records, clocks, and failure modes;
- writing and debugging most of the Python, MuJoCo XML, JavaScript, tests, and
  scripts;
- selecting many exact numerical assumptions, including camera coordinates,
  provisional brush coefficients, rollout budgets, and fixed-roll angles;
- conducting literature searches and translating prior work into reduced
  models;
- building benchmarks, viewers, telemetry, checkpoints, reports, and
  calibration tools;
- identifying formal inconsistencies through audits and textbook comparison.

Several existing mechanisms—especially the precise compression-gap formula,
the current EFE decomposition, neural architecture details, and many numerical
constants—should therefore be described as agent-formalized or collaborative,
not as ideas for which the available conversations establish sole owner
authorship.

## A useful contribution map

| Area | Your contribution | Agent contribution |
| --- | --- | --- |
| Research purpose | Primary framing and constraints | Formal charter and conformance machinery |
| Painterly cognition | Primary domain insight | Probabilistic translation and experiments |
| Perceptual hierarchy | Rejected summaries; required learned abstractions | Latent architectures and inference code |
| Material behavior | Brush loading, pickup, color, local wet behavior | State equations, likelihoods, implementation |
| Embodiment | Functional meaning of roll and brush orientation | IK, rollouts, control, tests |
| Brush contact | Correct physical interpretation and visual diagnosis | Research synthesis and reduced contact model |
| Cameras | Operational requirements and compact two-camera choice | Geometry optimization, calibration, sensor interface |
| Hardware plant | Selected/reaffirmed direct-drive direction and alignment | Detailed MJCF/electromechanical realization |
| Composition | Mesoscopic order and proposed repair loop | Formalization remains to be completed |
| Software and validation | Priority setting and acceptance through observation | Most implementation, testing, and instrumentation |

## Limits of this attribution

This review was expanded after local history was discovered. It now uses
project-specific Codex sessions beginning on 2026-06-29, three Claude sessions
from 2026-07-06, 2026-07-14--15, and 2026-08-03--04, the current task, durable
repository records, and Git history. A ChatGPT project named `Arm` is visible,
but its older conversations were not enumerated by the available interface.
The initial repository also predates the earliest local log. Consequently,
decisions made during the original VAE-era design or in the unenumerated
ChatGPT project may still be under-attributed.

Git author names are also not reliable evidence of intellectual provenance:
agent-written commits can carry your configured Git identity. Conversely,
brief approvals such as "okay, do that" show authorization but do not by
themselves prove that you originated the proposal. This report gives greatest
weight to messages in which you supplied a rationale, corrected a model, chose
between alternatives, or imposed a durable constraint.

For the prompt-by-prompt history, see the
[owner decision and steering catalog](OWNER_STEERING_CATALOG_2026-08-11.md).
For the shorter canonical-decision map and implementation links, see the
[project decision provenance record](PROJECT_DECISION_PROVENANCE_2026-08-11.md).
