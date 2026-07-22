# Active-Inference Painter Research Charter

Status: July 22, 2026

## 1. Project identity

Active-Inference Painter is an embodied cognition research project expressed
through a robotic abstract painter. Its intended balance is approximately:

- 80 percent research project;
- 20 percent conceptual art project.

The paintings are not primarily entries in an image-quality competition. They
are persistent, inspectable traces of perception, prediction, action, error,
and belief revision. Their aesthetic and conceptual force matters, but it must
not substitute for a clear account of the mechanism that produced them.

## 2. Central research question

> Can an embodied, hierarchical active-inference agent develop temporally
> extended spatial organization through foveated perception and sensorimotor
> prediction, without image targets, aesthetic rewards, or demonstrated
> painting policies?

The project investigates whether behavior resembling abstract composition can
arise from multiscale generative modeling, active observation, bodily
constraints, uncertainty, and persistent latent beliefs. It does not assume
that compression, prediction, or self-evidencing is equivalent to beauty.

"Develop" in this question means that organization should arise through
inference and learning from the agent's sensorimotor history. Structure that is
directly supplied by code, demonstrations, pretrained policies, or aesthetic
labels must be identified separately.

## 3. Research intent

The primary object of study is the coupled process formed by:

- a body with constrained and uncertain joint dynamics;
- a foveated visual system that must choose where to look;
- a persistent, materially changing canvas;
- a hierarchical generative model operating at multiple spatial and temporal
  scales;
- policy inference over gaze, motion, marks, passages, and stopping;
- learning from the sensory consequences of action.

The desired result is not necessarily conventionally attractive painting. A
successful result is an agent whose behavior can be understood as a coherent
consequence of explicit beliefs, uncertainties, prior preferences, and learned
sensorimotor regularities, and whose organization changes in interpretable
ways when those mechanisms are altered.

## 4. Governing scientific commitments

### 4.1 Active inference is the cognitive architecture

Painting decisions must be grounded as likelihoods, transition priors, prior
preferences, precision beliefs, variational free-energy terms,
expected-free-energy terms, or policy priors and posteriors.

Ordinary rewards, aesthetic scores, and weighted heuristic objectives must not
be relabeled as active inference. Conventional control and safety engineering
may realize a selected policy below the painting-policy boundary, but must not
silently choose that policy.

### 4.2 The generative process and generative model remain distinct

The simulated or physical world generates observations. The agent's
probabilistic model predicts those observations and hidden causes. The model
must not receive simulator state or physical variables that a real robot could
not sense.

### 4.3 Uncertainty is functional, not decorative

Likelihood variance, transition uncertainty, parameter uncertainty, and
precision weighting must affect inference in declared ways and be tested for
calibration. A deterministic error or embedding distance is not free energy by
itself.

### 4.4 Approximations are named

Amortized inference, neural density models, finite candidate sets, ensembles,
frozen feature encoders, deterministic coarse-graining, and conventional SGD
are legitimate engineering approximations when their roles and limitations are
explicit. They must not be presented as exact Bayesian inference.

### 4.5 Negative results must remain interpretable

A failure to produce global organization is scientifically meaningful only
after the model passes the lower-level capability gates in this charter.
Otherwise the result may only show inadequate perception, training, proposal
support, or predictive capacity.

## 5. Scope

### In scope

- Foveated and active visual observation.
- Pixel-local paint and brush transition learning.
- Multiscale visual, material, and relational latent states.
- Explicit proprioceptive, current, contact, and energetic modalities.
- Joint-space and Cartesian consequences of painting policies.
- Temporally extended mark, polyline, passage, and passage-plan inference.
- Online adaptation supported by offline self-supervised pretraining.
- Controlled use of pretrained open-weight perceptual representations.
- Simulation-to-hardware transfer when the cognitive architecture is stable
  enough to justify it.

### Out of scope for the current research program

- Optimizing against human ratings or an aesthetic reward model.
- Reproducing reference images as the primary task.
- Training a video foundation model from scratch.
- Building a general-purpose robot intelligence or complete artificial life.
- Treating a proprietary model's judgment as the agent's preference model.
- Solving every aspect of camera realism, paint rheology, and robot mechanics
  before testing the central cognitive hypotheses.
- Claiming a general theory of human artistic composition.

## 6. Working hypotheses

These hypotheses organize experiments; they are not assumptions that the
project must vindicate.

### H1: Active foveation changes spatial organization

An agent that chooses where to sample visual detail will develop different
relational beliefs and longer-range spatial dependencies than an otherwise
matched agent receiving a uniform full-canvas observation.

### H2: Embodied prediction shapes mark vocabulary

Explicit uncertainty and energetic consequences in joint space will favor
different trajectories, including different curvatures and continuities, than
an agent that treats Cartesian marks as perfectly realizable commands.

### H3: Slow latent states support organization across passages

Latents updated at passage and painting timescales will carry predictive
information across local mark sequences that cannot be captured by independent
one-mark rollouts.

### H4: Relational prediction can support nonsemantic composition

Spatial relationships among regions, motions, and previous marks can become
predictively useful without object labels, reference images, or explicit
preferences for symmetry, balance, motifs, or other named aesthetic devices.

### H5: Epistemic action is not sufficient by itself

Expected information gain may produce useful exploration, but may also favor
noise, novelty, or unstable sampling. The project should test what combination
of embodiment, temporal hierarchy, precision, and viable action constrains it
into organized behavior.

### H6: Pretrained perception helps but imports structure

A frozen self-supervised visual representation may make the project feasible
under limited compute, while also importing visual invariances and developmental
history. Its contribution must be measured against a from-scratch observation
model rather than treated as neutral.

## 7. Capability gates

Interpretation of high-level painting behavior requires evidence that the
following capabilities work in sequence:

1. **Observation:** Relevant material, spatial, bodily, and visual changes are
   represented rather than discarded.
2. **Local prediction:** The model predicts individual mark consequences on
   held-out transitions with calibrated uncertainty.
3. **Embodied prediction:** It distinguishes motor realizations and predicts
   their proprioceptive, energetic, contact, and canvas consequences.
4. **Relational representation:** Separated regions and marks influence a
   shared latent belief, and this influence survives controlled transformations.
5. **Temporal persistence:** Slow beliefs retain predictive information across
   marks and passages without merely copying the latest observation.
6. **Policy sensitivity:** Changes in beliefs and precisions produce
   corresponding, explainable changes in policy posteriors.
7. **Generalization:** Predictive performance extends to held-out geometries,
   canvas arrangements, and motor conditions.

Failure at a gate should redirect effort toward that mechanism before visual
incoherence is interpreted as evidence about composition.

## 8. Evidence and evaluation

No single metric defines successful painting. Evaluation should combine
mechanistic measurements, controlled interventions, and qualitative records.

### Predictive and inferential measures

- Held-out observation and transition log likelihood.
- Multi-step predictive degradation.
- Posterior calibration and ensemble calibration.
- VFE decomposition during state inference.
- EFE decomposition during policy inference.
- Predictive advantage of hierarchical models over flat baselines.
- Latent effective dimensionality and collapse diagnostics.
- Dependence of future predictions on slow latent states.

### Behavioral and spatial measures

- Persistence and transformation of relations across passages.
- Long-range dependence between separated marks and passages.
- Changes caused by spatial scrambling, rotation, translation, or occlusion.
- Gaze allocation relative to uncertainty and subsequent prediction error.
- Energy, current, contact, and execution-prediction error.
- Diversity and stability of inferred motor and passage policies.

These are measurements of mechanism and organization, not proxies for beauty.

### Controlled comparisons

Important claims should be supported by matched ablations where feasible:

- foveated versus uniform observation;
- local-only versus hierarchical prediction;
- Cartesian-only versus embodied joint-space prediction;
- fast-only versus fast-and-slow latent dynamics;
- from-scratch versus frozen pretrained visual features;
- intact versus spatially scrambled observations;
- learned uncertainty versus fixed uncertainty;
- passage inference versus independent mark inference.

Results should include unsuccessful and ambiguous runs rather than only selected
paintings.

## 9. Policy on pretrained models

Pretrained components are permitted when they preserve feasibility and their
epistemic role is explicit.

They may provide:

- deterministic visual features;
- initialization for an observation model;
- generic spatial or temporal representations;
- offline teacher targets for a smaller model.

They may not provide:

- aesthetic evaluation;
- terminal outcome preferences;
- an undeclared painting policy;
- opaque values inserted into VFE or EFE;
- claims of learning that actually originate in the pretrained model.

A frozen encoder is treated as an inherited sensory transformation. The active
inference model must still define explicit probabilistic beliefs and densities
over the resulting observations and transitions. Raw material and
proprioceptive pathways should remain available where pretrained features may
discard task-relevant detail.

## 10. Feasibility constraints

This is a single-investigator project. Near-term model-training expenditure is
limited to approximately USD 1,000. Recommendations must therefore account for
implementation time, debugging burden, experimental repetitions, inference
cost, storage, and maintainability, not merely theoretical possibility.

The practical scaling policy is:

> Preserve conceptual integrity; economize on scale and scope.

Consequences of this policy include:

- Do not train foundation-scale perceptual models from scratch.
- Prefer frozen open-weight encoders with small probabilistic adapters when
  pretraining is empirically justified.
- Cache expensive frozen representations for repeated experiments.
- Train compact domain-specific transition and inference models.
- Add one modality or hierarchy change at a time.
- Preserve simple baselines and reversible interfaces.
- Increase capacity only after learning curves show underfitting.
- Prefer experiments in which a negative result is still informative.

Every substantial proposal should state:

1. its scientific intent;
2. its relationship to the central question;
3. whether it is active inference or conventional supporting engineering;
4. expected implementation and compute burden;
5. what is compromised or imported;
6. what observation could falsify or weaken its rationale;
7. a smaller credible alternative.

## 11. Role of the conceptual art component

The art component should make the research process experientially legible. The
canvas can function as an external memory of prediction, correction, attention,
and bodily constraint.

Useful presentation material includes:

- completed and interrupted paintings;
- gaze trajectories and foveated observations;
- passage boundaries and motor realizations;
- synchronized belief, precision, VFE, and EFE traces;
- prediction failures and subsequent adaptation;
- comparisons between controlled architectural variants.

The work should not hide failure or curate only attractive outcomes. The
relationship between mechanism and artifact is part of the work.

## 12. Near-term research program

### Phase A: Establish a trustworthy baseline

Validate current local paint prediction, embodied forecasts, uncertainty,
checkpointing, and hierarchy behavior. Resolve known data-scale and long-run
training problems before interpreting compositional output.

### Phase B: Introduce foveated observation

Define an explicit gaze state, observation likelihood, sensory precision, and
gaze policies. Compare active foveation with matched uniform-observation and
random-gaze baselines.

### Phase C: Build the probabilistic perceptual hierarchy

Connect foveated visual observations to fast and slow stochastic latent states.
Demonstrate held-out prediction, calibrated uncertainty, temporal persistence,
and sensitivity to spatial interventions.

### Phase D: Test temporally extended embodied organization

Integrate gaze, joint-space consequences, marks, and passages within policy
inference. Test whether slow beliefs causally affect later passages and whether
embodiment changes the learned mark vocabulary.

### Phase E: Evaluate pretrained perception

Compare a small from-scratch encoder with frozen open-weight spatial and video
encoders using the same probabilistic model above them. Retain pretraining only
if it provides a clear predictive or feasibility advantage, and document what
it imports.

### Phase F: Transfer deliberately

Move mechanisms to a higher-fidelity simulator or physical robot only when the
transfer answers a research question that the existing process cannot answer.
Treat simulator fidelity, calibration, and hardware safety as necessary
engineering work rather than evidence for active inference by themselves.

## 13. Current open questions

- What slow latent variables are both discoverable and predictively necessary?
- Which outcome preferences are required for sustained, nontrivial behavior
  without smuggling in aesthetic objectives?
- Can foveation create useful spatial-temporal structure without producing
  fixation loops or indiscriminate novelty seeking?
- How should uncertainty be represented across pixels, regions, body states,
  and temporal levels under the compute budget?
- How much visual competence comes from embodiment, and how much must be
  inherited through pretraining?
- What proposal mechanism permits genuinely new organization without becoming
  a hidden compositional algorithm?
- Which findings survive transfer from simulation to a physical arm?

These questions should remain visible. Architecture should be selected to make
them answerable, not to conceal them behind increasingly elaborate output.

## 14. Standard for project decisions

A direction is worth pursuing when it has a clear relationship to the central
question, is feasible for a single investigator, preserves the declared active-
inference boundary, and can produce an informative result even if its preferred
hypothesis fails.

Technical novelty, model size, visual polish, and apparent intelligence are not
sufficient reasons on their own.
