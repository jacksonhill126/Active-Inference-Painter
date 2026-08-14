# What The New Conditional VAE Does

Date: 2026-08-11

## Correction made on 2026-08-12

I previously described this as the first implementation of the VAE idea you
brought back from the project's early design. That was incorrect. Your idea
was a stochastic, action-conditioned **visual** predictor: fresh image patch
plus proposed mark and brush context in, plausible post-mark image patches
out. The model described here predicts coarse inferred material fields.

The experiment and its negative result remain useful for that material target,
but neither implements nor argues against your visual VAE proposal. See
`docs/VISUAL_GENERATIVE_MODEL_BOUNDARY.md` for the corrected boundary.

## The short version

We implemented a careful shadow cVAE over recorded local material-posterior
transitions. It can train on recorded painting transitions and tell us how
well it predicts that declared target, but it has no vote in what the live
painter chooses to do.

Its job is narrow and concrete:

> Given what the local canvas currently seems to contain, the mark we intend
> to make, the motor realization, and what we believe is loaded in the brush,
> what local material outcomes are plausible?

That is the right job for a conditional VAE. It is not yet the model that looks
at a patch and says, "this passage is confused," or "these overlapping fragments
can be better explained by one decisive brushmark." That is the next,
mesoscopic representational problem.

## Why the VAE is useful here

The existing local CNN predicts one Gaussian outcome for each model member. A
VAE adds a hidden random variable `z`. This lets one declared action have more
than one plausible consequence—for example, a slightly broken edge, a fuller
deposit, or a different local pickup/release pattern—without pretending that
those differences were all specified in the action.

The ensemble adds another distinction:

- variation across `z` asks, "what different outcomes does this learned model
  think are possible?";
- disagreement across independently bootstrapped members asks, "how unsure are
  we about the learned model itself?"

Those are different uncertainties and should affect active inference
differently later. The implementation keeps them separate rather than merging
them into one confidence number.

## What data it sees

The new corpus record contains:

- the canvas material belief before the mark, including its uncertainty;
- the mark geometry, width, amount, tone, and curvature;
- whether the mark was realized with Cartesian IK or another allowed motor
  realization;
- the inferred brush load and pigment belief before the stroke;
- the canvas material belief after a causally later camera observation.

It does **not** see the simulator's exact hidden paint grid, exact held paint,
or exact bristle state. That matters because a model trained on those values
could look excellent in simulation and then become impossible to run from real
cameras.

Older training files did not record brush belief. They still load, but the
model receives an explicit "brush context unavailable" bit. Missing data is
not silently called an empty brush.

## What the training score means

The VAE minimizes variational free energy, equivalently the negative ELBO. It
has two separately reported pieces:

1. how surprising the observed after-patch is under the predicted material
   likelihood;
2. how much information the latent explanation `z` had to use beyond its
   simple normal prior.

This is a legitimate generative-model learning objective. We are **not** using
VAE reconstruction error as "ugliness," "disorder," or a reward for the
painter. A physically surprising mark and a compositionally unresolved patch
are not the same thing.

## How we will know whether it is actually learning conditions

The training report evaluates unseen trajectories and then deliberately gives
the model wrong conditions:

- shuffled mark/action;
- removed motor-realization channels;
- shuffled brush belief, when the test set has enough real brush variation.

If the correct condition predicts the held-out result better, the reported gap
is positive. If the gap is zero or negative, that capability has not been
demonstrated. This guards against a common failure where a "conditional" VAE
quietly ignores the action or its latent variable.

## What is finished and what is not

Finished now:

- the model architecture;
- the whole-trajectory leakage boundary;
- recording pre-stroke brush belief for new data;
- backward compatibility with the old corpus;
- offline training and isolated checkpointing;
- held-out likelihood, condition-ablation, uncertainty, and rough calibration
  reports;
- 10 focused passing tests.
- a later three-seed AI-109 comparison showing that this material cVAE did not
  improve beyond seed variation and became unstable in recursive rollout.

Not finished—and not tested by this model:

- retaining registered pre/post camera images;
- predicting visible tone and oriented boundaries after an action;
- evaluating a visual interaction latent against deterministic visual
  baselines;
- calibrated multistep visual rollout;
- using sequences of marks to explain local visual order;
- representing compatibility between a local patch and the larger painting.

## Corrected recommended next move

Do not make a larger material-field cVAE the default next experiment. Retain it
as a measured negative control. Collect a trajectory-isolated corpus of
registered pre/post images with camera, action, brush, motor, crop, and
termination provenance. On identical splits, compare a deterministic visual
predictor, a stochastic action-conditioned visual model, and appropriate
identity/no-action baselines using normalized likelihood, calibration,
tone/edge fidelity, condition ablations, and multistep rollout.

Only after the local visual model is credible should the slower hierarchy be
tested for compact tone masses, continuous uncertain boundaries, and
compatibility with larger painting structure. The canonical sequence and
admission gates are in `docs/VISUAL_GENERATIVE_MODEL_BOUNDARY.md`.
