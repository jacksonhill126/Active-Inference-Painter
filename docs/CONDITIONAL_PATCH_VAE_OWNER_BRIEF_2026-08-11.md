# What The New Conditional VAE Does

Date: 2026-08-11

## The short version

We implemented the first careful version of the VAE idea you brought back from
the project's early design. It is currently a **shadow model**: it can train on
recorded painting transitions and tell us how well it predicts them, but it has
no vote in what the live painter chooses to do.

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

Not finished:

- collecting enough diverse transitions to learn a good model;
- demonstrating that it beats the existing local transition CNN;
- showing calibrated uncertainty by tone, curvature, load, wetness, reach, and
  motor realization;
- connecting it to counterfactual planning;
- using sequences of marks to explain local visual order;
- representing compatibility between a local patch and the larger painting.

## Recommended next move

The next move is evidence, not integration. Collect a larger v2 corpus with
real variation in brush load, black/white tone, curvature, wet overlap, motor
realization, canvas region, and mark size. Train the cVAE ensemble and the
existing CNN on the exact same trajectory splits. Compare:

- held-out density;
- calibration;
- condition-ablation gaps;
- sample diversity and physical plausibility;
- multi-step rollout drift.

If the VAE wins those tests, we can decide how it should replace or augment the
one-step material likelihood. In parallel, we can design the layer you were
really pointing toward: a model that tries to explain a patch as a small set of
coherent brush events and measures whether that explanation is economical and
compatible with slower painting structure.

That division of labor is important: the VAE learns **what marks can do**; the
later hierarchy learns **whether the painting is becoming more intelligibly
organized**.
