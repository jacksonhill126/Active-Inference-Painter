# What We Changed Before Building The Visual Hierarchy

Date: 2026-08-26

Policy profiles: `m1-formal-policy-baseline-v0` (default) and
`legacy-material-composition-diagnostic-v0` (explicit diagnostic only).

## The short version

We removed a circular dependency from the research plan and made the clean M1
baseline stop using the old coarse-material "composition" machinery to choose
marks or decide when to stop.

The old machinery is still available for controlled comparisons. It is simply
no longer allowed to masquerade as the visual hierarchy we actually want.

## Why the plan had become circular

M1 is meant to establish what the current active-inference baseline really
does before M2 builds the visual model. But two remaining M1 tasks had been
rewritten to require the future visual model:

1. AI-109 was waiting for completed visual paintings and hierarchy learning
   curves.
2. AI-110 was waiting for a tone/edge/mass representation before deciding what
   to do with the composition preference.

Those are M2 jobs. At the same time, M2 formally waited for M1. The code could
keep moving, but the milestone could never honestly close.

## What is disabled now

By default, the painter no longer uses:

- the coarse-material compression gap as a terminal preference;
- canvas or relational transition risks from that old hierarchy;
- its passage-trajectory terms; or
- the change in compression gap as an extra reason to stop.

The ordinary terminal coverage preference remains. Immediate `stop` remains
available. VFE and EFE decomposition, motor forecasts, camera/sensor boundary,
and local transition learning remain intact.

This gives us a cleaner experimental baseline. When we later collect visual
trajectories, an unaccepted self-trained notion of composition is not quietly
shaping the examples that will be used to evaluate its replacement.

## What happened to the earlier learning-curve task

The material-model learning curves did answer their M1 question: larger
generic networks were not the answer, the material VAE did not earn promotion,
and the best mixture was still uncalibrated. That is a useful negative result,
so AI-109 can close for M1.

The unfinished work has not been waved away. The registered visual mark model,
recursive visual rollouts, full-canvas masses, and hierarchy learning curves
now live in M2, where they belong.

## What this does not mean

This change does not claim that M1 is finished. We still need to decide exactly
what persists through checkpoints, measure where runtime is spent, and capture
three reproducible baseline replicas before the owner lock decision.

It also does not reject the idea of composition preferences. It says that a
preference over painting structure must be grounded in the admitted visual
generative model and protected against self-reinforcement. Until that model
exists, zero is the scientifically cleaner preference weight.

The technical record is `docs/M1_GATE_REPAIR_TECHNICAL_2026-08-26.md`; the
visual destination remains `docs/VISUAL_GENERATIVE_MODEL_BOUNDARY.md`.
