# AI-109 Predictive Learning Curves — Owner Brief

Date: 2026-08-12

## August 26 scope update

The material-model experiment described below now closes AI-109 for M1. Its
unfinished registered visual and hierarchy work moves to M2 rather than
keeping M1 dependent on the architecture it is meant to unblock. Nothing below
turns truncations into completed paintings; the original evidence boundary is
unchanged. See `docs/M1_GATE_REPAIR_OWNER_BRIEF_2026-08-26.md`.

## Where we landed

We learned something decisive about the low-level material predictor, but we
have not finished the composition half of the task.

The current ordinary CNN is learning real mark consequences. Giving it more
independent trajectories helps somewhat. Making it much larger does not help,
and the VAE does not currently justify its extra machinery. The best local
model is a new, explicit mixture that represents two possibilities:

- this part of the patch mostly persists; or
- the mark causes a continuous material change.

That distinction improves the probability it assigns to held-out camera-
posterior transitions by a large margin. Its average test NLL is 1.392 nats
better than the base CNN, far larger than the run-to-run variation. It also
stays reasonably stable when its own predictions are fed back recursively for
eight marks.

This is the strongest evidence so far that our previous local likelihood had
the wrong shape. It was trying to describe a very sparse transition—many
nearly unchanged cells and a smaller set of genuinely changed cells—with one
Gaussian bell curve per ensemble member.

## Why it still is not ready

A generative model must do more than put a high average probability on the
right answer. Its stated uncertainty must also mean what it says.

If a model gives a 90% prediction interval, about 90% of held-out observations
should land inside it. The new mixture's nominal 90% interval contains about
95.7%. More revealingly, its nominal 50% interval contains about 90.6%.
That is like a weather forecaster whose “coin-flip confidence band” captures
the outcome nine times out of ten: the central probability mass is much too
concentrated or broadly allocated to be interpreted literally.

The mixture is substantially better than the CNN and VAE on this measure, but
all three fail the calibration gate. No model is being promoted into live
policy inference.

## What the VAE result means

The VAE was a reasonable hypothesis. It could, in principle, represent
several different material outcomes for the same current patch and mark.
In this experiment it did not make meaningful use of that latent variable.
Its held-out likelihood was only 0.057 nats better than the base CNN, which is
smaller than the CNN's 0.098-nat seed variation. At eight recursive marks its
mean error was about 0.225, compared with 0.087 for the CNN and 0.074 for the
new mixture.

So the conclusion is not “VAEs are bad.” It is narrower: this material-field
cVAE, on this small camera-posterior corpus, is not earning a place in the
active model. It did not train on registered visual pre/post patches, so it
says nothing about whether your original visual mark-consequence VAE would be
useful. We should keep this code as a controlled shadow comparison but stop
treating more training of this material-field model as the default next move.

## What the data and model-size curves say

The experiment used three random seeds throughout and kept the same three
validation and three test trajectories fixed.

- Going from 3 to 10 training trajectories improved CNN test NLL by 0.159
  nats, just enough to count as larger than seed variation.
- The 6-trajectory CNN was actually a little better than the 10-trajectory
  CNN. With so few independent episodes, which physical conditions land in a
  subset still matters a lot.
- A small 26,964-parameter CNN slightly beat the 137,028-parameter baseline.
  A 716,772-parameter CNN was substantially worse on both training and test
  data. We are not being held back by a lack of generic network capacity.
- Five bootstrap members materially improved held-out density over one member,
  but did not fix uncertainty calibration.

The practical message is: spend effort on the probability model and on more
independent episodes, not on making the network bigger.

## Why composition is still open

The accepted AI-108 corpus contains sixteen independent trajectories, but all
of them end because the collector reached a fixed number of marks. None ends
because the painter actually selected `stop`.

For low-level mark prediction that is fine: every adjacent camera-posterior
transition is still a valid example. For a slow composition model it is not
fine. Calling the final frame a completed painting would teach the hierarchy
that “the data collector's timer expired” means “this painting is resolved.”

I deliberately did not manufacture those labels. At the time of this record,
AI-109 therefore remained active. The 2026-08-26 gate repair closes its M1
material-model question and moves the hierarchy branch to M2, where genuine
terminal visual paintings remain required.

## What I recommend next

The most pressing next step is a small, explicit terminal-corpus extension:

1. Run independent sensor-path episodes in which `stop` is always available
   and record whether termination was genuinely selected.
2. Keep fixed-horizon or watchdog endings as truncations, never as completed
   paintings.
3. Split whole paintings into train/validation/test before extracting any
   hierarchy examples.
4. Then run the same three-seed data/capacity curve for the slow canvas and
   relational models.

In parallel, the local mixture needs one focused calibration iteration. We
should inspect whether it assigns the right probability to the persistence
versus consequence components, separately inside and outside the stroke
footprint. A true spike/hurdle persistence event or a bounded/correlated
consequence distribution may be more appropriate than two ordinary Gaussian
components. Whatever we try must remain a normalized likelihood, not turn into
an aesthetic reward or weighted score.

The detailed methods, complete table, limitations, and reproducibility paths
are in `docs/AI109_PREDICTIVE_LEARNING_CURVES_TECHNICAL_2026-08-12.md`.
