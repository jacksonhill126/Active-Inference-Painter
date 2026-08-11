# What the accepted painting corpus now gives us

The short version: we now have a clean baseline dataset that the learning code
can actually use without train/test leakage, and it covers the robot and mark
conditions we said it needed to cover. This closes AI-108. It does not yet give
us terminal paintings for composition learning.

## What is in it

The accepted corpus has 16 independently simulated painting episodes and 256
recorded marks. Ten episodes are training data, three are validation data, and
three are test data. The split happens at the episode level before local
patches are cut out.

That episode-level split is important. If mark 12 and mark 13 from the same
painting landed in train and test, the test set would partly repeat the same
canvas history. A model could appear to generalize while mostly recognizing a
nearly identical local situation. Keeping the entire painting on one side of
the split removes that leakage path.

Every split contains examples of:

- black and white paint;
- blank canvas and painting over existing marks;
- interior and edge marks;
- narrow, medium, and broad marks;
- short, medium, and long marks;
- straight marks and gentle/strong curves in both directions;
- vertical, horizontal, and diagonal directions;
- center, middle, and outer-canvas reach;
- neutral IK, fixed roll in both directions, and dynamic roll sweeps in both
  directions.

The exact counts are in
`docs/AI108_CORPUS_TECHNICAL_2026-08-11.md`. The rarest important cases are the
dynamic sweeps in the test set, so a larger corpus should add more of those.

## What the model is allowed to see

The records contain what the active-inference runtime believes before and
after a mark: the camera-derived material posterior, its uncertainty, the
selected mark, the selected motor realization, and a compact belief about how
loaded the brush is.

They do not contain the simulator's exact hidden canvas, exact contact state,
exact paint held in the brush, or exact bristle microstructure as training
inputs. That keeps the dataset aligned with the eventual embodied problem:
the learner must model consequences from beliefs and observations, not from
state variables a physical robot will never receive.

## What “live scale” means here

The process canvas is simulated at 256 pixels. For speed, the provisional
inference/camera path still uses a reduced 64-pixel acquisition configuration
and a 16 by 16 material-belief grid. Those three resolutions are now recorded
separately.

This is still an uncalibrated simulation baseline. “256 process canvas” does
not mean that the cameras, optics, noise, or brush physics are hardware
calibrated or sensor-equivalent.

## Two things this corpus does not solve

First, it has no real stop decisions. Every episode was cut off after 16 marks.
That is fine for learning local mark consequences, but a truncated canvas is
not evidence that the agent believed a painting was finished. We cannot use
these canvases as terminal composition examples.

Second, it does not contain genuine observed wet-over-wet state. The current
camera likelihood does not claim to measure bulk wetness. We deliberately did
not copy exact simulator wetness into the belief just to fill a dataset bin.
The corpus distinguishes blank canvas from existing/overlapping paint; wetness
needs its own legitimate observation-model work.

## What changed operationally

The collector now starts a fresh simulator runtime for each trajectory. This
removed the earlier slowdown where later paintings in one long-lived runtime
took progressively longer. Eight 16-mark trajectories took about 4.3 minutes
with the three-way fixed-roll profile and 4.9 minutes with the five-way
full-roll profile.

If one worker fails, the collector still discovers, hashes, splits, and audits
all completed trajectories and explicitly reports the partial batch as
incomplete. A computer shutdown no longer turns an otherwise useful run into
an all-or-nothing recovery exercise.

## What comes next

The immediate next task is AI-107: use the held-out validation and test
trajectories to measure negative log likelihood, interval coverage, z-scores,
and uncertainty decomposition. After that, AI-109 should run learning curves
at several corpus sizes and model capacities.

Those experiments will tell us whether the current local transition models are
learning real conditional structure, merely fitting average mark appearance,
or limited by data/model capacity. Composition learning and the “order versus
mess” idea remain downstream because we still need genuine terminal canvases
and a formally accepted higher-level preference.
