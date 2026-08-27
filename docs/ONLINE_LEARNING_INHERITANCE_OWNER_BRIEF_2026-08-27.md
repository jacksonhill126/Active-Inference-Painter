# What The Painter Is Allowed To Remember

Date: 2026-08-27

AI-112 is complete. The useful way to think about the decision is that the
painter now has three different kinds of memory instead of one giant
checkpoint-shaped bucket.

First, it has learned world-model knowledge: parameters that predict what
marks and passages do. Those can survive between paintings. Second, it has an
individual learning history: replay examples, optimizer momentum, and learned
calibration. Those may survive only when we explicitly resume the same
individual. Third, it has beliefs about the current episode: what is on this
canvas, what the brush holds, where the body is, and which passage is active.
Those are cleared for a new painting.

That distinction matters because shared simulator pretraining is like giving a
new painter a textbook, not giving it someone else's autobiography. A new
individual can inherit the shared model parameters, but it does not inherit the
other run's replay, confidence calibration, optimizer momentum, or canvas
history.

The audit found one real violation of this idea. The belief about how much the
composition gap had recently changed belonged to the current canvas, but the
old checkpoint saved it and reset did not clear it. That could make a new blank
painting begin with confidence imported from a previous canvas. The new
checkpoint schema removes that state and resets it to its declared prior.

There is also now a hard evidence label at every online replay entry. A
registered camera update, a physical sensor observation, or a clearly marked
oracle diagnostic may train the model. A counterfactual imagined by the model
may not. This prevents the system from becoming more confident merely by
retraining on its own guesses.

What this does not prove is that the painter learns indefinitely without
forgetting. We have defined the required check: compare likelihood on recent
experience with likelihood on a fixed held-out anchor set, and fail the
learning report if anchor performance degrades beyond a predeclared
uncertainty band. The three AI-114 replicas still need to instantiate and run
that check.

The detailed record is
`docs/ONLINE_LEARNING_INHERITANCE_2026-08-27.md`. This work changes memory
ownership and checkpoint semantics only; it does not add an aesthetic reward,
preference, or new policy influence.
