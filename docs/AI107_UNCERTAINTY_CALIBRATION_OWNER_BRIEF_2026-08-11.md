# What AI-107 Found

The short version is: training is clearly doing something useful, but the
models are not yet honest about their uncertainty.

Both the ordinary CNN and the conditional VAE assign much better probability
to held-out transitions after training. That is good. But when we ask whether
their error bars mean what they claim, both fail.

## What “calibrated” means here

Suppose a model gives a 90% prediction interval. Across many genuinely
held-out consequences, about 90% should land inside it. If 99.4% land inside,
the interval is too broad for most cases. It may sound safer, but it is not an
accurate probability statement.

That is what happened here:

| Model | nominal 50% interval actually contained | nominal 90% interval actually contained |
| --- | ---: | ---: |
| CNN | 97.9% | 99.4% |
| conditional VAE | 93.9% | 99.4% |

The residuals appear to have two regimes: lots of very small or nearly zero
changes, plus a much smaller set of larger errors. One Gaussian uncertainty
shape tries to cover both. It becomes far too wide around ordinary cells yet
still relies on rare errors to set its total variance.

I checked the cells under the actual brush footprint separately, so this is
not merely blank background overwhelming the statistics. The same pattern
remains there.

## Did the VAE help?

Not yet.

The VAE can, in principle, represent several materially different outcomes
for the same state and action through its latent variable. In this run, its
within-model latent variance became extremely small. Most of its uncertainty
came from a broad decoder variance and disagreement between separately trained
ensemble members.

Its held-out probability was slightly worse than the simpler CNN, and its
interval calibration failed in the same way. This does not prove that a VAE is
the wrong idea. It says this VAE, trained on 160 transitions with this
objective, has not yet earned promotion into the active runtime.

## Did ensemble disagreement detect unfamiliar situations?

Not reliably.

I trained a separate CNN only on the current neutral and fixed-roll motor
conditions. When it saw held-out dynamic roll sweeps, ensemble disagreement
rose by only 1.09 times. We had declared that at least 1.5 times would count as
useful preliminary OOD sensitivity.

The CNN did react strongly to an impossible normalized paint amount of 1.5,
but that is only a stress test. Detecting an invalid number is not the same as
recognizing a physically meaningful unfamiliar motor realization.

## A subtle terminology correction

The checkpoint contains eight precision entries, but none received any
precision-update observations during offline training. They are therefore
still declared priors, not inferred posteriors. They also are not the model's
prediction noise: they are inverse-temperature multipliers used when combining
EFE terms.

Likewise, the assumed camera noise is fixed. The report keeps all of these
separate from learned transition variance so a favorable-looking “uncertainty”
number cannot be assembled by mixing unrelated quantities.

## What we could not test

We still cannot honestly calibrate wet-over-wet behavior. The current camera
likelihood does not identify bulk wetness, and using exact simulator wetness
would violate the sensor-only boundary we deliberately set.

We also have only one effective patch-size class because this fast provisional
camera path reduces inference to a 16x16 posterior. And the test set has only
three independent painting trajectories. The report is useful evidence, but
not a final statistical characterization.

## What I think we should do next

AI-109 is now the right next task. It should train several model sizes on
several amounts of data and with several random seeds. That will tell us
whether these failures improve with more evidence or whether the likelihood
shape itself is wrong.

My leading hypothesis is that the next model should explicitly represent two
questions:

1. did a meaningful local material change occur?;
2. if it did, how large and what kind of change was it?

That is similar to a hurdle or mixture likelihood: a near-no-change component
plus a continuous material-consequence component. It is still active-inference
compatible because it is a more faithful generative likelihood, not a reward
or aesthetic score. We should test it against the current CNN and VAE during
AI-109 rather than assume it is correct.

So the honest project status is: AI-107 is finished, model learning is real,
calibration is not yet accepted, and we now have a concrete diagnosis and a
reproducible evaluator for the next modeling round.
