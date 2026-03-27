# Observed End-to-End Latency Definition

This repository should not treat `X-Plane rx -> ACK rx` as the primary
end-to-end latency metric when discussing HILS fidelity.

That ACK-based metric is still useful, but it is only a digital-path proxy:

- host receive of X-Plane sample
- controller scheduling / command issue
- serial transfer
- MCU processing
- ACK return to host

It does **not** directly measure when the physical motion becomes visible in the
PX4-observed response.

## Definition Used Here

For the current Stewart-platform HILS workflow, the preferred end-to-end metric
is:

> the observed response delay from the host-received X-Plane attitude used by
> control to the corresponding bias-corrected PX4 attitude response

For pitch, this is estimated as the lag `tau` that best aligns the two
trajectories over the trim-transition scenario:

`tau_e2e = argmax_tau corr(theta_xp(t), theta_px4_corr(t + tau))`

where:

- `theta_xp` is the host-received X-Plane pitch trajectory
- `theta_px4_corr` is the PX4 pitch trajectory after mounting-bias correction
- `tau >= 0`

Equivalent phrasing:

- observed end-to-end delay
- shape-matched end-to-end delay
- effective response delay

Avoid describing this quantity as pure "communication latency".

## Why Shape Matching Is Needed

Pointwise timestamp matching is not reliable here because:

- PX4 and X-Plane samples are asynchronous
- PX4 attitude is noisy
- platform mechanics and PX4 estimation smooth the waveform
- the relevant scientific question is response alignment, not packet pairing

Because of that, the estimator should compare the overall waveform shape rather
than attempt bit-by-bit or sample-by-sample matching.

## Why The Trim-Transition Scenario Is Used

The trim-transition scenario is a good identification window because it contains
a structured `0 -> step -> 0` pitch shape.

That shape is much more informative for delay estimation than:

- steady holds, where delay is weakly observable
- arbitrary noisy motion, where alignment can be ambiguous

In practice, the useful phase set is:

- `hold_0`
- `warmup_step`
- `hold_step`
- `warmup_return`
- `hold_return`

## Practical Estimation Procedure

1. Read the host-timestamped `xplane_att.csv` and `px4_att.csv` streams.
2. Identify each trim-transition repeat from `tick.csv`.
3. Bias-correct PX4 pitch using the mounting-bias estimate.
4. Resample X-Plane and PX4 pitch onto a common time grid.
5. Sweep a positive lag window, e.g. `0-150 ms`.
6. For each candidate lag, compute normalized correlation.
7. Select the lag with the highest correlation.
8. Use the median repeat-level lag as the representative delay for that run.
9. Aggregate those run-level delays across runs `#5-#9`.

## Reporting Recommendation

For clarity, keep these latency quantities separate:

- `X-Plane sample age`
- `PX4 sample age`
- `Serial RTT`
- `Observed end-to-end delay` (shape-matched X-Plane -> PX4 response)

This makes the paper/presentation scientifically cleaner:

- the first three are transport / scheduling style metrics
- the last one is a system-level fidelity metric
