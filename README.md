# GTA VI trophy rarity forecaster

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its
earn rate on Exophase at 2026-11-26T00:00:00Z as five quantiles
(5%, 25%, 50%, 75%, 95%) — `main.py`, Python 3.13, standard library only.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` case in the input dir and writes one `/output/<id>.out`
per case: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`.

## Model

A cohort-mixture forecaster on the logit scale:

- **Holders only grow.** Per trophy, `H(t) = rate(t) x players_tracked(t)`
  is fitted (log-linear on time, last 24 h of reads) and projected to the
  target for the old cohort.
- **Newcomers dilute.** Players tracked is projected the same way, with a
  refresh floor: launch night Exophase went 638 -> 3,868 players in one
  refresh, so the model assumes refreshes keep arriving (~every 6 h, ~2.5x
  each, capped at 8x) even when the history so far is flat. The median is
  the mixture of the old cohort's projected rate and the newcomers' rate.
- **Newcomers earn over time.** Their share of a trophy is a tier-based
  fraction of the old cohort's rate (platinum ~4%, bronze ~50-70%),
  scaled by `1 - exp(-horizon/24h)`: over 90 minutes they earn almost
  nothing; over 4.5 days nearly the full fraction.
- **Spread** is a random walk (`0.5 * sqrt(days)`) plus the fit residual
  plus a refresh shock: a refresh before the target is a discrete logit
  shock, largest for trophies newcomers have not earned. Wide on short
  horizons, decaying to a floor on long ones.

## Validation (local, practice cases p00/p01)

- Both cases valid (55 rows, non-decreasing quantiles in [0, 100]),
  byte-identical across two runs, ~0.08 s per case, 2,797 bytes.
- Scored against two synthetic practice answers built from the documented
  launch-night refresh (638 -> 3,868 players): worst case (newcomers hold
  nothing) loss **0.84** vs house baseline **1.69**; mild case (newcomers
  hold 25%) loss **0.43** vs baseline **0.88**. 90% coverage 1.00 in both.
- A synthetic 104-read final-like case (48 h, players 638 -> 22,000)
  runs valid in 0.08 s with sensible medians (platinum diluted low,
  early trophies rising).

The exact practice answers are server-side only; the synthetic answers
above are the documented refresh applied to the last read.
