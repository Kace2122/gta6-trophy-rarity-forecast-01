#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its
earn rate on Exophase at the case target as five quantiles (5/25/50/75/95%).

Model: a cohort-mixture forecaster on the logit scale.
  * Holders H(t) = rate(t) * players_tracked(t) only ever grow, so per trophy
    we fit log H on time (last 24 h of reads) and project the old cohort's
    holders to the target; players_tracked is projected the same way.
  * New players arriving before the target dilute every rate toward the
    share of newcomers who will have earned the trophy by then. That share
    is a tier-based fraction of the old cohort's rate (platinum ~4%,
    bronze ~50-70%): newcomers play too, but they start from zero.
  * The median is the mixture of the old cohort's projected rate and the
    newcomers' rate, weighted by the projected player counts.
  * The spread is a random walk (house: 0.5 * sqrt(days)) plus the fit's
    residual plus a refresh shock: Exophase refreshes its counts on its own
    schedule (launch night: 638 -> 3,868 players in one refresh, every rate
    but the 100% one falling), so a refresh before the target is a discrete
    logit shock, largest for trophies newcomers have not earned yet. The
    shock term is wide on short horizons and decays to a floor on long ones,
    because the timing of the last refresh before the answer never stops
    mattering.

Standard library only. Deterministic: no randomness, no wall-clock use,
no network. Reads every *.in file in $MAKEFUN_INPUT_DIR (else /input) and
writes one /output/<id>.out per case ($MAKEFUN_OUTPUT_DIR else /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

# Standard normal quantiles at the five levels.
ZS = (-1.645, -0.674, 0.0, 0.674, 1.645)

# Release-day origin for the time axis (2026-11-19T00:00:00Z).
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Refresh prior: launch night's first refresh came after ~4.5 h of flat
# counts and multiplied players tracked by ~6. We assume refreshes arrive
# about every 6 h and multiply the base by 2.5x (conservative on both),
# capped at 8x total, as a floor under the fitted player growth.
REFRESH_GAP_H = 6.0
REFRESH_MULT = 2.5
GROWTH_FLOOR_CAP = 8.0

# Newcomer reach by tier: the fraction of the old cohort's rate that
# players first tracked before the target will have reached by then.
TIER_PHI = {"platinum": 0.05, "gold": 0.25, "silver": 0.45, "bronze": 0.60}

# Fallback last rates when a trophy was never seen in any read.
TIER_PRIOR = {"platinum": 0.5, "gold": 5.0, "silver": 15.0, "bronze": 30.0}


def parse_time(s):
    """ISO 8601 UTC timestamp -> aware datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(dt):
    return (dt - RELEASE).total_seconds() / 3600.0


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit(p):
    """p in percent -> logit, clamped to the scorer's [0.01, 99.99]."""
    p = clamp(p, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(l):
    return 100.0 / (1.0 + math.exp(-l))


def ols_slope(xs, ys):
    """Least-squares slope and residual SD of y on x; (0.0, 0.0) if degenerate."""
    n = len(xs)
    if n < 2:
        return 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    resid = [y - (my + slope * (x - mx)) for x, y in zip(xs, ys)]
    var = sum(r * r for r in resid) / max(n - 2, 1)
    return slope, math.sqrt(var)


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history", [])
    horizon_h = case["horizon_minutes"] / 60.0
    horizon_d = case["horizon_minutes"] / 1440.0
    at = parse_time(case["at"])

    # --- Fit window: the last 24 h of reads (all of them if fewer). ---
    window = [r for r in history
              if (at - parse_time(r["at"])).total_seconds() <= 24 * 3600]
    if len(window) < 3:
        window = list(history)

    # --- Project players tracked. ---
    pts = [(hours_since_release(parse_time(r["at"])), r["players_tracked"])
           for r in window if r.get("players_tracked")]
    if pts:
        xs = [p[0] for p in pts]
        ys = [math.log(max(p[1], 1)) for p in pts]
        b, _ = ols_slope(xs, ys)
        b = clamp(b, 0.0, 0.10)
        p_last = pts[-1][1]
    else:
        b, p_last = 0.0, 1
    g_fitted = math.exp(b * horizon_h)
    g_floor = min(GROWTH_FLOOR_CAP,
                  math.exp(math.log(REFRESH_MULT) * horizon_h / REFRESH_GAP_H))
    growth = clamp(max(g_fitted, g_floor), 1.0, 50.0)
    p_target = p_last * growth

    rows = []
    for idx, trophy in enumerate(trophies):
        tier = trophy.get("tier", "bronze")
        # Last observed rate for this trophy (None if never seen).
        r_last = None
        for read in reversed(history):
            rates = read.get("rates") or []
            if idx < len(rates) and rates[idx] is not None:
                r_last = float(rates[idx])
                break
        if r_last is None:
            r_last = TIER_PRIOR.get(tier, 30.0)

        # A trophy every tracked player holds stays held: 100 across.
        if r_last >= 100.0:
            rows.append({"id": trophy["id"], "q": [100.0] * 5})
            continue

        # --- Project the old cohort's holders. ---
        hs = []
        for read in window:
            rates = read.get("rates") or []
            if idx < len(rates) and rates[idx] is not None:
                t = hours_since_release(parse_time(read["at"]))
                h = float(rates[idx]) / 100.0 * read["players_tracked"]
                hs.append((t, h))
        if len(hs) >= 2:
            xs = [p[0] for p in hs]
            ys = [math.log(p[1] + 0.5) for p in hs]
            a, resid_sd = ols_slope(xs, ys)
            a = clamp(a, 0.0, 0.20)
            fit_sd = clamp(resid_sd, 0.0, 0.8)
        else:
            a, fit_sd = 0.0, 0.25
        h_last = (r_last / 100.0) * p_last
        h_old = min(h_last * math.exp(a * horizon_h), p_last * 0.999)
        r_old = clamp(100.0 * h_old / p_last, 0.01, 99.99)

        # --- Newcomers' share, and the mixture median. ---
        # Newcomers earn over the time they have had: over a 90-minute
        # horizon almost nothing, over 4.5 days nearly the tier fraction.
        phi = TIER_PHI.get(tier, 0.60)
        phi = clamp(phi * (0.7 + 0.6 * math.sqrt(r_last / 100.0)), 0.03, 0.85)
        phi *= 1.0 - math.exp(-horizon_h / 24.0)
        q_new = phi * r_old
        r_med = (r_old + (growth - 1.0) * q_new) / growth
        r_med = clamp(r_med, 0.01, 99.99)

        # --- Spread: random walk + fit residual + refresh shock. ---
        exposure = clamp(1.0 - q_new / max(r_old, 1e-9), 0.0, 1.0)
        shock = exposure * (1.15 * math.exp(-horizon_h / 48.0) + 0.35)
        sigma = math.sqrt((0.5 * math.sqrt(horizon_d)) ** 2
                          + fit_sd ** 2 + shock ** 2)
        base = logit(r_med)
        qs = []
        for z in ZS:
            qs.append(round(clamp(inv_logit(base + z * sigma), 0.01, 99.99), 4))
        # Rounding must not break the non-decreasing order.
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        rows.append({"id": trophy["id"], "q": qs})

    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    if not names:
        sys.stderr.write("no .in files in %s\n" % in_dir)
        return 1
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        out = forecast_case(case)
        out_path = os.path.join(out_dir, name[:-3] + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, separators=(",", ":"), sort_keys=False)
            f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
