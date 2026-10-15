"""Inferential statistics for the size-matched multi-seed ablation.

size_matched_multiseed_results.json holds nine end-to-end runs: three corpora
(auth901, synth901, mixed901) x three seeds (7, 42, 99). The paper reports the
spread between corpus means against the pooled within-corpus SD
and concludes that no corpus-composition effect is detectable above seed noise. This
script supplies the test behind that claim and, more usefully, the interval
that bounds how large an effect the design could still have missed.

Two models are fitted:

  * one-way ANOVA, treating the nine runs as independent. This ignores that the
    same three seeds were used for every corpus.
  * randomised-block (repeated-measures) ANOVA with seed as the blocking
    factor. Seed 7 means the same initialisation and data order for every
    corpus, so seed is a genuine block; removing that variance is the more
    appropriate and more powerful test.

Pairwise differences are reported as paired-t confidence intervals over the
three shared seeds, which is what bounds the effect size.

Run:  python size_matched_multiseed_stats.py
"""
from __future__ import annotations

import json
import math
import statistics as st
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # repo root
RESULTS_DIR = ROOT / "results" / "experiments" / "size_matched_dapt"
RESULTS = RESULTS_DIR / "size_matched_multiseed_results.json"
OUT_JSON = RESULTS_DIR / "size_matched_multiseed_stats.json"

CORPORA = ["auth901", "synth901", "mixed901"]
SEEDS = [7, 42, 99]
ALPHA = 0.05


def load(metric="f1"):
    res = json.loads(RESULTS.read_text())
    tbl = {}
    for c in CORPORA:
        row = []
        for s in SEEDS:
            key = f"{c}_s{s}"
            if key not in res:
                raise SystemExit(f"missing {key} in {RESULTS.name}")
            row.append(res[key][metric])
        tbl[c] = row
    return tbl


def one_way_anova(tbl):
    from scipy import stats
    groups = [tbl[c] for c in CORPORA]
    F, p = stats.f_oneway(*groups)
    grand = st.mean([v for g in groups for v in g])
    k, n = len(groups), len(groups[0])
    ss_between = n * sum((st.mean(g) - grand) ** 2 for g in groups)
    ss_within = sum((v - st.mean(g)) ** 2 for g in groups for v in g)
    return {"F": round(F, 4), "p": round(p, 4),
            "df": [k - 1, k * (n - 1)],
            "ss_between": round(ss_between, 4),
            "ss_within": round(ss_within, 4),
            "eta_sq": round(ss_between / (ss_between + ss_within), 4)}


def block_anova(tbl):
    """Randomised-block ANOVA: corpus effect with seed variance removed."""
    from scipy import stats
    k, n = len(CORPORA), len(SEEDS)
    vals = [tbl[c] for c in CORPORA]
    grand = st.mean([v for g in vals for v in g])
    corpus_means = [st.mean(g) for g in vals]
    seed_means = [st.mean([tbl[c][j] for c in CORPORA]) for j in range(n)]

    ss_corpus = n * sum((m - grand) ** 2 for m in corpus_means)
    ss_seed = k * sum((m - grand) ** 2 for m in seed_means)
    ss_total = sum((v - grand) ** 2 for g in vals for v in g)
    ss_err = ss_total - ss_corpus - ss_seed

    df_corpus, df_seed = k - 1, n - 1
    df_err = df_corpus * df_seed
    ms_corpus, ms_err = ss_corpus / df_corpus, ss_err / df_err
    F = ms_corpus / ms_err
    p = 1 - stats.f.cdf(F, df_corpus, df_err)

    F_seed = (ss_seed / df_seed) / ms_err
    p_seed = 1 - stats.f.cdf(F_seed, df_seed, df_err)

    return {"F_corpus": round(F, 4), "p_corpus": round(float(p), 4),
            "df_corpus": [df_corpus, df_err],
            "F_seed": round(F_seed, 4), "p_seed": round(float(p_seed), 4),
            "ss_corpus": round(ss_corpus, 4), "ss_seed": round(ss_seed, 4),
            "ss_error": round(ss_err, 4),
            "resid_sd": round(ms_err ** 0.5, 4)}


def paired_intervals(tbl):
    from scipy import stats
    out = {}
    n = len(SEEDS)
    tcrit = stats.t.ppf(1 - ALPHA / 2, n - 1)
    for a, b in combinations(CORPORA, 2):
        d = [x - y for x, y in zip(tbl[a], tbl[b])]
        m, sd = st.mean(d), st.stdev(d)
        se = sd / n ** 0.5
        t = m / se if se else float("nan")
        p = 2 * (1 - stats.t.cdf(abs(t), n - 1)) if se else float("nan")
        out[f"{a} - {b}"] = {
            "mean_diff": round(m, 3), "sd": round(sd, 3),
            "ci95": [round(m - tcrit * se, 3), round(m + tcrit * se, 3)],
            "t": round(t, 3), "p": round(float(p), 4),
            "per_seed": [round(v, 3) for v in d]}
    return out


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {}
    for metric in ("f1", "em"):
        tbl = load(metric)
        means = {c: round(st.mean(tbl[c]), 3) for c in CORPORA}
        sds = {c: round(st.stdev(tbl[c]), 3) for c in CORPORA}
        pooled = round(
            math.sqrt(
                sum((len(tbl[c]) - 1) * st.variance(tbl[c]) for c in CORPORA)
                / sum(len(tbl[c]) - 1 for c in CORPORA)
            ),
            3,
        )
        spread = round(max(means.values()) - min(means.values()), 3)

        print("=" * 70)
        print(f"metric: {metric.upper()}   (n=3 seeds per corpus, seeds {SEEDS})")
        print("=" * 70)
        for c in CORPORA:
            print(f"  {c:9s} {tbl[c]}  mean {means[c]:.3f}  SD {sds[c]:.3f}")
        print(f"  spread of means {spread:.3f}   pooled within-corpus SD {pooled:.3f}")

        ow = one_way_anova(tbl)
        bl = block_anova(tbl)
        pi = paired_intervals(tbl)

        print(f"\n  one-way ANOVA      F({ow['df'][0]},{ow['df'][1]}) = {ow['F']:.3f}"
              f"   p = {ow['p']:.4f}   eta^2 = {ow['eta_sq']:.4f}")
        print(f"  block ANOVA (seed) F({bl['df_corpus'][0]},{bl['df_corpus'][1]}) ="
              f" {bl['F_corpus']:.3f}   p = {bl['p_corpus']:.4f}"
              f"   residual SD = {bl['resid_sd']:.3f}")
        print(f"  seed (block) effect  F = {bl['F_seed']:.3f}   p = {bl['p_seed']:.4f}")

        print("\n  paired differences over the three shared seeds:")
        widest = 0.0
        for k, v in pi.items():
            lo, hi = v["ci95"]
            widest = max(widest, abs(lo), abs(hi))
            print(f"    {k:22s} {v['mean_diff']:+7.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]"
                  f"   t={v['t']:+.3f}  p={v['p']:.4f}")
        print(f"\n  largest magnitude inside any pairwise 95% CI: {widest:.2f}")
        print()

        report[metric] = {"per_seed": tbl, "means": means, "sds": sds,
                          "pooled_sd": pooled, "spread_of_means": spread,
                          "one_way_anova": ow, "block_anova": bl,
                          "paired": pi, "max_abs_ci_bound": round(widest, 3)}

    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"written to {OUT_JSON}")


if __name__ == "__main__":
    main()
