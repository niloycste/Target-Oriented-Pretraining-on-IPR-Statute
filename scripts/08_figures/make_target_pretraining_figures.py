from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "papers" / "target_pretraining" / "plots"
OUT.mkdir(parents=True, exist_ok=True)


def load_json(path: str) -> dict:
    with (ROOT / path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def check_close(actual: float, expected: float, tol: float = 0.01) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"{actual=} differs from {expected=}")


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def qa_by_code(qa: dict) -> dict:
    return {row["code"]: row for row in qa["results_summary"]}


def mlm_ppl_by_code(mlm: dict) -> dict:
    rows = {}
    for row in mlm["results"]:
        label = row["label"]
        if label.startswith("InLegalBERT baseline"):
            rows["base"] = row
        elif label.startswith("H' base"):
            rows["H'"] = row
        elif label.startswith("C' base"):
            rows["C'"] = row
        elif label.startswith("I' base"):
            rows["I'"] = row
    return rows


def make_pipeline_overview() -> None:
    """DAPT-safe version of the combined-paper IPR pipeline figure."""

    fig, ax = plt.subplots(figsize=(13.8, 7.0))
    ax.set_xlim(0, 13.8)
    ax.set_ylim(0, 7.0)
    ax.axis("off")

    panel_bg = ["#E0F2FE", "#FFEDD5", "#E0E7FF"]
    panel_edge = "#334155"
    box_edge = "#1E293B"
    dark = "#0F172A"
    blue = "#075985"
    orange = "#C2410C"
    indigo = "#3730A3"
    green = "#047857"
    yellow = "#A16207"

    panels = [
        (0.15, 0.25, 4.1, 6.5, "Benchmark and source text", panel_bg[0]),
        (4.85, 0.25, 4.1, 6.5, "Seed-guided DAPT corpora", panel_bg[1]),
        (9.55, 0.25, 4.1, 6.5, "Transfer, fine-tuning\nand evaluation", panel_bg[2]),
    ]
    for x, y, w, h, title, color in panels:
        ax.add_patch(
            plt.Rectangle((x, y), w, h, facecolor=color, edgecolor=panel_edge, linewidth=1.5)
        )
        ax.add_patch(
            plt.Rectangle((x, y + h - 0.48), w, 0.48, facecolor=panel_edge, edgecolor=panel_edge)
        )
        ax.text(x + w / 2, y + h - 0.24, title, ha="center", va="center", color="white", weight="bold", fontsize=10.5)

    def box(x, y, w, h, text, fc, color="white", fs=9.0, lw=1.2):
        ax.add_patch(
            plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=box_edge, linewidth=lw)
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=color, fontsize=fs, wrap=True)

    def rounded_box(x, y, w, h, text, fc, color=dark, fs=8.8):
        patch = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=box_edge, linewidth=1.2)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=color, fontsize=fs, wrap=True)

    def arrow(x1, y1, x2, y2, color="#475569"):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", lw=1.4, color=color, shrinkA=2, shrinkB=2),
        )

    box(0.55, 5.55, 3.3, 0.55, "Indian Code website: 7 IPR Acts", blue)
    box(0.9, 4.65, 2.6, 0.70, "PDF extraction and\nstructure-aware chunking", blue, fs=8.2)
    rounded_box(0.85, 3.45, 2.7, 0.85, "4,462 QA pairs\n1,116 curated contexts", "#FFFFFF")
    box(0.9, 2.25, 2.6, 0.70, "Section-level split", blue)
    rounded_box(0.55, 1.05, 3.3, 0.75, "Train / validation / test\n3,710 / 408 / 344", "#FFFFFF")
    arrow(2.2, 5.55, 2.2, 5.35)
    arrow(2.2, 4.65, 2.2, 4.30)
    arrow(2.2, 3.45, 2.2, 2.95)
    arrow(2.2, 2.25, 2.2, 1.80)

    box(5.15, 5.55, 3.5, 0.55, "Train-split legal seed extraction", orange)
    rounded_box(5.35, 4.62, 3.1, 0.70, "Rules + KeyBERT\nstatutory terms, entities, Act names", "#FFFFFF")
    box(5.35, 3.65, 3.1, 0.63, "Gemma 3 12B/27B\nstatutory-style contexts", orange, fs=8.2)
    rounded_box(5.15, 2.55, 3.5, 0.78, "Archived DAPT corpora\nH': 3,710 authentic\nC': 14,214 mixed\nI': 13,211 synthetic", "#FFFFFF", fs=8.3)
    rounded_box(5.15, 1.10, 3.5, 1.00, "Record-matched comparison\n901 records per corpus\nsame training recipe\nnot token-matched", "#FEF3C7", fs=8.0)
    arrow(6.9, 5.55, 6.9, 5.32)
    arrow(6.9, 4.62, 6.9, 4.28)
    arrow(6.9, 3.65, 6.9, 3.33)
    arrow(6.9, 2.55, 6.9, 2.10)

    box(9.95, 5.55, 3.3, 0.55, "Encoder initialization", indigo)
    rounded_box(10.2, 4.78, 2.8, 0.52, "InLegalBERT for\nDAPT routes", "#FFFFFF", fs=8.5)
    box(10.2, 3.90, 2.8, 0.55, "MLM DAPT: 15% masking", indigo)
    box(10.2, 3.00, 2.8, 0.55, "SQuAD v1 transfer", indigo)
    box(10.2, 2.10, 2.8, 0.55, "IndicIPR-QA fine-tuning", indigo)
    rounded_box(9.95, 0.90, 3.3, 0.78, "Gold-context evaluation\nEM/F1, seed variance,\nMLM perplexity", "#FFFFFF", fs=8.3)
    arrow(11.6, 5.55, 11.6, 5.30)
    arrow(11.6, 4.78, 11.6, 4.45)
    arrow(11.6, 3.90, 11.6, 3.55)
    arrow(11.6, 3.00, 11.6, 2.65)
    arrow(11.6, 2.10, 11.6, 1.68)
    arrow(4.25, 1.42, 5.15, 1.64)
    arrow(8.65, 1.64, 9.95, 1.28)

    save(fig, "01_ipr_dapt_pipeline")


def make_reader_inventory(qa: dict) -> None:
    by_code = qa_by_code(qa)
    expected = {
        "A": 85.0249,
        "B'": 86.0216,
        "E2'": 87.3721,
        "H'": 85.7314,
        "C'": 87.0367,
        "I'": 85.9577,
    }
    for code, val in expected.items():
        check_close(by_code[code]["f1"], val)

    order = ["A", "B'", "H'", "I'", "C'", "E2'"]
    labels = {
        "A": "A  InLegalBERT -> IPR",
        "D": "D  BERT-base -> IPR",
        "E": "E  LegalBERT-US -> IPR",
        "G": "G  InCaseLawBERT -> IPR",
        "B'": "B'  InLegalBERT -> SQuAD -> IPR",
        "D2'": "D2'  BERT-base -> SQuAD -> IPR",
        "E2'": "E2'  LegalBERT-US -> SQuAD -> IPR",
        "F'": "F'  InCaseLawBERT -> SQuAD -> IPR",
        "H'": "H'  Authentic DAPT -> SQuAD -> IPR",
        "C'": "C'  Mixed DAPT -> SQuAD -> IPR",
        "I'": "I'  Synthetic DAPT -> SQuAD -> IPR",
    }
    colors = []
    for code in order:
        if code in {"H'", "C'", "I'"}:
            colors.append("#D97706")
        elif code in {"B'", "D2'", "E2'", "F'"}:
            colors.append("#2563EB")
        else:
            colors.append("#64748B")
    vals = [by_code[code]["f1"] for code in order]

    fig, ax = plt.subplots(figsize=(9.7, 6.4))
    y = np.arange(len(order))
    ax.barh(y, vals, color=colors, alpha=0.90)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[c] for c in order])
    ax.set_xlabel("Test-set token F1")
    ax.set_xlim(84.6, 87.8)
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("Archived DAPT routes and relevant comparators")
    for yi, val, code in zip(y, vals, order):
        weight = "bold" if code in {"E2'", "C'"} else "normal"
        ax.text(val + 0.05, yi, f"{val:.2f}", va="center", fontsize=9, weight=weight)
    ax.text(
        0.02,
        0.02,
        "Orange = DAPT route; blue = SQuAD transfer without DAPT; grey = direct fine-tuning.",
        transform=ax.transAxes,
        fontsize=8.6,
        color="#475569",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#CBD5E1"),
    )
    save(fig, "02_archived_reader_inventory")


def latest_trainer_state(root: Path) -> Path | None:
    states = list(root.glob("**/trainer_state.json"))
    if not states:
        return None

    def score(path: Path) -> tuple[int, str]:
        m = re.search(r"checkpoint-(\d+)", str(path))
        return (int(m.group(1)) if m else -1, str(path))

    return max(states, key=score)


def make_mlm_loss_curve() -> None:
    variants = [
        ("H'", ROOT / "data" / "dapt" / "pretrained_model_original_only", "H': authentic DAPT", "#0EA5E9"),
        ("C'", ROOT / "data" / "dapt" / "pretrained_model_v2", "C': mixed DAPT", "#2563EB"),
        ("I'", ROOT / "data" / "dapt" / "pretrained_model_synth_only", "I': synthetic DAPT", "#F97316"),
    ]
    curves = []
    for code, root_dir, label, color in variants:
        state_path = latest_trainer_state(root_dir)
        if state_path is None:
            continue
        with state_path.open("r", encoding="utf-8") as f:
            state = json.load(f)
        entries = [entry for entry in state.get("log_history", []) if "loss" in entry and "step" in entry]
        if entries:
            curves.append((code, [e["step"] for e in entries], [e["loss"] for e in entries], label, color))

    if len(curves) < 3:
        raise FileNotFoundError("Expected trainer_state.json for H', C', and I' DAPT routes")

    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    offsets = {"H'": (-8, 12), "C'": (-10, 10), "I'": (-10, -14)}
    for code, steps, losses, label, color in curves:
        ax.plot(steps, losses, marker="o", linewidth=2.0, markersize=4.5, label=label, color=color)
        ax.annotate(
            f"{losses[-1]:.2f}",
            xy=(steps[-1], losses[-1]),
            xytext=offsets[code],
            textcoords="offset points",
            color=color,
            weight="bold",
            fontsize=9,
            ha="right",
            va="center",
            bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.92),
        )
    ax.set_xlabel("Training step")
    ax.set_ylabel("MLM training loss")
    ax.set_title("Archived DAPT optimization traces")
    ax.grid(True, linestyle="--", alpha=0.32)
    ax.legend(loc="upper right", frameon=True)
    ax.margins(x=0.05, y=0.10)
    save(fig, "03_archived_mlm_loss")


def make_perplexity_bars(mlm: dict) -> None:
    rows = mlm_ppl_by_code(mlm)
    check_close(rows["base"]["perplexity"], 11.1985, tol=0.02)
    check_close(rows["H'"]["perplexity"], 2.4921, tol=0.02)
    check_close(rows["C'"]["perplexity"], 2.5697, tol=0.02)
    check_close(rows["I'"]["perplexity"], 2.7936, tol=0.02)

    labels = ["InLegalBERT\nbaseline", "H'\nAuthentic", "C'\nMixed", "I'\nSynthetic"]
    vals = [rows["base"]["perplexity"], rows["H'"]["perplexity"], rows["C'"]["perplexity"], rows["I'"]["perplexity"]]
    colors = ["#64748B", "#0EA5E9", "#2563EB", "#F97316"]

    fig, ax = plt.subplots(figsize=(8.7, 4.9))
    x = np.arange(len(vals))
    ax.bar(x, vals, color=colors, alpha=0.90)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Perplexity on 113 test-split contexts")
    ax.set_title("DAPT lowers test-split MLM perplexity")
    ax.set_ylim(0, 12.5)
    ax.grid(axis="y", alpha=0.25)
    for i, val in enumerate(vals):
        ax.text(i, val + 0.18, f"{val:.2f}", ha="center", fontsize=9.5, weight="bold" if i == 0 else "normal")
    ax.text(
        0.98,
        0.94,
        "Lower is better",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#475569",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#CBD5E1"),
    )
    save(fig, "04_archived_mlm_perplexity")


def make_multiseed_figure(stats: dict) -> None:
    f1 = stats["f1"]
    labels = ["auth901", "synth901", "mixed901"]
    pretty = ["Authentic", "Synthetic", "Mixed"]
    colors = ["#0EA5E9", "#F97316", "#2563EB"]
    x = np.arange(len(labels))

    check_close(f1["means"]["auth901"], 86.029)
    check_close(f1["means"]["synth901"], 85.731)
    check_close(f1["means"]["mixed901"], 85.313)
    check_close(f1["spread_of_means"], 0.716)
    check_close(f1["pooled_sd"], 0.706)
    check_close(f1["one_way_anova"]["p"], 0.5007)

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    for i, key in enumerate(labels):
        vals = np.array(f1["per_seed"][key])
        jitter = np.array([-0.08, 0.0, 0.08])
        ax.scatter(
            np.full_like(vals, x[i], dtype=float) + jitter,
            vals,
            s=75,
            color=colors[i],
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
            label=pretty[i],
        )
        mean = f1["means"][key]
        sd = f1["sds"][key]
        ax.errorbar(
            x[i],
            mean,
            yerr=sd,
            fmt="o",
            color="#0F172A",
            ecolor="#0F172A",
            capsize=6,
            markersize=6,
            zorder=4,
        )
        ax.text(x[i], mean + sd + 0.12, f"{mean:.2f} +/- {sd:.2f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(pretty)
    ax.set_ylabel("Test-set token F1")
    ax.set_ylim(84.2, 87.0)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Record-matched DAPT: mean differences track seed variation")
    ax.text(
        0.5,
        0.04,
        "Mean spread = 0.72 F1; pooled within-condition SD = 0.71; seed-blocked ANOVA p = 0.63",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="#CBD5E1"),
    )
    save(fig, "05_size_matched_multiseed_f1")


def make_mlm_vs_qa_figure(mlm: dict, qa: dict) -> None:
    by_code = qa_by_code(qa)
    rows = mlm_ppl_by_code(mlm)
    labels = ["H'\nAuthentic", "C'\nMixed", "I'\nSynthetic"]
    ppl = [rows["H'"]["perplexity"], rows["C'"]["perplexity"], rows["I'"]["perplexity"]]
    f1 = [by_code["H'"]["f1"], by_code["C'"]["f1"], by_code["I'"]["f1"]]
    colors = ["#0EA5E9", "#2563EB", "#F97316"]

    check_close(by_code["H'"]["f1"], 85.7314)
    check_close(by_code["C'"]["f1"], 87.0367)
    check_close(by_code["I'"]["f1"], 85.9577)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=True)
    axes[0].bar(labels, ppl, color=colors, alpha=0.88)
    axes[0].set_title("MLM fit")
    axes[0].set_ylabel("Perplexity (lower is better)")
    axes[0].set_ylim(0, 3.2)
    axes[0].grid(axis="y", alpha=0.22)
    for i, v in enumerate(ppl):
        axes[0].text(i, v + 0.07, f"{v:.2f}", ha="center", fontsize=9)

    axes[1].bar(labels, f1, color=colors, alpha=0.88)
    axes[1].set_title("Downstream extractive QA")
    axes[1].set_ylabel("Test-set token F1")
    axes[1].set_ylim(84.8, 87.4)
    axes[1].grid(axis="y", alpha=0.22)
    for i, v in enumerate(f1):
        axes[1].text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)

    fig.suptitle("MLM perplexity and downstream F1 do not induce the same ranking", y=1.04)
    save(fig, "06_mlm_fit_vs_qa")


def main() -> None:
    setup_style()
    qa = load_json("results/evaluation/qa_eval_report.json")
    mlm = load_json("results/evaluation/mlm_eval_report.json")
    stats = load_json("results/experiments/size_matched_dapt/size_matched_multiseed_stats.json")

    make_pipeline_overview()
    make_reader_inventory(qa)
    make_mlm_loss_curve()
    make_perplexity_bars(mlm)
    make_multiseed_figure(stats)
    make_mlm_vs_qa_figure(mlm, qa)
    print(f"Wrote DAPT paper figures to {OUT}")


if __name__ == "__main__":
    main()
