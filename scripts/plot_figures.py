"""Generate publication figures and tables for MOD-Bench paper.

Figures:
  fig4: Persona Depth effect (Finding 1)
  fig5: β-Diversity UCR (Finding 2, visual)
  fig7: Interaction effects (Finding 4)
Tables:
  tab_architecture: Architecture MPD comparison (Finding 2, precise)
  tab_tricks: Low-cost strategies effectiveness (Finding 3)
Appendix:
  fig_a1: Full factorial heatmap

Usage:
    python scripts/plot_figures.py
    python scripts/plot_figures.py --only fig4
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from tueplots import bundles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────

MODELS = ["DeepSeek-V3.2", "GPT-5.4-mini", "Gemini-3-Flash", "Kimi-K2.5",
          "Llama-4-Mav.", "Grok-4.1", "Qwen-3.6 Plus"]
MODELS_SHORT = ["DeepSeek", "GPT-mini", "Gemini", "Kimi", "Llama", "Grok", "Qwen"]
MODELS_NO_GPT = [0, 2, 3, 4, 5, 6]

MPD_INDEP = {
    "None":  [0.5979, 0.5774, 0.6290, 0.6230, 0.5353, 0.6257, 0.6065],
    "Role":  [0.6661, 0.6102, 0.7134, 0.7079, 0.5763, 0.6536, 0.6829],
    "Basic": [0.6847, 0.5799, 0.6943, 0.6826, 0.5945, 0.6511, 0.6761],
    "Mid":   [0.6757, 0.5887, 0.7013, 0.6862, 0.5974, 0.6577, 0.6811],
    "Pro":   [0.7022, 0.5967, 0.7224, 0.7052, 0.6156, 0.6649, 0.6952],
}

MPD_ABLATION_6 = {
    "None":       [0.5979, 0.6290, 0.6230, 0.5353, 0.6257, 0.6065],
    "Enhanced":   [0.6054, 0.6305, 0.6380, 0.5363, 0.6329, 0.6190],
    "TempHigh":   [0.6002, 0.6304, None,   0.5394, 0.6266, 0.6102],
    "Dimension":  [0.5949, 0.6300, 0.6266, 0.5366, 0.6252, 0.6172],
    "RandomSys":  [0.6185, 0.6675, 0.6631, 0.5423, 0.6357, 0.6411],
    "Role":       [0.6661, 0.7134, 0.7079, 0.5763, 0.6536, 0.6829],
}

MPD_SEQ = {
    "None": [0.6584, None, 0.6727, 0.6852, 0.5830, 0.6808, 0.6525],
    "Role": [0.7193, None, 0.7417, 0.7689, 0.6559, 0.7221, 0.7467],
    "Basic":[0.7251, None, 0.7270, 0.7364, 0.6394, 0.7129, 0.7144],
    "Mid":  [0.7183, None, 0.7326, 0.7405, 0.6455, 0.7197, 0.7225],
    "Pro":  [0.7429, 0.6670, 0.7464, 0.7519, 0.6682, 0.7248, 0.7474],
}

MPD_INT = {
    "None": [0.6144, None, 0.6497, 0.6865, 0.5586, 0.6673, 0.6727],
    "Role": [0.6763, None, 0.7328, 0.7580, 0.6161, 0.7171, 0.7540],
    "Basic":[0.6859, None, 0.7228, 0.7397, 0.5926, 0.6954, 0.7453],
    "Mid":  [0.6681, None, 0.7222, 0.7445, 0.6030, 0.6993, 0.7402],
    "Pro":  [0.7166, 0.6363, 0.7414, 0.7551, 0.6277, 0.7069, 0.7552],
}

MPD_5P = {
    "Indep": [0.6889, 0.5899, 0.7176, 0.7017, 0.5995, 0.6619, 0.6855],
    "Seq":   [0.7332, 0.6680, 0.7416, 0.7471, 0.6593, 0.7218, 0.7398],
    "Int":   [0.7166, 0.6363, 0.7414, 0.7551, 0.6277, 0.7069, 0.7552],
}

RAREFIED_CC = {
    "Independent": {
        "None": [59.7, 93.3, 67.9, 40.3, 111.5, 79.0],
        "Pro":  [95.8, 136.6, 91.1, 62.6, 132.6, 127.4],
    },
    "Sequential": {
        "None": [153.3, 148.3, 102.8, 60.3, 123.5, 129.1],
        "Pro":  [267.9, 271.5, 170.2, 119.7, 204.6, 264.8],
    },
    "Interactive": {
        "None": [75.0, 100.5, 110.9, 35.5, 87.6, 133.4],
        "Pro":  [93.2, 124.2, 104.4, 44.8, 92.0, 155.5],
    },
}

UCR_SEQ_VS_INT = {
    "DeepSeek-V3.2": (0.85, 0.84), "GPT-5.4-mini": (0.67, 0.58),
    "Gemini-3-Flash": (0.87, 0.87), "Kimi-K2.5": (0.89, 0.89),
    "Llama-4-Mav.": (0.67, 0.59), "Grok-4.1": (0.82, 0.80),
    "Qwen-3.6 Plus": (0.85, 0.86),
}

# ── Style ─────────────────────────────────────────────────────────────────────

OKABE_ITO = [
    "#0072B2", "#E69F00", "#009E73", "#D55E00",
    "#CC79A7", "#56B4E9", "#F0E442",
]

MODEL_COLORS = {
    "DeepSeek-V3.2":  OKABE_ITO[0],
    "GPT-5.4-mini":   OKABE_ITO[1],
    "Gemini-3-Flash":  OKABE_ITO[2],
    "Kimi-K2.5":      OKABE_ITO[3],
    "Llama-4-Mav.":   OKABE_ITO[4],
    "Grok-4.1":       OKABE_ITO[5],
    "Qwen-3.6 Plus":  OKABE_ITO[6],
}

MODEL_MARKERS = {
    "DeepSeek-V3.2": "o", "GPT-5.4-mini": "s", "Gemini-3-Flash": "^",
    "Kimi-K2.5": "D", "Llama-4-Mav.": "v", "Grok-4.1": "P",
    "Qwen-3.6 Plus": "X",
}

FW_COLORS = ["#64748B", "#0072B2", "#009E73"]
FW_LABELS = ["Single-Call", "Multi-Turn", "Multi-Agent"]
FW_KEYS = ["Independent", "Sequential", "Interactive"]


def setup_style():
    plt.style.use(["science", "no-latex", "grid"])
    plt.rcParams.update(bundles.neurips2024(usetex=False, family="serif"))
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.15,
        "xtick.major.width": 0.15,
        "ytick.major.width": 0.15,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "grid.alpha": 0.2,
        "grid.linewidth": 0.2,
        "savefig.dpi": 300,
    })


def _save(fig, name: str):
    for fmt in ["pdf", "png"]:
        fig.savefig(FIG_DIR / f"{name}.{fmt}")
    plt.close(fig)
    print(f"  {name} → {FIG_DIR}/{name}.pdf")


# ── Figure 4A: Persona Depth — 3 panels per architecture ─────────────────────

def plot_fig4a():
    depths = ["None", "Role", "Basic", "Mid", "Pro"]
    x = np.arange(len(depths))
    fw_data = [
        ("(a) Single-Call", MPD_INDEP),
        ("(b) Multi-Turn", MPD_SEQ),
        ("(c) Multi-Agent", MPD_INT),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 3.1), sharey=True,
                             layout="none")
    fig.subplots_adjust(wspace=0.08, bottom=0.28, top=0.91, left=0.09,
                        right=0.97)

    for ax, (fw_name, data) in zip(axes, fw_data):
        for i in MODELS_NO_GPT:
            model = MODELS[i]
            vals_raw = [data[d][i] for d in depths]
            if any(v is None for v in vals_raw):
                continue
            ax.plot(x, vals_raw, color=MODEL_COLORS[model],
                    marker="None", linewidth=1.0, zorder=3,
                    label=model if ax is axes[0] else None)

        ax.set_xticks(x)
        ax.set_xticklabels(depths, fontsize=6)
        ax.set_title(fw_name, fontsize=7.5, pad=4)
        ax.set_xlim(-0.3, 4.3)
        ax.tick_params(axis="y", labelsize=6)

    axes[0].set_ylabel("Mean Pairwise Distance", fontsize=7)
    axes[0].set_ylim(0.52, 0.78)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, ncol=3, loc="upper center",
               bbox_to_anchor=(0.53, 0.02), frameon=False, fontsize=5.5,
               columnspacing=1.2, handletextpad=0.5)

    _save(fig, "fig4a_persona_panels")


# ── Figure 4B: Persona Depth — mean ± SE across architectures ────────────────

def plot_fig4b():
    depths = ["None", "Role", "Basic", "Mid", "Pro"]
    x = np.arange(len(depths))
    fw_plot = [
        ("Single-Call", MPD_INDEP),
        ("Multi-Turn", MPD_SEQ),
        ("Multi-Agent", MPD_INT),
    ]

    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    for fw_name, data in fw_plot:
        means, ses = [], []
        for d in depths:
            vals = [v for v in data[d] if v is not None]
            means.append(np.mean(vals))
            ses.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        means, ses = np.array(means), np.array(ses)

        ax.plot(x, means, marker="o", markersize=3.5, linewidth=1.3,
                label=fw_name, zorder=4)
        ax.fill_between(x, means - ses, means + ses, alpha=0.15, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(depths)
    ax.set_xlabel("Persona Depth")
    ax.set_ylabel("Mean Pairwise Distance")
    ax.set_ylim(0.57, 0.76)
    ax.set_title("Persona Depth Effect by Architecture")

    ax.legend(frameon=True, framealpha=0.9, edgecolor="#ccc",
              fontsize=6, loc="lower right")

    _save(fig, "fig4b_persona_mean")


# ── Figure 5: β-Diversity UCR ───────────────────────────────────────────────

def plot_fig5(baseline_ucr: float = None):
    fig, ax = plt.subplots(figsize=(2.8, 2.2), layout="none")
    fig.subplots_adjust(bottom=0.30, top=0.90, left=0.17, right=0.96)

    models_ucr = list(UCR_SEQ_VS_INT.keys())
    xb = np.arange(len(models_ucr))
    ucr_avg = [np.mean(UCR_SEQ_VS_INT[m]) for m in models_ucr]

    bars = ax.bar(xb, [v * 100 for v in ucr_avg], width=0.35,
                  color="#5B7B9A", alpha=0.85, edgecolor="none",
                  zorder=3)

    for bar, val in zip(bars, ucr_avg):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                f"{val:.0%}", ha="center", va="bottom", fontsize=5,
                color="#333")

    overall = np.mean(ucr_avg) * 100
    ax.axhline(overall, color="#555", linestyle=":", linewidth=0.6, zorder=2)
    ax.text(len(models_ucr) - 1, overall - 3, f"mean: {overall:.0f}%",
            color="#555", fontsize=5, ha="right")

    if baseline_ucr is not None:
        ax.axhline(baseline_ucr * 100, color="#c0392b", linestyle="--",
                   linewidth=0.8, zorder=2)
        ax.text(0.3, baseline_ucr * 100 + 1.5,
                f"split-half baseline: {baseline_ucr:.0%}",
                color="#c0392b", fontsize=5, ha="left")

    ax.set_xticks(xb)
    ax.set_xticklabels(models_ucr, fontsize=5, rotation=35, ha="right")
    ax.set_ylabel("Unique Cluster Ratio (%)", fontsize=7)
    ax.set_ylim(0, 108)
    ax.set_title(r"$\beta$-Diversity: Sequential vs Interactive (UCR)",
                 fontsize=7.5)

    _save(fig, "fig5_beta_ucr")


# ── Figure 7: Interaction Effects ─────────────────────────────────────────────

def plot_fig7():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.5, 2.0), layout="none")
    fig.subplots_adjust(wspace=0.38, bottom=0.18, top=0.88, left=0.12, right=0.97)

    np.random.seed(42)

    mpd_gains = {}
    for fw, data in [("Independent", MPD_INDEP), ("Sequential", MPD_SEQ),
                     ("Interactive", MPD_INT)]:
        gains = []
        for i in MODELS_NO_GPT:
            n, p = data["None"][i], data["Pro"][i]
            if n is not None and p is not None:
                gains.append((p - n) / n * 100)
        mpd_gains[fw] = gains

    x = np.arange(3)
    mpd_m = [np.mean(mpd_gains[fw]) for fw in FW_KEYS]
    mpd_se = [np.std(mpd_gains[fw], ddof=1) / np.sqrt(len(mpd_gains[fw]))
              for fw in FW_KEYS]

    ax1.bar(x, mpd_m, width=0.55, color=FW_COLORS, edgecolor="none",
            zorder=3)
    ax1.errorbar(x, mpd_m, yerr=mpd_se, fmt="none", ecolor="#333",
                 capsize=2, linewidth=0.35, zorder=4)

    ax1.set_xticks(x)
    ax1.set_xticklabels(FW_LABELS, fontsize=6)
    ax1.set_ylabel("MPD Gain (%)", fontsize=7)
    ax1.set_title("(a) MPD Gain (Dispersion)", fontsize=7.5)
    for i, m in enumerate(mpd_m):
        ax1.text(i, m + mpd_se[i] + 0.5, f"+{m:.1f}%", ha="center",
                 fontsize=6)

    cc_gains = {}
    for fw in FW_KEYS:
        gains = []
        for i in range(6):
            n = RAREFIED_CC[fw]["None"][i]
            p = RAREFIED_CC[fw]["Pro"][i]
            gains.append((p - n) / n * 100)
        cc_gains[fw] = gains

    cc_m = [np.mean(cc_gains[fw]) for fw in FW_KEYS]
    cc_se = [np.std(cc_gains[fw], ddof=1) / np.sqrt(len(cc_gains[fw]))
             for fw in FW_KEYS]

    ax2.bar(x, cc_m, width=0.55, color=FW_COLORS, edgecolor="none",
            zorder=3)
    ax2.errorbar(x, cc_m, yerr=cc_se, fmt="none", ecolor="#333",
                 capsize=2, linewidth=0.35, zorder=4)

    ax2.set_xticks(x)
    ax2.set_xticklabels(FW_LABELS, fontsize=6)
    ax2.set_ylabel("Rarefied CC Gain (%)", fontsize=7)
    ax2.set_title("(b) Rarefied CC Gain (Richness)", fontsize=7.5)
    for i, m in enumerate(cc_m):
        ax2.text(i, m + cc_se[i] + 1.5, f"+{m:.0f}%", ha="center",
                 fontsize=6)

    _save(fig, "fig7_interaction")


# ── Appendix: A1 Heatmap ─────────────────────────────────────────────────────

def plot_fig_a1():
    depths = ["None", "Role", "Basic", "Mid", "Pro"]
    fw_data = [("Independent", MPD_INDEP), ("Sequential", MPD_SEQ),
               ("Interactive", MPD_INT)]

    grid = np.zeros((3, 5))
    for fi, (_, data) in enumerate(fw_data):
        for di, d in enumerate(depths):
            vals = [v for v in data[d] if v is not None]
            grid[fi, di] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(5.5, 1.8))
    im = ax.imshow(grid, cmap="YlOrRd", aspect="auto", vmin=0.58, vmax=0.76)

    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(depths)
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels([n for n, _ in fw_data])
    ax.set_xlabel("Persona Depth")

    for fi in range(3):
        for di in range(5):
            color = "white" if grid[fi, di] > 0.71 else "black"
            ax.text(di, fi, f"{grid[fi, di]:.3f}", ha="center", va="center",
                    fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("MPD (avg across models)", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    ax.set_title("Factorial Grid: MPD by Persona Depth × Architecture")
    _save(fig, "fig_a1_heatmap")


# ── Tables ────────────────────────────────────────────────────────────────────

def generate_tab_architecture():
    """LaTeX table: Architecture effect on MPD (5-persona matched, Pro)."""

    # C4 Indep→Seq @Pro: all 7 models significant
    # C5 Indep→Int @Pro: 6/7 significant (Llama not sig)
    c5_sig = [True, True, True, True, False, True, True]

    lines = [
        r"\begin{table}[t]",
        r"\caption{MPD by interaction architecture (Pro persona, 5-persona matched).",
        r"All Multi-Turn values significantly exceed Single-Call",
        r"($p_{\text{FDR}} < 0.05$, Wilcoxon signed-rank).",
        r"$\dagger$: Multi-Agent also significantly exceeds Single-Call.}",
        r"\label{tab:architecture}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & Single-Call & Multi-Turn & Multi-Agent \\",
        r"\midrule",
    ]

    for i, model in enumerate(MODELS):
        indep = MPD_5P["Indep"][i]
        seq = MPD_5P["Seq"][i]
        intr = MPD_5P["Int"][i]

        int_mark = r"$^\dagger$" if c5_sig[i] else ""
        lines.append(
            f"{model} & {indep:.3f} & {seq:.3f} & {intr:.3f}{int_mark} \\\\"
        )

    mean_i = np.mean(MPD_5P["Indep"])
    mean_s = np.mean(MPD_5P["Seq"])
    mean_t = np.mean(MPD_5P["Int"])

    lines.extend([
        r"\midrule",
        f"Mean & {mean_i:.3f} & {mean_s:.3f} & {mean_t:.3f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    tex = "\n".join(lines)
    out_path = FIG_DIR / "tab_architecture.tex"
    out_path.write_text(tex)
    print(f"  tab_architecture → {out_path}")


def generate_tab_tricks():
    """LaTeX table: Low-cost strategies vs None baseline."""

    none_vals = MPD_ABLATION_6["None"]

    # (display_name, data_key, r_rb_per_model, n_sig, n_total)
    # r_rb from summary Section 2 (B-tests for MPD, 6 models excl. GPT)
    tricks = [
        ("Role assignment", "Role",
         [-0.82, -1.00, -1.00, -0.76, -0.62, -0.90], 6, 6),
        ("Random personality", "RandomSys",
         [-0.44, -0.88, -0.84, -0.30, -0.32, -0.76], 6, 6),
        ("Enhanced prompt", "Enhanced",
         [-0.18, -0.08, -0.39, 0.09, -0.28, -0.41], 4, 6),
        ("Dimension priming", "Dimension",
         [0.18, -0.10, -0.15, 0.04, 0.00, -0.39], 1, 6),
        ("High temperature", "TempHigh",
         [-0.06, -0.12, -0.22, -0.18, -0.19], 2, 5),
    ]

    lines = [
        r"\begin{table}[t]",
        r"\caption{Low-cost strategies for increasing opinion diversity",
        r"(Independent framework, no structured persona).",
        r"$\Delta$MPD: mean improvement over None baseline across models.",
        r"$\tilde{r}_{rb}$: median matched-pairs rank-biserial.",
        r"Sig.: models with $p_{\text{FDR}} < 0.05$ out of total tested.}",
        r"\label{tab:tricks}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Strategy & Mean $\Delta$MPD & $\tilde{r}_{rb}$ & Sig. \\",
        r"\midrule",
    ]

    for name, key, r_rbs, n_sig, n_total in tricks:
        raw = MPD_ABLATION_6[key]
        deltas = [r - n for r, n in zip(raw, none_vals) if r is not None]
        mean_d = np.mean(deltas)
        median_r = float(np.median(r_rbs))

        lines.append(
            f"{name} & $+${mean_d:.4f} & ${median_r:.2f}$ & {n_sig}/{n_total} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    tex = "\n".join(lines)
    out_path = FIG_DIR / "tab_tricks.tex"
    out_path.write_text(tex)
    print(f"  tab_tricks → {out_path}")


# ── Figure: Low-cost Tricks ΔMPD ─────────────────────────────────────────────

def plot_fig_tricks():
    none_vals = np.array(MPD_ABLATION_6["None"])
    strategies = [
        ("Single-Call\n& Role", "Role", 6, 6),
        ("Random\npersonality", "RandomSys", 6, 6),
        ("Enhanced\nprompt", "Enhanced", 4, 6),
        ("High\ntemperature", "TempHigh", 2, 5),
        ("Dimension\npriming", "Dimension", 1, 6),
    ]

    means, ses, labels, sig_labels = [], [], [], []
    for label, key, n_sig, n_total in strategies:
        raw = np.array([v for v in MPD_ABLATION_6[key] if v is not None])
        base = np.array([n for n, v in zip(none_vals, MPD_ABLATION_6[key])
                         if v is not None])
        deltas = raw - base
        means.append(np.mean(deltas))
        ses.append(np.std(deltas, ddof=1) / np.sqrt(len(deltas)))
        labels.append(label)
        sig_labels.append(f"{n_sig}/{n_total}")

    x = np.arange(len(strategies))
    fig, ax = plt.subplots(figsize=(2.8, 1.9), layout="none")
    fig.subplots_adjust(bottom=0.24, top=0.92, left=0.20, right=0.96)

    colors = ["#5B7B9A", "#999", "#999", "#999", "#999"]
    bars = ax.bar(x, means, width=0.35, color=colors, alpha=0.85,
                  edgecolor="none", zorder=3)
    ax.errorbar(x, means, yerr=ses, fmt="none", ecolor="#333", capsize=3,
                linewidth=0.6, zorder=4)

    for i, (m, se, sig) in enumerate(zip(means, ses, sig_labels)):
        y_top = m + se + 0.001
        ax.text(i, y_top, f"+{m:.4f}", ha="center", va="bottom", fontsize=5.5,
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5.5)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("$\\Delta$MPD vs None", fontsize=7)
    ax.set_ylim(0, max(means) * 1.45)
    ax.axhline(0, color="black", linewidth=0.4)

    _save(fig, "fig_tricks_delta")


# ── Main ──────────────────────────────────────────────────────────────────────

ALL_ITEMS = ["fig4a", "fig4b", "fig5", "fig7", "fig_a1", "fig_tricks",
             "tab_arch", "tab_tricks"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=None)
    p.add_argument("--baseline-ucr", type=float, default=None)
    args = p.parse_args()

    setup_style()

    def go(name: str) -> bool:
        return args.only is None or args.only == name

    if go("fig4a"):
        print("Fig 4A: Persona Depth (3 panels)")
        plot_fig4a()

    if go("fig4b"):
        print("Fig 4B: Persona Depth (mean ± SE)")
        plot_fig4b()

    if go("fig5"):
        print("Fig 5: β-Diversity UCR")
        ucr = args.baseline_ucr
        cal_file = PROJECT_ROOT / "results" / "baseline_calibration.json"
        if ucr is None and cal_file.exists():
            with open(cal_file) as f:
                cal = json.load(f)
            vals = [v["split_half"]["ucr"] for v in cal.values()
                    if v.get("split_half")]
            if vals:
                ucr = np.mean(vals)
                print(f"  baseline UCR: {ucr:.1%}")
        plot_fig5(baseline_ucr=ucr)

    if go("fig7"):
        print("Fig 7: Interaction Effects")
        plot_fig7()

    if go("fig_a1"):
        print("Fig A1: Heatmap")
        plot_fig_a1()

    if go("fig_tricks"):
        print("Fig Tricks: Low-cost Strategies ΔMPD")
        plot_fig_tricks()

    if go("tab_arch"):
        print("Tab: Architecture MPD")
        generate_tab_architecture()

    if go("tab_tricks"):
        print("Tab: Low-cost Tricks")
        generate_tab_tricks()

    print(f"\nDone → {FIG_DIR}")


if __name__ == "__main__":
    main()
