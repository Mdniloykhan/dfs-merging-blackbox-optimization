"""
Phase 4: Figure Generation for IEEE TIFS Paper
Author: Md. Robiul Islam Niloy
Generates all 5 publication-quality figures from CSV results

Requirements:
pip install matplotlib seaborn scipy pandas numpy

Output: C:/Users/user1/Desktop/jailbreak_q1/figures/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── PATHS ────────────────────────────────────────────────────
BASE_DIR     = Path("C:/Users/user1/Desktop/jailbreak_q1")
RESULTS_DIR  = BASE_DIR / "results"
PHASE2_DIR   = BASE_DIR / "phase2_results"
PHASE3_DIR   = BASE_DIR / "phase3_results"
FIGURES_DIR  = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── IEEE TIFS STYLE ───────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "Times New Roman",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linewidth":   0.5,
})

# ── COLOR PALETTE ─────────────────────────────────────────────
COLORS = {
    "blue":   "#2166AC",
    "red":    "#D6604D",
    "green":  "#1A7941",
    "orange": "#E08214",
    "purple": "#762A83",
    "gray":   "#878787",
    "light_blue": "#92C5DE",
    "light_red":  "#F4A582",
}

MODEL_COLORS = {
    "mistral_7b":  "#D6604D",
    "llama31_8b":  "#E08214",
    "llama32_3b":  "#762A83",
    "llama3_8b":   "#2166AC",
    "qwen2_7b":    "#1A7941",
    "gemma2_9b":   "#878787",
    "phi4_14b":    "#4393C3",
}

MODEL_LABELS = {
    "mistral_7b":  "Mistral-7B",
    "llama31_8b":  "Llama-3.1-8B",
    "llama32_3b":  "Llama-3.2-3B",
    "llama3_8b":   "Llama-3-8B",
    "qwen2_7b":    "Qwen2.5-7B",
    "gemma2_9b":   "Gemma2-9B",
    "phi4_14b":    "Phi-4-14B",
}

FAMILY_LABELS = {
    "direct_request":      "Direct Request",
    "roleplay_jailbreak":  "Roleplay",
    "hypothetical_framing":"Hypothetical",
    "instruction_override":"Override",
    "obfuscation_attack":  "Obfuscation",
}

# ── LOAD DATA ─────────────────────────────────────────────────
def load_all_results():
    phase1_files = list(RESULTS_DIR.glob("*_results.csv"))
    phase2_files = list(PHASE2_DIR.glob("*_results.csv"))
    all_files = phase1_files + phase2_files
    if not all_files:
        raise FileNotFoundError("No result CSV files found.")
    dfs = []
    for f in all_files:
        df = pd.read_csv(f)
        if "model" in df.columns:
            dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    return combined

def load_defense_results():
    files = list(PHASE3_DIR.glob("*_results.csv"))
    if not files:
        return None
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)

# ══════════════════════════════════════════════════════════════
# FIGURE 1: ASR HEATMAP
# ══════════════════════════════════════════════════════════════
def figure1_heatmap(combined):
    print("Generating Figure 1: ASR Heatmap...")

    pivot = combined.groupby(["model","attack_family"])["jailbreak_success"].mean() * 100
    pivot = pivot.reset_index()
    pivot = pivot.pivot(index="model", columns="attack_family", values="jailbreak_success")

    # Reorder rows by overall ASR
    overall = combined.groupby("model")["jailbreak_success"].mean() * 100
    row_order = overall.sort_values(ascending=False).index.tolist()
    col_order = ["direct_request","roleplay_jailbreak","hypothetical_framing",
                 "instruction_override","obfuscation_attack"]

    pivot = pivot.reindex(index=row_order, columns=col_order)
    pivot.index   = [MODEL_LABELS.get(m, m) for m in pivot.index]
    pivot.columns = [FAMILY_LABELS.get(c, c) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(7.16, 3.2))

    sns.heatmap(
        pivot, ax=ax,
        annot=True, fmt=".0f",
        annot_kws={"size": 9, "weight": "bold"},
        cmap="RdYlGn_r",
        vmin=0, vmax=70,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "ASR (%)", "shrink": 0.8}
    )

    ax.set_xlabel("Attack Family", fontsize=10, labelpad=6)
    ax.set_ylabel("Model", fontsize=10, labelpad=6)
    ax.set_title("Figure 1: Attack Success Rate Heatmap (7 Models × 5 Attack Families)",
                 fontsize=10, pad=8)
    ax.tick_params(axis="x", rotation=20)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    path = FIGURES_DIR / "fig1_heatmap.pdf"
    plt.savefig(path)
    plt.savefig(str(path).replace(".pdf",".png"))
    plt.close()
    print(f"  Saved: {path}")

# ══════════════════════════════════════════════════════════════
# FIGURE 2: OVERALL ASR WITH CONFIDENCE INTERVALS
# ══════════════════════════════════════════════════════════════
def figure2_overall_asr(combined):
    print("Generating Figure 2: Overall ASR with CI...")

    models = combined.groupby("model")["jailbreak_success"].apply(list)
    results = []
    for model, vals in models.items():
        arr = np.array(vals)
        n   = len(arr)
        mu  = arr.mean() * 100
        se  = stats.sem(arr) * 100
        ci  = stats.t.interval(0.95, df=n-1, loc=mu, scale=se)
        results.append({
            "model": model, "asr": mu,
            "ci_lo": mu - ci[0], "ci_hi": ci[1] - mu
        })

    df = pd.DataFrame(results).sort_values("asr", ascending=True)
    df["label"] = df["model"].map(MODEL_LABELS)
    df["color"] = df["model"].map(MODEL_COLORS)

    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    bars = ax.barh(
        df["label"], df["asr"],
        xerr=[df["ci_lo"], df["ci_hi"]],
        color=df["color"], alpha=0.85,
        error_kw={"elinewidth": 1.2, "capsize": 3, "ecolor": "black"},
        height=0.6
    )

    for bar, row in zip(bars, df.itertuples()):
        ax.text(bar.get_width() + row.ci_hi + 0.8, bar.get_y() + bar.get_height()/2,
                f"{row.asr:.1f}%", va="center", ha="left", fontsize=8.5)

    ax.set_xlabel("Overall Attack Success Rate (%)", fontsize=10)
    ax.set_title("Figure 2: Overall ASR Across 7 LLMs\n(with 95% Confidence Intervals)",
                 fontsize=10, pad=6)
    ax.set_xlim(0, 68)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = FIGURES_DIR / "fig2_overall_asr.pdf"
    plt.savefig(path)
    plt.savefig(str(path).replace(".pdf",".png"))
    plt.close()
    print(f"  Saved: {path}")

# ══════════════════════════════════════════════════════════════
# FIGURE 3: DEFENSE COMPARISON
# ══════════════════════════════════════════════════════════════
def figure3_defense(defense_df):
    print("Generating Figure 3: Defense Comparison...")

    attack_df = defense_df[defense_df["prompt_type"]=="attack"]
    baselines = {"mistral_7b": 44.0, "llama31_8b": 48.0}
    defenses  = ["keyword_blacklist","self_reminder","paraphrase","perplexity_filter"]
    def_labels = {"keyword_blacklist":"Keyword\nBlacklist",
                  "self_reminder":"Self-Reminder",
                  "paraphrase":"Paraphrase",
                  "perplexity_filter":"Perplexity\nFilter"}

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2), sharey=False)

    for ax, (model, baseline) in zip(axes, baselines.items()):
        asrs, errs_lo, errs_hi = [], [], []
        for defense in defenses:
            sub = attack_df[
                (attack_df["defense"]==defense) &
                (attack_df["model"]==model)
            ]["jailbreak_success"].values
            if len(sub) == 0:
                asrs.append(baseline); errs_lo.append(0); errs_hi.append(0)
                continue
            mu = sub.mean() * 100
            se = stats.sem(sub) * 100
            ci = stats.t.interval(0.95, df=len(sub)-1, loc=mu, scale=se)
            asrs.append(mu)
            errs_lo.append(mu - ci[0])
            errs_hi.append(ci[1] - mu)

        x = np.arange(len(defenses))
        colors = ["#D6604D" if a > baseline else "#1A7941" for a in asrs]

        bars = ax.bar(x, asrs,
                      yerr=[errs_lo, errs_hi],
                      color=colors, alpha=0.8,
                      error_kw={"elinewidth":1.2,"capsize":3,"ecolor":"black"},
                      width=0.55)

        ax.axhline(y=baseline, color="black", linestyle="--",
                   linewidth=1.2, label=f"Baseline ({baseline}%)")
        ax.set_xticks(x)
        ax.set_xticklabels([def_labels[d] for d in defenses], fontsize=8.5)
        ax.set_ylabel("Residual ASR (%)", fontsize=9)
        ax.set_ylim(0, 75)
        ax.set_title(f"{MODEL_LABELS[model]}", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

        for bar, asr in zip(bars, asrs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f"{asr:.0f}%", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Figure 3: Defense Evaluation — Residual ASR After Each Defense\n"
                 "(dashed line = baseline, green = improvement, red = degradation)",
                 fontsize=9.5, y=1.02)
    plt.tight_layout()
    path = FIGURES_DIR / "fig3_defense_comparison.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(str(path).replace(".pdf",".png"), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# ══════════════════════════════════════════════════════════════
# FIGURE 4: MODEL SIZE VS ROBUSTNESS
# ══════════════════════════════════════════════════════════════
def figure4_size_vs_robustness(combined):
    print("Generating Figure 4: Model Size vs Robustness...")

    size_map = {
        "llama32_3b":3,"mistral_7b":7,"qwen2_7b":7,
        "llama3_8b":8,"llama31_8b":8,"gemma2_9b":9,"phi4_14b":14,
    }
    org_map = {
        "llama32_3b":"Meta","mistral_7b":"Mistral AI","qwen2_7b":"Alibaba",
        "llama3_8b":"Meta","llama31_8b":"Meta","gemma2_9b":"Google","phi4_14b":"Microsoft",
    }

    rows = []
    for model, size in size_map.items():
        sub = combined[combined["model"]==model]
        if len(sub) == 0:
            continue
        asr = sub["jailbreak_success"].mean() * 100
        rows.append({
            "model": model, "size": size, "asr": asr,
            "label": MODEL_LABELS.get(model, model),
            "org": org_map.get(model,""),
            "color": MODEL_COLORS.get(model,"gray"),
        })

    df = pd.DataFrame(rows)
    corr, p_val = stats.spearmanr(df["size"], df["asr"])

    fig, ax = plt.subplots(figsize=(5.0, 3.8))

    for _, row in df.iterrows():
        ax.scatter(row["size"], row["asr"],
                   color=row["color"], s=90, zorder=5,
                   edgecolors="white", linewidth=0.8)
        offset_x = 0.3
        offset_y = 1.2
        ax.annotate(row["label"],
                    xy=(row["size"], row["asr"]),
                    xytext=(row["size"]+offset_x, row["asr"]+offset_y),
                    fontsize=8, color=row["color"])

    # Trend line
    z = np.polyfit(df["size"], df["asr"], 1)
    p = np.poly1d(z)
    xs = np.linspace(df["size"].min()-0.5, df["size"].max()+0.5, 100)
    ax.plot(xs, p(xs), "--", color=COLORS["gray"], linewidth=1, alpha=0.6)

    sig = "p<0.05" if p_val < 0.05 else "ns"
    ax.text(0.05, 0.95,
            f"Spearman r={corr:.2f}, {sig}",
            transform=ax.transAxes, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Model Size (Billion Parameters)", fontsize=10)
    ax.set_ylabel("Overall Attack Success Rate (%)", fontsize=10)
    ax.set_title("Figure 4: Model Size vs Adversarial Robustness\n(Spearman correlation)",
                 fontsize=10, pad=6)
    ax.set_xlim(1, 16)
    ax.set_ylim(0, 60)

    plt.tight_layout()
    path = FIGURES_DIR / "fig4_size_vs_robustness.pdf"
    plt.savefig(path)
    plt.savefig(str(path).replace(".pdf",".png"))
    plt.close()
    print(f"  Saved: {path}")

# ══════════════════════════════════════════════════════════════
# FIGURE 5: DEFENSE LATENCY VS EFFECTIVENESS
# ══════════════════════════════════════════════════════════════
def figure5_latency_tradeoff(defense_df):
    print("Generating Figure 5: Defense Latency vs Effectiveness Tradeoff...")

    attack_df = defense_df[defense_df["prompt_type"]=="attack"]
    baselines = {"mistral_7b": 44.0, "llama31_8b": 48.0}

    defenses  = ["keyword_blacklist","self_reminder","paraphrase","perplexity_filter"]
    def_colors = {
        "keyword_blacklist": COLORS["blue"],
        "self_reminder":     COLORS["green"],
        "paraphrase":        COLORS["orange"],
        "perplexity_filter": COLORS["purple"],
    }
    def_labels = {
        "keyword_blacklist":"Keyword Blacklist",
        "self_reminder":"Self-Reminder",
        "paraphrase":"Paraphrase",
        "perplexity_filter":"Perplexity Filter",
    }
    model_markers = {"mistral_7b": "o", "llama31_8b": "s"}
    model_short   = {"mistral_7b": "Mistral", "llama31_8b": "Llama-3.1"}

    fig, ax = plt.subplots(figsize=(6.0, 3.8))

    plotted_defenses = set()
    for model, baseline in baselines.items():
        for defense in defenses:
            sub = attack_df[
                (attack_df["defense"]==defense) &
                (attack_df["model"]==model)
            ]
            if len(sub) == 0:
                continue
            residual_asr = sub["jailbreak_success"].mean() * 100
            reduction    = baseline - residual_asr
            avg_latency  = sub["total_latency_s"].mean()

            label = def_labels[defense] if defense not in plotted_defenses else None
            ax.scatter(avg_latency, reduction,
                       color=def_colors[defense],
                       marker=model_markers[model],
                       s=100, zorder=5,
                       edgecolors="white", linewidth=0.8,
                       label=label)
            plotted_defenses.add(defense)

            ax.annotate(f"{model_short[model]}\n{def_labels[defense].split()[0]}",
                        xy=(avg_latency, reduction),
                        xytext=(avg_latency+0.08, reduction+0.8),
                        fontsize=7.5, color=def_colors[defense])

    # Ideal quadrant annotation
    ax.axhline(y=0, color="black", linewidth=0.8, alpha=0.5)
    ax.axvline(x=attack_df["total_latency_s"].mean(),
               color="gray", linewidth=0.8, linestyle=":", alpha=0.5)

    ax.set_xlabel("Average Inference Latency (seconds)", fontsize=10)
    ax.set_ylabel("ASR Reduction (percentage points)", fontsize=10)
    ax.set_title("Figure 5: Defense Effectiveness vs Latency Tradeoff\n"
                 "(higher reduction + lower latency = better; ● Mistral  ■ Llama-3.1)",
                 fontsize=9.5, pad=6)

    legend_patches = [
        mpatches.Patch(color=c, label=def_labels[d])
        for d, c in def_colors.items()
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right",
              framealpha=0.8)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = FIGURES_DIR / "fig5_latency_tradeoff.pdf"
    plt.savefig(path)
    plt.savefig(str(path).replace(".pdf",".png"))
    plt.close()
    print(f"  Saved: {path}")

# ── MAIN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("PHASE 4: IEEE TIFS FIGURE GENERATION")
    print("="*60)
    print(f"Output directory: {FIGURES_DIR}\n")

    print("Loading Phase 1 + Phase 2 data...")
    combined = load_all_results()
    print(f"  Loaded {len(combined)} rows from {combined['model'].nunique()} models\n")

    print("Loading Phase 3 defense data...")
    defense_df = load_defense_results()
    if defense_df is not None:
        print(f"  Loaded {len(defense_df)} defense evaluation rows\n")

    figure1_heatmap(combined)
    figure2_overall_asr(combined)

    if defense_df is not None:
        figure3_defense(defense_df)
        figure5_latency_tradeoff(defense_df)

    figure4_size_vs_robustness(combined)

    print("\n" + "="*60)
    print("ALL FIGURES GENERATED")
    print("="*60)
    print(f"Saved to: {FIGURES_DIR}")
    print("\nFiles:")
    for f in sorted(FIGURES_DIR.glob("fig*")):
        print(f"  {f.name}")
    print("\nNext step: Include PDF figures in your LaTeX paper.")
