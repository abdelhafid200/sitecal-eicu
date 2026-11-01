# ================================================================
# 10_figures: Figures 1 and 2 of the paper (vector PDF and 300 dpi PNG)
# Reads: resultats_finaux/master_frontier.csv and master_frontier_casemix.csv
# Output: resultats_finaux/figures_article/
# ================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MASTER_DIR = "resultats_finaux"
FIG_DIR    = os.path.join(MASTER_DIR, "figures_article")
os.makedirs(FIG_DIR, exist_ok=True)

F = pd.concat([pd.read_csv(os.path.join(MASTER_DIR, "master_frontier.csv")),
               pd.read_csv(os.path.join(MASTER_DIR, "master_frontier_casemix.csv"))], ignore_index=True)
F["reglage"] = F["reglage"].astype(str)
G = (F.groupby(["alpha", "score_model", "condition", "famille", "reglage"])
       .agg(ex=("exces", "mean"), ex_sd=("exces", "std"),
            exs=("exces_petits", "mean"), exs_sd=("exces_petits", "std"),
            al_s=("alertes_petits", "mean"))
       .reset_index())
G["worst"] = np.maximum(G["ex"], G["exs"])          # valid when both excesses are ≤ 2 points
close = lambda s, v: (s - v).abs() < 1e-9

def pick(alpha, fam, knob, sm, cond):
    return G[close(G["alpha"], alpha) & (G["famille"] == fam) & (G["reglage"] == knob)
             & (G["score_model"] == sm) & (G["condition"] == cond)]

# First three hues of the reference palette (checked for colour vision deficiency) + neutral inks
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.titlesize": 8.5,
                     "axes.labelsize": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

def finish(ax, xgrid=False):
    ax.grid(axis="y", color=GRID, lw=0.6)
    if xgrid:
        ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.6, labelcolor=INK2)
    ax.axhline(2, color=MUTED, lw=0.8, ls="--", zorder=1)   # validity margin
    ax.axhline(0, color=AXIS, lw=0.8, zorder=1)

ALPHAS = [0.10, 0.20]

# ------------------------------------------------------------
# Figure 1: pooled excess by condition (all and small hospitals)
# ------------------------------------------------------------
CONDS  = ["réel", "case-mix pur", "rot a=0", "rot a=0.5", "rot a=1"]
CLABEL = ["Real\noutcomes", "Pure\ncase-mix", "Concept\nshift\na = 0", "Concept\nshift\na = 0.5",
          "Concept\nshift\na = 1"]
SERIES = [("pool_conformal", "-",     "global",       "Pooled threshold, global model",             ORANGE, "s", -0.15),
          ("pool_conformal", "-",     "personalized", "Pooled threshold, personalized models",      BLUE,   "o",  0.00),
          ("shrink_pool",    "n0=10", "personalized", r"Shrinkage $n_0$ = 10, personalized models", AQUA,   "^",  0.15)]
xs = np.arange(len(CONDS))
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
for j, alpha in enumerate(ALPHAS):
    for i, (col, sdcol, who) in enumerate([("ex", "ex_sd", "All hospitals"),
                                          ("exs", "exs_sd", "Smallest third of hospitals")]):
        ax = axes[i, j]
        for fam, knob, sm, lab, color, mk, dx in SERIES:
            y, e = [], []
            for c in CONDS:
                r = pick(alpha, fam, knob, sm, c)
                y.append(100 * r[col].iloc[0] if len(r) else np.nan)
                e.append(100 * r[sdcol].iloc[0] if len(r) and pd.notna(r[sdcol].iloc[0]) else 0.0)
            ax.errorbar(xs + dx, y, yerr=e, color=color, marker=mk, ms=5, lw=1.4, elinewidth=0.8,
                        capsize=2, markeredgecolor="white", markeredgewidth=0.6, label=lab, zorder=3)
        finish(ax)
        ax.axvline(0.5, color=GRID, lw=0.8, zorder=0)       # separates real from semi-synthetic conditions
        ax.set_title(f"({'abcd'[2 * i + j]}) {who}, {1 - alpha:.0%} target", loc="left", color=INK)
        ax.set_xticks(xs)
        ax.set_xticklabels(CLABEL)
        if j == 0:
            ax.set_ylabel("Excess hospitals with a significant\nshortfall (percentage points)")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=7.5)
fig.tight_layout(rect=[0, 0.06, 1, 1])
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG_DIR, f"Fig1_pooling_excess.{ext}"), dpi=300, metadata={"CreationDate": None})
plt.show()

# ------------------------------------------------------------
# Figure 2: alerts in small hospitals vs worst excess (real outcomes, personalized models)
# ------------------------------------------------------------
FAMS = [("shrink_pool",      r"Shrinkage ($n_0$ pseudo-deaths)",         AQUA,   "^"),
        ("pool_bayes",       r"Empirical Bayes, all hospitals ($\eta$)", BLUE,   "o"),
        ("donors_conformal", r"Donor hospitals ($M$)",                   ORANGE, "s"),
        ("donors_bayes",     r"Empirical Bayes, donors ($\eta$)",        MUTED,  "D")]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
for j, (ax, alpha) in enumerate(zip(axes, ALPHAS)):
    sub = G[close(G["alpha"], alpha) & (G["score_model"] == "personalized") & (G["condition"] == "réel")]
    for fam, lab, color, mk in FAMS:
        d = sub[sub["famille"] == fam].sort_values("al_s")
        ax.plot(100 * d["al_s"], 100 * d["worst"], "-", color=color, lw=1.2, marker=mk, ms=4.5,
                markeredgecolor="white", markeredgewidth=0.5, label=lab, zorder=3)
    for fam, lab, mk in [("pool_conformal", "Pooled threshold", "X"), ("local_conformal", "Local threshold", "P")]:
        d = sub[sub["famille"] == fam]
        if len(d):
            ax.plot(100 * d["al_s"], 100 * d["worst"], marker=mk, ms=7, color=INK, ls="none",
                    markeredgecolor="white", markeredgewidth=0.6, label=lab, zorder=4)
    for knob, txt, off in [("n0=10", r"$n_0$ = 10", (40, 18)), ("n0=19", r"$n_0$ = 19", (14, 16))]:
        d = sub[(sub["famille"] == "shrink_pool") & (sub["reglage"] == knob)]
        if len(d):
            ax.annotate(txt, (100 * d["al_s"].iloc[0], 100 * d["worst"].iloc[0]), xytext=off,
                        textcoords="offset points", fontsize=7, color=INK2,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    finish(ax, xgrid=True)
    ax.set_xlabel("Patients flagged in the smallest third of hospitals (%)")
    ax.set_title(f"({'ab'[j]}) Real outcomes, {1 - alpha:.0%} target", loc="left", color=INK)
axes[0].set_ylabel("Worst excess, all or small hospitals\n(percentage points)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=7.5)
fig.tight_layout(rect=[0, 0.15, 1, 1])
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG_DIR, f"Fig2_frontier.{ext}"), dpi=300, metadata={"CreationDate": None})
plt.show()
print(f"✓ Figures enregistrées dans {FIG_DIR}")
