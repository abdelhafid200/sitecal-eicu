# ================================================================
# 08_analysis_main: pre-specified analysis (docs/analysis_plan.md); choices must not change after seeing results
# Reads resultats_finaux/master_*.csv produced by 07.
# Produces: T0 conditions, T1 main table, T2 robust settings, mechanism figure and hypothesis checks.
# Printed labels H1, H3 and H4 correspond to P1, P2 and P3 in the analysis plan.
# ================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MASTER_DIR = "resultats_finaux"
EXCES_MAX  = 0.02
F = pd.read_csv(os.path.join(MASTER_DIR, "master_frontier.csv"))
C = pd.read_csv(os.path.join(MASTER_DIR, "master_conditions.csv"))
F["reglage"] = F["reglage"].astype(str)
COND_ORDER = [c for c in ["réel", "rot a=0", "rot a=0.5", "rot a=1"] if c in set(F["condition"])]
MODELS     = [m for m in ["personalized", "global"] if m in set(F["score_model"])]
PRIMARY = [("local_conformal",  "-",        "local"),
           ("pool_conformal",   "-",        "pool"),
           ("shrink_pool",      "n0=10",    "shrink n0=10 (recommandé)"),
           ("shrink_pool",      "n0=19",    "shrink n0=19 (Shahid)"),
           ("donors_conformal", "M=10",     "donneurs conformal M=10"),
           ("pool_bayes",       "eta=0.05", "pool + bayes η=0.05"),
           ("donors_bayes",     "eta=0.05", "donneurs + bayes η=0.05")]
short = lambda m: "perso" if m == "personalized" else "global"

# ------------------------------------------------------------
# T0: description of conditions
# ------------------------------------------------------------
T0 = (C.groupby("condition")
        .agg(runs=("model_seed", "size"), mortalite=("mortalite", "mean"), exces_01b=("exces_01b", "mean"),
             auroc_global=("auroc_global", "mean"), auroc_perso=("auroc_personalized", "mean"),
             auroc_local=("auroc_local", "mean"))
        .reindex(COND_ORDER))
print(f"{'=' * 100}\n  T0 — CONDITIONS (moyennes sur les seeds)\n{'=' * 100}")
print(T0.round(3).to_string())
T0.to_csv(os.path.join(MASTER_DIR, "T0_conditions.csv"))

# ------------------------------------------------------------
# Averages over seeds (group assignments × model seeds)
# ------------------------------------------------------------
G = (F.groupby(["alpha", "score_model", "condition", "famille", "reglage"])
       .agg(exces=("exces", "mean"), exces_sd=("exces", "std"),
            exces_petits=("exces_petits", "mean"),
            alertes=("alertes", "mean"), alertes_sd=("alertes", "std"),
            alertes_petits=("alertes_petits", "mean"), runs=("exces", "size"))
       .reset_index())
G["valide"] = (G["exces"] <= EXCES_MAX) & (G["exces_petits"] <= EXCES_MAX)
G.to_csv(os.path.join(MASTER_DIR, "T_moyennes_par_condition.csv"), index=False)

def pick(alpha, fam, knob, sm, cond):
    return G[(G["alpha"] == alpha) & (G["famille"] == fam) & (G["reglage"] == knob)
             & (G["score_model"] == sm) & (G["condition"] == cond)]

def get(alpha, fam, knob, sm, cond, col):
    x = pick(alpha, fam, knob, sm, cond)
    return float(x[col].iloc[0]) if len(x) else np.nan

def f_exces(x):
    if len(x) == 0:
        return "-"
    m, sd = 100 * x["exces"].iloc[0], x["exces_sd"].iloc[0]
    return f"{m:+.0f}" + (f" ±{100 * sd:.0f}" if pd.notna(sd) else "")

def f_col(col):
    return lambda x: "-" if len(x) == 0 else f"{x[col].iloc[0]:.3f}"

# ------------------------------------------------------------
# T1: main table (methods and settings fixed in advance)
# ------------------------------------------------------------
for alpha in sorted(G["alpha"].unique()):
    for key, titre, fmt in [("exces", "excès de violations (points de %, moyenne ± écart-type sur les seeds)", f_exces),
                            ("alertes", "taux d'alertes (tous les hôpitaux)", f_col("alertes")),
                            ("alertes_petits", "taux d'alertes (petits hôpitaux)", f_col("alertes_petits"))]:
        rows = []
        for fam, knob, lab in PRIMARY:
            r = {"méthode": lab}
            for sm in MODELS:
                for cond in COND_ORDER:
                    r[f"{short(sm)} | {cond}"] = fmt(pick(alpha, fam, knob, sm, cond))
            rows.append(r)
        T1 = pd.DataFrame(rows)
        print(f"\n{'=' * 130}\n  T1 — TABLEAU PRINCIPAL | α = {alpha:.2f} | {titre}\n{'=' * 130}")
        with pd.option_context("display.width", 300, "display.max_columns", 50):
            print(T1.to_string(index=False))
        T1.to_csv(os.path.join(MASTER_DIR, f"T1_alpha{alpha:g}_{key}.csv"), index=False)

# ------------------------------------------------------------
# T2: one fixed setting per method, tested in every cell (condition × model)
# ------------------------------------------------------------
for alpha in sorted(G["alpha"].unique()):
    g = G[G["alpha"] == alpha]
    rows = []
    for (fam, knob), d in g.groupby(["famille", "reglage"], sort=False):
        fails = d.loc[~d["valide"]]
        rows.append({"méthode": fam, "réglage": knob,
                     "valide": f"{int(d['valide'].sum())}/{len(d)}", "_nv": int(d["valide"].sum()),
                     "excès max": f"{100 * d['exces'].max():+.0f}",
                     "alertes moy": round(float(d["alertes"].mean()), 3),
                     "alertes petits moy": round(float(d["alertes_petits"].mean()), 3),
                     "échoue dans": ", ".join(f"{short(m)} {c}" for m, c in zip(fails["score_model"], fails["condition"])) or "-"})
    T2 = (pd.DataFrame(rows).sort_values(["_nv", "alertes moy"], ascending=[False, True])
            .drop(columns="_nv").reset_index(drop=True))
    print(f"\n{'=' * 130}\n  T2 — RÉGLAGES ROBUSTES | α = {alpha:.2f} | trié : le plus souvent valide, puis le moins d'alertes\n{'=' * 130}")
    with pd.option_context("display.width", 300, "display.max_colwidth", 90):
        print(T2.to_string(index=False))
    T2.to_csv(os.path.join(MASTER_DIR, f"T2_alpha{alpha:g}_reglages_robustes.csv"), index=False)

# ------------------------------------------------------------
# FIGURE: mechanism, pooled excess by concept shift intensity, global vs personalized models
# ------------------------------------------------------------
ROT = [c for c in COND_ORDER if c.startswith("rot")]
if ROT:
    a_vals = [float(c.split("=")[1]) for c in ROT]
    SERIES = [("pool_conformal", "-",     "global",       "pool — modèle global",          "#eb6834", "s"),
              ("pool_conformal", "-",     "personalized", "pool — modèles personnalisés",  "#2a78d6", "o"),
              ("shrink_pool",    "n0=10", "global",       "shrink n0=10 — modèle global",  "#1baf7a", "^")]
    INK, MUTED = "#1f1f1e", "#6b6b66"
    alphas = sorted(G["alpha"].unique())
    fig, axes = plt.subplots(1, len(alphas), figsize=(5.6 * len(alphas), 4.2), squeeze=False)
    for ax, alpha in zip(axes[0], alphas):
        for fam, knob, sm, lab, col, mk in SERIES:
            ys, es = [], []
            for cond in ROT:
                x = pick(alpha, fam, knob, sm, cond)
                ys.append(100 * x["exces"].iloc[0] if len(x) else np.nan)
                es.append(100 * x["exces_sd"].iloc[0] if len(x) and pd.notna(x["exces_sd"].iloc[0]) else 0.0)
            ax.errorbar(a_vals, ys, yerr=es, color=col, marker=mk, ms=7, lw=2, capsize=3,
                        markeredgecolor="white", label=lab)
        ax.axhline(0, color=MUTED, lw=1, ls=":")
        ax.axhline(100 * EXCES_MAX, color=MUTED, lw=1, ls="--")
        ax.set_xticks(a_vals)
        ax.set_xlabel("intensité du décalage de concept (a)", color=INK)
        ax.set_ylabel("excès de violations sur le hasard (points de %)", color=INK)
        ax.set_title(f"sensibilité cible ≥ {1 - alpha:.0%}", color=INK, fontsize=11, loc="left")
        ax.grid(True, color="#e6e5e0", lw=0.8); ax.set_axisbelow(True)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]: ax.spines[sp].set_color("#c9c8c2")
        ax.tick_params(colors=MUTED)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(SERIES), frameon=False, fontsize=9)
    fig.suptitle("Excès de violations selon l'intensité du décalage de concept (ligne pointillée = seuil de validité)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(os.path.join(MASTER_DIR, "F_mecanisme.png"), dpi=200)
    plt.show()

# ------------------------------------------------------------
# HYPOTHESIS CHECKS (criteria fixed in advance, α = 0.10)
# ------------------------------------------------------------
A0 = 0.10
print(f"\n{'=' * 100}\n  HYPOTHÈSES (α = {A0:.2f})\n{'=' * 100}")
if "réel" in COND_ORDER:
    v = [get(A0, "local_conformal", "-", sm, "réel", "alertes_petits") for sm in MODELS]
    print(f"H1  calibration locale, petits hôpitaux, données réelles : alertes = {np.nanmean(v):.3f} "
          f"→ {'CONFIRMÉE' if np.nanmean(v) >= 0.90 else 'NON confirmée'} (critère ≥ 0.90)")
if {"rot a=0", "rot a=1"} <= set(COND_ORDER) and {"global", "personalized"} <= set(MODELS):
    d_glob = get(A0, "pool_conformal", "-", "global", "rot a=1", "exces") - get(A0, "pool_conformal", "-", "global", "rot a=0", "exces")
    perso1 = get(A0, "pool_conformal", "-", "personalized", "rot a=1", "exces")
    ok3 = (d_glob >= 0.05) and (perso1 <= EXCES_MAX)
    print(f"H3  pool : hausse de l'excès sous modèle global de a=0 à a=1 = {100 * d_glob:+.0f} pts (critère ≥ +5) ; "
          f"excès sous modèles personnalisés à a=1 = {100 * perso1:+.0f} pts (critère ≤ +2) → {'CONFIRMÉE' if ok3 else 'NON confirmée'}")
sub = G[(G["alpha"] == A0) & (G["famille"] == "shrink_pool") & (G["reglage"] == "n0=10")]
if len(sub) and "réel" in COND_ORDER:
    all_valid = bool(sub["valide"].all())
    better = all(get(A0, "shrink_pool", "n0=10", sm, "réel", "alertes_petits")
                 < get(A0, "local_conformal", "-", sm, "réel", "alertes_petits") for sm in MODELS)
    print(f"H4  shrink n0=10 : valide dans {int(sub['valide'].sum())}/{len(sub)} cellules ; "
          f"moins d'alertes que le local chez les petits (réel, tous modèles) = {better} "
          f"→ {'CONFIRMÉE' if (all_valid and better) else 'NON confirmée'}")
for fam in ["pool_bayes", "donors_bayes"]:
    eb = G[(G["alpha"] == A0) & (G["famille"] == fam) & (G["reglage"] == "eta=0.05")]
    if len(eb) and len(sub):
        print(f"H5  {fam} η=0.05 : valide {int(eb['valide'].sum())}/{len(eb)} ; alertes moy {eb['alertes'].mean():.3f} "
              f"(petits {eb['alertes_petits'].mean():.3f}) contre shrink n0=10 {sub['alertes'].mean():.3f} "
              f"(petits {sub['alertes_petits'].mean():.3f}) — exploratoire")
print(f"\n✓ Tableaux et figure enregistrés dans {MASTER_DIR}")
