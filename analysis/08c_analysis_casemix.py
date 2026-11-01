# ================================================================
# 08c_analysis_casemix: H3'c and H4b (A2a and A2b of amendment A2, fixed before any case-mix run)
# Compares pure case-mix with rotation at a = 0 and with real outcomes (read from master_frontier.csv).
# ================================================================
import os
import numpy as np
import pandas as pd

MASTER_DIR = "resultats_finaux"
EXCES_MAX  = 0.02
close = lambda s, v: (s - v).abs() < 1e-9
short = lambda m: "perso" if m == "personalized" else "global"
SMS   = ["personalized", "global"]

FC = pd.read_csv(os.path.join(MASTER_DIR, "master_frontier_casemix.csv")); FC["reglage"] = FC["reglage"].astype(str)
CC = pd.read_csv(os.path.join(MASTER_DIR, "master_conditions_casemix.csv"))
FM = pd.read_csv(os.path.join(MASTER_DIR, "master_frontier.csv"));         FM["reglage"] = FM["reglage"].astype(str)

def agg(d):
    return (d.groupby(["alpha", "score_model", "famille", "reglage"])
             .agg(exces=("exces", "mean"), exces_sd=("exces", "std"), exces_petits=("exces_petits", "mean"),
                  alertes=("alertes", "mean"), alertes_petits=("alertes_petits", "mean"),
                  sens_moy=("sens_moy", "mean"), runs=("exces", "size"))
             .reset_index())
GC = agg(FC)                                                                   # pure case-mix
G0 = agg(FM[(FM["label_mode"] == "rotation") & close(FM["shift_a"], 0.0)])     # rotation at a = 0 (reference)
GR = agg(FM[FM["label_mode"] == "real"])                                       # real outcomes (reference)

def val(G, alpha, fam, knob, sm, col):
    x = G[close(G["alpha"], alpha) & (G["famille"] == fam) & (G["reglage"] == knob) & (G["score_model"] == sm)]
    return float(x[col].iloc[0]) if len(x) else np.nan

# ------------------------------------------------------------
# Description of the condition
# ------------------------------------------------------------
print(f"{'=' * 100}\n  CONDITION CASE-MIX PUR (moyennes sur {len(CC)} runs : tirages × seeds)\n{'=' * 100}")
print(CC[["mortalite", "exces_01b", "auroc_global", "auroc_personalized", "auroc_local"]].mean().round(3).to_string())

# ------------------------------------------------------------
# H3'c: are pooled thresholds invalid when only patient composition differs?
# ------------------------------------------------------------
print(f"\n{'=' * 100}\n  H3'c — excès du pool sur le hasard (points de %)\n{'=' * 100}")
for alpha in [0.10, 0.20]:
    e  = {sm: val(GC, alpha, "pool_conformal", "-", sm, "exces") for sm in SMS}
    sd = {sm: val(GC, alpha, "pool_conformal", "-", sm, "exces_sd") for sm in SMS}
    n  = np.nanmin([val(GC, alpha, "pool_conformal", "-", sm, "runs") for sm in SMS])
    ok = all(e[sm] > EXCES_MAX for sm in SMS)
    principal = abs(alpha - 0.10) < 1e-9
    verdict = ("CONFIRMÉE" if ok else "NON confirmée") if principal else ("oui" if ok else "non")
    print(f"\n  α = {alpha:.2f} | runs par modèle : {n:.0f}" + ("   ← TEST PRINCIPAL" if principal else ""))
    for sm in SMS:
        print(f"    {short(sm):<6} : case-mix pur {100 * e[sm]:+.1f} (± {100 * sd[sm]:.1f}) | "
              f"rotation a=0 {100 * val(G0, alpha, 'pool_conformal', '-', sm, 'exces'):+.1f} | "
              f"réel {100 * val(GR, alpha, 'pool_conformal', '-', sm, 'exces'):+.1f}")
    print(f"    H3'c (excès > +2 sous les deux modèles) → {verdict}")

# ------------------------------------------------------------
# T4: primary methods in the pure case-mix condition
# ------------------------------------------------------------
PRIMARY = [("local_conformal",  "-",        "local"),
           ("pool_conformal",   "-",        "pool"),
           ("shrink_pool",      "n0=10",    "shrink n0=10 (recommandé)"),
           ("shrink_pool",      "n0=19",    "shrink n0=19 (Shahid)"),
           ("donors_conformal", "M=10",     "donneurs conformal M=10"),
           ("pool_bayes",       "eta=0.05", "pool + bayes η=0.05"),
           ("donors_bayes",     "eta=0.05", "donneurs + bayes η=0.05"),
           ("pool_bayes",       "eta=0.2",  "pool + bayes η=0.2 (exploratoire)")]
for alpha in [0.10, 0.20]:
    rows = []
    for fam, knob, lab in PRIMARY:
        r = {"méthode": lab}
        for sm in SMS:
            e  = val(GC, alpha, fam, knob, sm, "exces")
            ep = val(GC, alpha, fam, knob, sm, "exces_petits")
            r[f"{short(sm)} excès"]          = f"{100 * e:+.1f}"
            r[f"{short(sm)} excès petits"]   = f"{100 * ep:+.1f}"
            r[f"{short(sm)} alertes"]        = f"{val(GC, alpha, fam, knob, sm, 'alertes'):.3f}"
            r[f"{short(sm)} alertes petits"] = f"{val(GC, alpha, fam, knob, sm, 'alertes_petits'):.3f}"
            r[f"{short(sm)} sens moy"]       = f"{val(GC, alpha, fam, knob, sm, 'sens_moy'):.3f}"
            r[f"{short(sm)} valide"]         = "oui" if (e <= EXCES_MAX and ep <= EXCES_MAX) else "NON"
        rows.append(r)
    T = pd.DataFrame(rows)
    print(f"\n{'=' * 160}\n  T4 — CASE-MIX PUR | α = {alpha:.2f} | méthodes principales (moyennes sur tirages × seeds)\n{'=' * 160}")
    with pd.option_context("display.width", 400, "display.max_columns", 50):
        print(T.to_string(index=False))
    T.to_csv(os.path.join(MASTER_DIR, f"T4_casemix_alpha{alpha:g}.csv"), index=False)

# ------------------------------------------------------------
# H4b: does shrinkage with n0 = 10 remain valid?
# ------------------------------------------------------------
A0  = 0.10
ok4 = all((val(GC, A0, "shrink_pool", "n0=10", sm, "exces") <= EXCES_MAX)
          and (val(GC, A0, "shrink_pool", "n0=10", sm, "exces_petits") <= EXCES_MAX) for sm in SMS)
print(f"\nH4b  shrink n0=10 valide sous les deux modèles (case-mix pur, α = 0.10) → "
      f"{'CONFIRMÉE' if ok4 else 'NON confirmée'}")
print(f"\n✓ Résultats enregistrés dans {MASTER_DIR}")
