# ================================================================
# 08d_analysis_small_hospitals: T5, excess violations for all hospitals and the smallest third (descriptive)
# Only reads resultats_finaux/master_frontier.csv, so it can run while 07c is running.
# ================================================================
import os
import numpy as np
import pandas as pd

MASTER_DIR = "resultats_finaux"
EXCES_MAX  = 0.02
F = pd.read_csv(os.path.join(MASTER_DIR, "master_frontier.csv"))
F["reglage"] = F["reglage"].astype(str)
short = lambda m: "perso" if m == "personalized" else "global"
COND_ORDER = [c for c in ["réel", "rot a=0", "rot a=0.5", "rot a=1"] if c in set(F["condition"])]
MODELS     = [m for m in ["personalized", "global"] if m in set(F["score_model"])]
PRIMARY = [("pool_conformal",   "-",        "pool"),
           ("shrink_pool",      "n0=10",    "shrink n0=10 (recommandé)"),
           ("shrink_pool",      "n0=19",    "shrink n0=19 (Shahid)"),
           ("donors_conformal", "M=10",     "donneurs conformal M=10"),
           ("pool_bayes",       "eta=0.05", "pool + bayes η=0.05"),
           ("donors_bayes",     "eta=0.05", "donneurs + bayes η=0.05"),
           ("pool_bayes",       "eta=0.2",  "pool + bayes η=0.2 (exploratoire)")]
G = (F.groupby(["alpha", "score_model", "condition", "famille", "reglage"])
       .agg(ex=("exces", "mean"), exp=("exces_petits", "mean"), exp_sd=("exces_petits", "std"))
       .reset_index())

for alpha in sorted(G["alpha"].unique()):
    rows = []
    for fam, knob, lab in PRIMARY:
        r = {"méthode": lab}
        for sm in MODELS:
            for c in COND_ORDER:
                x = G[(G["alpha"] == alpha) & (G["famille"] == fam) & (G["reglage"] == knob)
                      & (G["score_model"] == sm) & (G["condition"] == c)]
                if len(x):
                    flag = "" if (x["ex"].iloc[0] <= EXCES_MAX and x["exp"].iloc[0] <= EXCES_MAX) else " ✗"
                    r[f"{short(sm)} | {c}"] = (f"{100 * x['ex'].iloc[0]:+.0f} / {100 * x['exp'].iloc[0]:+.0f}"
                                               f" ±{100 * x['exp_sd'].iloc[0]:.0f}{flag}")
                else:
                    r[f"{short(sm)} | {c}"] = "-"
        rows.append(r)
    T = pd.DataFrame(rows)
    print(f"\n{'=' * 160}\n  T5 — α = {alpha:.2f} | excès : tous les hôpitaux / petits hôpitaux ± écart-type des petits "
          f"(points de %) | ✗ = invalide\n{'=' * 160}")
    with pd.option_context("display.width", 400, "display.max_columns", 50):
        print(T.to_string(index=False))
    T.to_csv(os.path.join(MASTER_DIR, f"T5_alpha{alpha:g}_exces_petits.csv"), index=False)
print(f"\n✓ T5 enregistré dans {MASTER_DIR}")
