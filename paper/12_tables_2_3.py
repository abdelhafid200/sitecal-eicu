# ================================================================
# 12_tables_2_3: exact values of Tables 2 and 3, written as LaTeX table rows
# Reads: resultats_finaux/master_frontier.csv and master_frontier_casemix.csv
# ================================================================
import os
import numpy as np
import pandas as pd

MASTER_DIR = "resultats_finaux"
OUT_DIR    = os.path.join(MASTER_DIR, "tables_article")
os.makedirs(OUT_DIR, exist_ok=True)
EXCES_MAX  = 0.02
F = pd.concat([pd.read_csv(os.path.join(MASTER_DIR, "master_frontier.csv")),
               pd.read_csv(os.path.join(MASTER_DIR, "master_frontier_casemix.csv"))], ignore_index=True)
F["reglage"] = F["reglage"].astype(str)
G = (F.groupby(["alpha", "score_model", "condition", "famille", "reglage"])
       .agg(ex=("exces", "mean"), exs=("exces_petits", "mean"),
            al=("alertes", "mean"), al_s=("alertes_petits", "mean"), sens=("sens_moy", "mean"))
       .reset_index())
G["valid"] = (G["ex"] <= EXCES_MAX) & (G["exs"] <= EXCES_MAX)
close = lambda s, v: (s - v).abs() < 1e-9

def val(alpha, fam, knob, sm, cond, col):
    x = G[close(G["alpha"], alpha) & (G["famille"] == fam) & (G["reglage"] == knob)
          & (G["score_model"] == sm) & (G["condition"] == cond)]
    return float(x[col].iloc[0]) if len(x) else np.nan

def pts(v):
    if np.isnan(v):
        return "--"
    s = f"{100 * v:+.1f}"
    return "0.0" if s in ("+0.0", "-0.0") else s.replace("-", "$-$")

# ---------------- Table 2: pooled excess (all / small hospitals) ----------------
CONDS = [("réel", "Real outcomes"), ("case-mix pur", "Pure case-mix"),
         ("rot a=0", "Concept shift, $a=0$"), ("rot a=0.5", "Concept shift, $a=0.5$"),
         ("rot a=1", "Concept shift, $a=1$")]
t2 = []
for cond, lab in CONDS:
    cells = []
    for sm in ["personalized", "global"]:
        for alpha in [0.10, 0.20]:
            cells += [pts(val(alpha, "pool_conformal", "-", sm, cond, "ex")),
                      pts(val(alpha, "pool_conformal", "-", sm, cond, "exs"))]
    t2.append(f"{lab:<24} & " + " & ".join(cells) + r" \\")

# ---------------- Table 3: methods ----------------
METHODS = [("local_conformal",  "-",        "Local"),
           ("pool_conformal",   "-",        "Pooled"),
           ("shrink_pool",      "n0=10",    r"Shrinkage, $n_0=10$"),
           ("shrink_pool",      "n0=19",    r"Shrinkage, $n_0=19$"),
           ("donors_conformal", "M=10",     r"Donors, $M=10$"),
           ("pool_bayes",       "eta=0.05", r"Empirical Bayes, all, $\eta=0.05$"),
           ("donors_bayes",     "eta=0.05", r"Empirical Bayes, donors, $\eta=0.05$"),
           ("pool_bayes",       "eta=0.2",  r"Empirical Bayes, all, $\eta=0.2$\textsuperscript{a}")]
t3 = []
for fam, knob, lab in METHODS:
    row = [lab]
    for alpha in [0.10, 0.20]:
        g = G[close(G["alpha"], alpha) & (G["famille"] == fam) & (G["reglage"] == knob)]
        row.append(f"{int(g['valid'].sum())}/{len(g)}")
    for alpha in [0.10, 0.20]:
        row.append(f"{100 * val(alpha, fam, knob, 'personalized', 'réel', 'al_s'):.1f}")
    for alpha in [0.10, 0.20]:
        row.append(f"{val(alpha, fam, knob, 'personalized', 'réel', 'sens'):.3f}")
    t3.append(" & ".join(row) + r" \\")

for name, lines in [("table2_pooled_excess", t2), ("table3_methods", t3)]:
    with open(os.path.join(OUT_DIR, f"{name}.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n% ---- {name} ----")
    print("\n".join(lines))

# ---------------- Check of the figures quoted in the text ----------------
print("\nChiffres du texte (réel, modèles personnalisés) :")
for fam, knob, lab in [("local_conformal", "-", "local"), ("shrink_pool", "n0=10", "shrink n0=10")]:
    for alpha in [0.10, 0.20]:
        print(f"  {lab:<13} α={alpha:.2f} : alertes tous {100 * val(alpha, fam, knob, 'personalized', 'réel', 'al'):.1f} % "
              f"| petits {100 * val(alpha, fam, knob, 'personalized', 'réel', 'al_s'):.1f} %")
print(f"\n✓ Lignes LaTeX enregistrées dans {OUT_DIR}")
