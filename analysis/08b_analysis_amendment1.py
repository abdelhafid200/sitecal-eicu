# ================================================================
# 08b_analysis_amendment1: hypothesis H3' (amendment A1), fixed after the QUICK preview and
# before analysis of the GROUP_SEED = 2 runs. Primary test: GROUP_SEED = 2 only (unseen assignments), α = 0.10.
# Printed labels H3'a and H3'b correspond to A1a and A1b in the analysis plan.
# Also produces T3: raw violations (real / random) and mean sensitivity of the primary methods.
# Run after 08, on the full plan (07 completed with QUICK = False).
# ================================================================
import os
import numpy as np
import pandas as pd

MASTER_DIR = "resultats_finaux"
EXCES_MAX  = 0.02
F = pd.read_csv(os.path.join(MASTER_DIR, "master_frontier.csv"))
F["reglage"] = F["reglage"].astype(str)
short = lambda m: "perso" if m == "personalized" else "global"
close = lambda s, v: (s - v).abs() < 1e-9

# ------------------------------------------------------------
# H3': pooled excess under rotation, averaged over model seeds
# ------------------------------------------------------------
def pool_exces(sub, sm, a, alpha):
    x = sub[(sub["famille"] == "pool_conformal") & (sub["score_model"] == sm)
            & (sub["label_mode"] == "rotation") & close(sub["shift_a"], a) & close(sub["alpha"], alpha)]
    return (float(x["exces"].mean()) if len(x) else np.nan), int(len(x))

SUBSETS = [("GROUP_SEED = 2 (CONFIRMATION)",                       F[F["group_seed"] == 2]),
           ("GROUP_SEED = 1 (descriptif, a servi à formuler H3')", F[F["group_seed"] == 1]),
           ("GROUP_SEED = 1 + 2 (descriptif)",                     F[F["label_mode"] == "rotation"])]

print(f"{'=' * 100}\n  H3' — excès du pool sur le hasard, points de %\n{'=' * 100}")
for name, sub in SUBSETS:
    for alpha in [0.10, 0.20]:
        g0, ng0 = pool_exces(sub, "global",       0.0, alpha)
        p0, np0 = pool_exces(sub, "personalized", 0.0, alpha)
        g1, ng1 = pool_exces(sub, "global",       1.0, alpha)
        p1, np1 = pool_exces(sub, "personalized", 1.0, alpha)
        n = min(ng0, np0, ng1, np1)
        if n == 0:
            print(f"\n  {name} | α = {alpha:.2f} : pas encore de runs")
            continue
        ok_a = (g0 > EXCES_MAX) and (p0 > EXCES_MAX)
        ok_b = ((p0 - p1) >= 0.05) and (p1 <= EXCES_MAX) and (g1 > EXCES_MAX)
        principal = name.startswith("GROUP_SEED = 2") and abs(alpha - 0.10) < 1e-9
        tag = (lambda ok: "CONFIRMÉE" if ok else "NON confirmée") if principal else (lambda ok: "oui" if ok else "non")
        print(f"\n  {name} | α = {alpha:.2f} | runs par cellule : {n}" + ("   ← TEST PRINCIPAL" if principal else ""))
        print(f"    H3'a  a=0 : excès global {100 * g0:+.1f}, perso {100 * p0:+.1f} "
              f"(critère : les deux > +2) → {tag(ok_a)}")
        print(f"    H3'b  perso : a=0 {100 * p0:+.1f} → a=1 {100 * p1:+.1f} (baisse {100 * (p0 - p1):+.1f}, "
              f"critère ≥ +5 et a=1 ≤ +2) ; global a=1 {100 * g1:+.1f} (critère > +2) → {tag(ok_b)}")

# ------------------------------------------------------------
# T3: raw violations (real / random, % of hospitals) and mean sensitivity, averaged over seeds
# ------------------------------------------------------------
COND_ORDER = [c for c in ["réel", "rot a=0", "rot a=0.5", "rot a=1"] if c in set(F["condition"])]
MODELS     = [m for m in ["personalized", "global"] if m in set(F["score_model"])]
PRIMARY = [("local_conformal",  "-",        "local"),
           ("pool_conformal",   "-",        "pool"),
           ("shrink_pool",      "n0=10",    "shrink n0=10 (recommandé)"),
           ("shrink_pool",      "n0=19",    "shrink n0=19 (Shahid)"),
           ("donors_conformal", "M=10",     "donneurs conformal M=10"),
           ("pool_bayes",       "eta=0.05", "pool + bayes η=0.05"),
           ("donors_bayes",     "eta=0.05", "donneurs + bayes η=0.05")]
G = (F.groupby(["alpha", "score_model", "condition", "famille", "reglage"])
       [["viol_reel", "viol_hasard", "sens_moy"]].mean().reset_index())

for alpha in sorted(G["alpha"].unique()):
    tv, ts = [], []
    for fam, knob, lab in PRIMARY:
        rv, rs = {"méthode": lab}, {"méthode": lab}
        for sm in MODELS:
            for c in COND_ORDER:
                x = G[close(G["alpha"], alpha) & (G["famille"] == fam) & (G["reglage"] == knob)
                      & (G["score_model"] == sm) & (G["condition"] == c)]
                col = f"{short(sm)} | {c}"
                rv[col] = (f"{100 * x['viol_reel'].iloc[0]:.0f} / {100 * x['viol_hasard'].iloc[0]:.0f}"
                           if len(x) else "-")
                rs[col] = f"{x['sens_moy'].iloc[0]:.3f}" if len(x) else "-"
        tv.append(rv); ts.append(rs)
    for rows, titre, fname in [(tv, "violations significatives brutes, % des hôpitaux : réel / hasard", "viol_brutes"),
                               (ts, f"sensibilité moyenne sur les hôpitaux (cible ≥ {1 - alpha:.2f})", "sens_moy")]:
        T = pd.DataFrame(rows)
        print(f"\n{'=' * 130}\n  T3 — α = {alpha:.2f} | {titre}\n{'=' * 130}")
        with pd.option_context("display.width", 300):
            print(T.to_string(index=False))
        T.to_csv(os.path.join(MASTER_DIR, f"T3_alpha{alpha:g}_{fname}.csv"), index=False)

print(f"\n✓ H3' et T3 enregistrés dans {MASTER_DIR}")
