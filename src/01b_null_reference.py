# ================================================================
# 01b_null_reference: random-reassignment control. Run immediately after 01_preprocessing (same session).
# Question: do violations under pooled calibration reflect genuine differences between hospitals,
#           or only chance (few deaths per hospital)?
# Method: holdout patients are shuffled between hospitals (sizes unchanged), which removes any
#         genuine difference; diagnostic B is then repeated exactly.
# ================================================================
from scipy.stats import spearmanr

N_NULL = 5
rs_null = np.random.RandomState(7)
cids   = list(clients.keys())
P_cat  = np.concatenate([P_hold[c] for c in cids])
Y_cat  = np.concatenate([clients[c]["y_test"] for c in cids])
cuts   = np.cumsum([len(clients[c]["y_test"]) for c in cids])[:-1]

null_viol = []
for rep in range(N_NULL):
    perm = rs_null.permutation(len(Y_cat))
    Pn = dict(zip(cids, np.split(P_cat[perm], cuts)))
    Yn = dict(zip(cids, np.split(Y_cat[perm], cuts)))
    sens = {c: [] for c in cids}
    for _ in range(DIAG_SPLITS):
        parts, calpos = {}, []
        for c in cids:
            n = len(Yn[c]); p = rs_null.permutation(n); h = n // 2
            parts[c] = (p[:h], p[h:])
            calpos.append(Pn[c][p[:h]][Yn[c][p[:h]] == 1])
        t = lower_threshold(np.concatenate(calpos))
        for c in cids:
            te = parts[c][1]; y = Yn[c][te]
            if (y == 1).sum() > 0:
                sens[c].append(float((Pn[c][te][y == 1] >= t).mean()))
    ms = np.array([np.mean(v) for v in sens.values() if len(v) > 0])
    null_viol.append(float((ms < thr).mean()))

obs = float((res["pool_sens"] < thr).mean())
print(f"Violations POOL observées                         : {obs:.0%}")
print(f"Violations POOL sous le hasard (hôpitaux mélangés) : {np.mean(null_viol):.0%} "
      f"(répétitions : {', '.join(f'{v:.0%}' for v in null_viol)})")
print(f"→ violations dues à de VRAIES différences entre hôpitaux ≈ {obs - np.mean(null_viol):+.0%}")

# Which hospitals lose coverage under pooled calibration?
mm = res.merge(summ[["cid", "mort", "n"]], on="cid")
mm["auc"] = mm["cid"].map(auc_h)
for col, lab in [("mort", "taux de mortalité"), ("auc", "AUROC de l'hôpital"), ("n", "taille (séjours)")]:
    r, p = spearmanr(mm["pool_sens"], mm[col], nan_policy="omit")
    print(f"  corrélation (Spearman) sensibilité POOL vs {lab:<20}: rho = {r:+.2f} (p = {p:.3g})")
