# ================================================================
# 03_threshold_study: trade-off between per-hospital guarantee and alert rate, real data vs random reassignment
# Each method has a conservativeness setting; all settings are swept rather than comparing a single one:
#   shrink_pool:      n0 pseudo-deaths borrowed from the pool (Shahid, 2026)
#   donors_conformal: the M most similar donor hospitals
#   pool_bayes:       empirical prior from all other hospitals, confidence 1 − η
#   donors_bayes:     empirical prior from M donor hospitals, confidence 1 − η
#   fixed references: local_conformal, pool_conformal
# For each setting: significant violations on the real data and under NULL_REPS random
# reassignments of patients to hospitals (excess = real − random), plus the alert rate.
# ================================================================
import os, json, pickle, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
assert "BASE_DIR" in globals(), "Lancez d'abord la cellule 00_Config"
OUT_DIR     = FRONTIER_DIR             # folders and seed are set in 00_config
SCORE_MODEL = globals().get("CFG_SCORE_MODEL", "personalized")   # "global" | "personalized" | "local_lr" (set by 07, 07c)
ALPHAS      = [0.10, 0.20]
N_SPLITS    = 50                       # calibration / test splits per dataset
CAL_FRAC    = 0.5
NULL_REPS   = globals().get("CFG_NULL_REPS", 2)                  # random reassignments (set by 07, 07c)
Z_SIG       = 1.645                    # significant violation: mean FNR > α with z > 1.645
NU_MIN, NU_MAX = 1.0, 200.0

SHRINK_N0S  = [2, 5, 10, 19, 50, 100, 300]
DONOR_MS    = [5, 10, 20, 40]
EB_M        = 10                       # number of donors for donors_bayes
ETAS        = [0.5, 0.3, 0.2, 0.1, 0.05]
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)

CONFIGS = (["local_conformal", "pool_conformal"]
           + [f"shrink_pool|n0={n}" for n in SHRINK_N0S]
           + [f"donors_conformal|M={m}" for m in DONOR_MS]
           + [f"pool_bayes|eta={e}" for e in ETAS]
           + [f"donors_bayes|eta={e}" for e in ETAS])

# ------------------------------------------------------------
# DATA AND MODELS
# ------------------------------------------------------------
with open(os.path.join(CLIENT_DIR, "metadata.json")) as f:
    META = json.load(f)
CL = {}
for cid in range(META["n_clients"]):
    with open(os.path.join(CLIENT_DIR, f"client_{cid:02d}.pkl"), "rb") as f:
        CL[cid] = pickle.load(f)
IDS = list(CL.keys())
XH0 = {c: CL[c]["X_test"] for c in IDS}
YH0 = {c: np.asarray(CL[c]["y_test"]).astype(int) for c in IDS}
deaths = {c: int(CL[c]["n_deaths_total"]) for c in IDS}
rank   = pd.Series(deaths).rank(method="first")
TAILLE = dict(zip(rank.index, pd.qcut(rank, 3, labels=["petits", "moyens", "grands"])))

class MLP(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        layers, cur = [], d
        for h in hidden:
            layers += [nn.Linear(cur, h), nn.ReLU()]; cur = h
        layers += [nn.Linear(cur, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)

@torch.no_grad()
def mlp_predict(model, X):
    model.eval()
    return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32, device=DEVICE))).cpu().numpy()

ck = torch.load(os.path.join(MODEL_DIR, f"models_seed{SEED}.pt"), map_location=DEVICE)
if SCORE_MODEL == "local_lr":
    with open(os.path.join(MODEL_DIR, f"local_lr_seed{SEED}.pkl"), "rb") as f:
        LLR = pickle.load(f)

def build_scores(XH):
    """S[(k, j)] = probabilities from the model of hospital k on the holdout of hospital j."""
    S = {}
    if SCORE_MODEL == "global":
        g = MLP(ck["input_dim"], ck["hidden"]).to(DEVICE); g.load_state_dict(ck["global"])
        P = {j: mlp_predict(g, XH[j]) for j in IDS}
        return {(k, j): P[j] for k in IDS for j in IDS}
    if SCORE_MODEL == "personalized":
        m = MLP(ck["input_dim"], ck["hidden"]).to(DEVICE)
        for k in IDS:
            m.load_state_dict(ck["personal"][k])
            for j in IDS:
                S[(k, j)] = mlp_predict(m, XH[j])
        return S
    if SCORE_MODEL == "local_lr":
        for k in IDS:
            for j in IDS:
                S[(k, j)] = LLR[k].predict_proba(XH[j])[:, 1]
        return S
    raise ValueError(SCORE_MODEL)

# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------
def conformal_threshold(pos, alpha):
    p = np.sort(np.asarray(pos, float)); n = len(p)
    m = int(np.floor(alpha * (n + 1)))
    return -np.inf if m == 0 else float(p[m - 1])

def last_ok_prefix(ok, grid):
    bad = np.where(~ok)[0]
    i = (bad[0] - 1) if len(bad) else len(grid) - 1
    return -np.inf if i < 0 else float(grid[i])

def misses(sorted_pos, grid):
    return np.searchsorted(sorted_pos, grid, side="left").astype(float)

def ks(a, b):
    a, b = np.sort(a), np.sort(b)
    v = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, v, side="right") / len(a)
                               - np.searchsorted(b, v, side="right") / len(b))))

def eb_prior(C, N):
    m = np.clip(C.sum(0) / N.sum(), 1e-4, 1 - 1e-4)
    if len(N) > 1:
        th   = C / N[:, None]
        tau2 = np.maximum(th.var(axis=0, ddof=1) - m * (1 - m) * np.mean(1.0 / N), 1e-12)
        nu   = m * (1 - m) / tau2 - 1
    else:
        nu = np.full_like(m, NU_MIN)
    return m, np.clip(nu, NU_MIN, min(NU_MAX, float(N.sum())))

def posterior_prob(c_k, n_k, m, nu, alpha):
    a = nu * m + c_k; b = nu * (1 - m) + (n_k - c_k)
    return beta_dist.cdf(alpha, np.maximum(a, 1e-6), np.maximum(b, 1e-6))

# ------------------------------------------------------------
# FULL STUDY ON ONE DATASET (real or randomly reassigned)
# ------------------------------------------------------------
def run_study(XH, YH, label):
    S = build_scores(XH)
    rec = []
    t0 = time.time()
    for r in range(N_SPLITS):
        rs = np.random.RandomState(10_000 + r)
        cal, te = {}, {}
        for j in IDS:
            p = rs.permutation(len(YH[j])); h = int(round(CAL_FRAC * len(p)))
            cal[j], te[j] = p[:h], p[h:]
        calpos = {j: cal[j][YH[j][cal[j]] == 1] for j in IDS}
        for k in IDS:
            y_te = YH[k][te[k]]
            if (y_te == 1).sum() == 0:
                continue
            s_te   = S[(k, k)][te[k]]
            others = [j for j in IDS if j != k]
            pos    = {j: np.sort(S[(k, j)][calpos[j]]) for j in IDS}
            grid   = np.concatenate([[-np.inf], np.unique(np.concatenate([pos[j] for j in IDS]))])
            n_k, c_k = len(pos[k]), misses(pos[k], grid)
            all_k  = S[(k, k)][cal[k]]
            order  = [others[i] for i in np.argsort([ks(all_k, S[(k, j)][cal[j]]) for j in others])]
            with_pos = [j for j in others if len(pos[j]) > 0]
            N_all = np.array([len(pos[j]) for j in with_pos], float)
            C_all = np.stack([misses(pos[j], grid) for j in with_pos])
            R_pool = C_all.sum(0) / max(N_all.sum(), 1.0)
            don_eb = [j for j in order[:EB_M] if len(pos[j]) > 0]
            m_p, nu_p = eb_prior(C_all, N_all)
            if don_eb:
                m_d, nu_d = eb_prior(np.stack([misses(pos[j], grid) for j in don_eb]),
                                     np.array([len(pos[j]) for j in don_eb], float))
            for alpha in ALPHAS:
                th = {"local_conformal": conformal_threshold(pos[k], alpha),
                      "pool_conformal":  conformal_threshold(np.concatenate([pos[j] for j in IDS]), alpha)}
                for n0 in SHRINK_N0S:
                    ne = n_k + n0
                    th[f"shrink_pool|n0={n0}"] = last_ok_prefix(
                        ((c_k + n0 * R_pool) + 1) / (ne + 1) <= alpha, grid)
                for M in DONOR_MS:
                    th[f"donors_conformal|M={M}"] = conformal_threshold(
                        np.concatenate([pos[k]] + [pos[j] for j in order[:M]]), alpha)
                prob_p = posterior_prob(c_k, n_k, m_p, nu_p, alpha)
                prob_d = (posterior_prob(c_k, n_k, m_d, nu_d, alpha) if don_eb
                          else posterior_prob(c_k, n_k, np.full(len(grid), 0.5), np.full(len(grid), 2.0), alpha))
                for e in ETAS:
                    th[f"pool_bayes|eta={e}"]   = last_ok_prefix(prob_p >= 1 - e, grid)
                    th[f"donors_bayes|eta={e}"] = last_ok_prefix(prob_d >= 1 - e, grid)
                for cfg in CONFIGS:
                    alert = s_te >= th[cfg]
                    rec.append((alpha, cfg, k, float(alert[y_te == 1].mean()), float(alert.mean()),
                                float(np.isinf(th[cfg]))))
        if (r + 1) % 10 == 0:
            print(f"    [{label}] découpage {r + 1}/{N_SPLITS} | {time.time() - t0:.0f}s")
    R = pd.DataFrame(rec, columns=["alpha", "config", "cid", "sens", "alert", "thr_inf"])
    H = R.groupby(["alpha", "config", "cid"]).mean().reset_index()
    H["n_pos_hold"] = H["cid"].map({c: int(YH[c].sum()) for c in IDS})
    H["z"] = ((1 - H["sens"]) - H["alpha"]) / np.sqrt(H["alpha"] * (1 - H["alpha"]) / H["n_pos_hold"].clip(lower=1))
    H["viol_sig"] = H["z"] > Z_SIG
    H["taille"] = H["cid"].map(TAILLE)
    H["dataset"] = label
    return H

def permuted(rep):
    X_cat = np.vstack([XH0[c] for c in IDS]); Y_cat = np.concatenate([YH0[c] for c in IDS])
    perm  = np.random.RandomState(999 + rep).permutation(len(Y_cat))
    cuts  = np.cumsum([len(YH0[c]) for c in IDS])[:-1]
    return dict(zip(IDS, np.split(X_cat[perm], cuts))), dict(zip(IDS, np.split(Y_cat[perm], cuts)))

# ------------------------------------------------------------
# EXECUTION: real data + NULL_REPS random-reassignment references
# ------------------------------------------------------------
TAG = f"{SCORE_MODEL}_seed{SEED}"
print(f"✓ {len(IDS)} hôpitaux | modèle '{SCORE_MODEL}' | {len(CONFIGS)} réglages × {len(ALPHAS)} α | "
      f"{N_SPLITS} découpages | {NULL_REPS} références hasard")
parts = [run_study(XH0, YH0, "reel")]
for rep in range(NULL_REPS):
    Xn, Yn = permuted(rep)
    parts.append(run_study(Xn, Yn, f"hasard{rep}"))
ALL = pd.concat(parts, ignore_index=True)
ALL.to_csv(os.path.join(OUT_DIR, f"frontier_per_hospital_{TAG}.csv"), index=False)

# ------------------------------------------------------------
# SUMMARY: excess violations (real − random) and alert rates, per setting
# ------------------------------------------------------------
rows = []
for (alpha, cfg), g in ALL.groupby(["alpha", "config"], sort=False):
    real, null = g[g["dataset"] == "reel"], g[g["dataset"] != "reel"]
    rs_, ns_ = real[real["taille"] == "petits"], null[null["taille"] == "petits"]
    fam, _, knob = cfg.partition("|")
    rows.append({"alpha": alpha, "famille": fam, "reglage": knob or "-",
                 "viol_reel": real["viol_sig"].mean(), "viol_hasard": null["viol_sig"].mean(),
                 "exces": real["viol_sig"].mean() - null["viol_sig"].mean(),
                 "exces_petits": rs_["viol_sig"].mean() - ns_["viol_sig"].mean(),
                 "alertes": real["alert"].mean(), "alertes_petits": rs_["alert"].mean(),
                 "sens_moy": real["sens"].mean(), "seuil_inf": real["thr_inf"].mean()})
SUM = pd.DataFrame(rows)
FAM_ORDER = ["local_conformal", "pool_conformal", "shrink_pool", "donors_conformal", "pool_bayes", "donors_bayes"]
SUM["_f"] = SUM["famille"].map({f: i for i, f in enumerate(FAM_ORDER)})
SUM["_k"] = pd.to_numeric(SUM["reglage"].str.split("=").str[-1], errors="coerce")
SUM = SUM.sort_values(["alpha", "_f", "_k"]).drop(columns=["_f", "_k"]).reset_index(drop=True)
SUM.to_csv(os.path.join(OUT_DIR, f"frontier_summary_{TAG}.csv"), index=False)
for alpha in ALPHAS:
    s = SUM[SUM["alpha"] == alpha]
    print(f"\n{'=' * 112}\n  α = {alpha:.2f} (sensibilité ≥ {1 - alpha:.2f}) | modèle = {SCORE_MODEL} | "
          f"violations significatives : réel, hasard (moyenne de {NULL_REPS}), excès\n{'=' * 112}")
    print(f"  {'famille':<17} {'réglage':<9} | {'réel':>5} | {'hasard':>6} | {'EXCÈS':>6} | {'excès petits':>12} | "
          f"{'alertes':>7} | {'alertes petits':>14} | {'sens. moy':>9}")
    for _, x in s.iterrows():
        print(f"  {x['famille']:<17} {x['reglage']:<9} | {x['viol_reel']:>5.0%} | {x['viol_hasard']:>6.0%} | "
              f"{x['exces']:>+6.0%} | {x['exces_petits']:>+12.0%} | {x['alertes']:>7.3f} | "
              f"{x['alertes_petits']:>14.3f} | {x['sens_moy']:>9.3f}")

# ------------------------------------------------------------
# FIGURE: excess violations vs alert rate frontier (one panel per α)
# Colours: reference categorical palette, 3 families in colour (the limit for a scatter plot),
# the others in grey with direct labels; distinct marker shapes so colour is not the only cue.
# ------------------------------------------------------------
STYLE = {"donors_bayes":     ("#2a78d6", "o", "-",  "notre méthode (donneurs + bayes)"),
         "shrink_pool":      ("#eb6834", "s", "-",  "shrink vers le pool (Shahid)"),
         "donors_conformal": ("#1baf7a", "^", "-",  "donneurs, conformal"),
         "pool_bayes":       ("#8a8a85", "D", "--", "pool + bayes (ablation)")}
INK, MUTED = "#1f1f1e", "#6b6b66"
fig, axes = plt.subplots(1, len(ALPHAS), figsize=(6.2 * len(ALPHAS), 4.6), squeeze=False)
for ax, alpha in zip(axes[0], ALPHAS):
    s = SUM[SUM["alpha"] == alpha]
    for fam, (col, mk, ls, lab) in STYLE.items():
        f = s[s["famille"] == fam].sort_values("alertes")
        ax.plot(f["alertes"], 100 * f["exces"], ls, color=col, lw=2, marker=mk, ms=7,
                markeredgecolor="white", markeredgewidth=1.2, label=lab, zorder=3)
    for fam, txt in [("local_conformal", "local"), ("pool_conformal", "pool")]:
        f = s[s["famille"] == fam]
        ax.scatter(f["alertes"], 100 * f["exces"], s=60, color=MUTED, marker="X", zorder=4)
        for _, x in f.iterrows():
            ax.annotate(txt, (x["alertes"], 100 * x["exces"]), xytext=(6, 4), textcoords="offset points",
                        color=INK, fontsize=9)
    ax.axhline(0, color=MUTED, lw=1, ls=":")
    ax.set_xlabel("taux d'alertes (part des patients signalés)", color=INK)
    ax.set_ylabel("excès de violations sur le hasard (points de %)", color=INK)
    ax.set_title(f"sensibilité cible ≥ {1 - alpha:.0%}", color=INK, fontsize=11, loc="left")
    ax.grid(True, color="#e6e5e0", lw=0.8); ax.set_axisbelow(True)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]: ax.spines[sp].set_color("#c9c8c2")
    ax.tick_params(colors=MUTED)
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(STYLE), frameon=False, fontsize=9)
fig.suptitle(f"Garantie par hôpital contre alertes — modèle {SCORE_MODEL}, seed {SEED} "
             f"(en bas à gauche = mieux)", color=INK, fontsize=12, x=0.01, ha="left")
fig.tight_layout(rect=[0, 0.07, 1, 0.97])
fig_path = os.path.join(OUT_DIR, f"frontier_{TAG}.png")
fig.savefig(fig_path, dpi=160)
plt.show()
print(f"\n✓ Tableaux et figure : {OUT_DIR}")
