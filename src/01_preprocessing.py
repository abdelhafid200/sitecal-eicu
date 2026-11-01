# ================================================================
# 01_preprocessing: cohort, hospital selection, optional semi-synthetic outcomes, and data splits
# Sites are actual eICU hospitals (hospitalid); the default outcome is observed in-hospital mortality.
# On real outcomes there are no synthetic labels, artificial groups or subsampling:
# small hospitals are small by nature.
# Outputs: client_XX.pkl and metadata.json (patient-level, never published),
# plus a calibration feasibility diagnostic (logistic regression, no federated learning).
# ================================================================
import os, json, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
assert "BASE_DIR" in globals(), "Lancez d'abord la cellule 00_Config"
PATIENT_PATH  = os.path.join(DATA_PATH, "patient.csv.gz")
HOSPITAL_PATH = os.path.join(DATA_PATH, "hospital.csv.gz")
APS_PATH      = os.path.join(DATA_PATH, "apacheApsVar.csv.gz")
APACHE_PATH   = os.path.join(DATA_PATH, "apachePatientResult.csv.gz")
# Semi-synthetic concept shift setting (used only when LABEL_MODE == "rotation", set in 00_config)
N_CONCEPT_GROUPS   = 8

PREP_SEED  = 42      # split seed (fixed: same patients in each split regardless of the model seed)
MIN_STAYS  = 200          # minimum stays to retain a hospital
MIN_DEATHS = 10           # minimum deaths to retain a hospital
TRAIN_SIZE, VAL_SIZE, HOLD_SIZE = 0.50, 0.10, 0.40
# train:   model training
# val:     model selection (best validation loss)
# holdout: calibration and test, randomly re-split several times downstream
#          (stored as X_test / y_test)

APS_FEATURES = ["pao2", "fio2", "creatinine", "bun", "wbc", "temperature",
                "meanbp", "eyes", "motor", "verbal", "vent", "dialysis"]
FEATURES     = ["age"] + APS_FEATURES
MISSING_FLAG = ["pao2", "fio2"]          # blood gases are often absent, so each gets a missingness indicator
FEATURE_COLS = FEATURES + [f"{c}_missing" for c in MISSING_FLAG]
INPUT_DIM    = len(FEATURE_COLS)

# Feasibility diagnostic
DIAG_ALPHA  = 0.10        # target: sensitivity ≥ 90%
DIAG_TOL    = 0.05        # a hospital is in violation if its mean sensitivity is < 0.85
DIAG_SPLITS = 100         # random calibration / test re-splits of the holdout

rng_global = np.random.default_rng(PREP_SEED)

# ============================================================
# 1. DATA LOADING
# ============================================================
print("=" * 70); print("1 - CHARGEMENT"); print("=" * 70)
patient  = pd.read_csv(PATIENT_PATH, usecols=["patientunitstayid", "uniquepid", "hospitalid",
                                               "age", "unitvisitnumber", "unittype"])
hospital = pd.read_csv(HOSPITAL_PATH, usecols=["hospitalid", "region", "numbedscategory", "teachingstatus"])
aps      = pd.read_csv(APS_PATH, usecols=["patientunitstayid"] + APS_FEATURES)
apache   = pd.read_csv(APACHE_PATH, usecols=["patientunitstayid", "apacheversion", "actualhospitalmortality"])

# apachePatientResult often has 2 rows per stay (APACHE IV and IVa): keep one
apache["apacheversion"] = apache["apacheversion"].astype(str).str.strip()
if (apache["apacheversion"] == "IVa").any():
    apache = apache[apache["apacheversion"] == "IVa"]
n_dup = apache["patientunitstayid"].duplicated().sum()
apache = apache.drop_duplicates("patientunitstayid")
print(f"  apachePatientResult : {len(apache):,} séjours (doublons restants supprimés : {n_dup})")

# ============================================================
# 2. COHORT
# ============================================================
print("\n" + "=" * 70); print("2 - COHORTE"); print("=" * 70)
patient["age"] = pd.to_numeric(patient["age"].astype(str).str.strip().replace({"> 89": "90"}), errors="coerce")
n0 = len(patient)
patient = patient[patient["unitvisitnumber"] == 1]                  # first ICU stay
patient = patient.drop_duplicates("uniquepid", keep="first")        # one stay per patient (prevents leakage)
patient = patient[patient["age"] >= 18]                             # adults
print(f"  patients : {n0:,} séjours → {len(patient):,} (1er séjour, 1 par patient, ≥ 18 ans)")

df = patient.merge(aps, on="patientunitstayid", how="inner").merge(apache, on="patientunitstayid", how="inner")
df["mortality"] = (df["actualhospitalmortality"].astype(str).str.strip().str.upper()
                   .map({"ALIVE": 0, "EXPIRED": 1}))
n_bad = df["mortality"].isna().sum()
df = df.dropna(subset=["mortality"]).copy()
df["mortality"] = df["mortality"].astype(int)
print(f"  après fusion : {len(df):,} séjours (label inconnu retiré : {n_bad}) | mortalité = {df['mortality'].mean():.2%}")

# In apacheApsVar, -1 encodes a missing value: report its share, then replace it with NaN
print("\n  Part de valeurs -1 (manquantes) par variable :")
for c in APS_FEATURES:
    frac = float((df[c] == -1).mean())
    print(f"    {c:<12} {frac:6.1%}")
    df.loc[df[c] == -1, c] = np.nan
for c in MISSING_FLAG:
    df[f"{c}_missing"] = df[c].isna().astype(float)

# ============================================================
# 3. HOSPITAL SELECTION
# ============================================================
print("\n" + "=" * 70); print("3 - HÔPITAUX"); print("=" * 70)
df["unittype"] = df["unittype"].fillna("Unknown").astype(str)
df["site"] = (df["hospitalid"].astype(str) if SITE_MODE == "hospital"
              else df["hospitalid"].astype(str) + "_" + df["unittype"])
stats = df.groupby("site")["mortality"].agg(n="size", deaths="sum")
keep  = stats[(stats["n"] >= MIN_STAYS) & (stats["deaths"] >= MIN_DEATHS)].index
print(f"  sites ({SITE_MODE}) : {len(stats)} → {len(keep)} gardés (≥ {MIN_STAYS} séjours et ≥ {MIN_DEATHS} décès)")
df = df[df["site"].isin(keep)].merge(hospital, on="hospitalid", how="left")
df["region"] = df["region"].fillna("Unknown").astype(str)
if SITE_MODE == "hospital_unit":
    print("  sites par type d'unité :", df.drop_duplicates("site")["unittype"].value_counts().to_dict())

# ============================================================
# 3b. (OPTIONAL) SEMI-SYNTHETIC OUTCOMES: concept shift by rotation (constant mortality and discrimination)
# LABEL_MODE = "real":     observed mortality, unchanged.
# LABEL_MODE = "rotation": each hospital is randomly assigned to a group (GROUP_SEED). Outcomes are drawn from
#   logit = s·[cos(θ)·z0 + sin(θ)·h_g] + c_g,   θ = SHIFT_A·π/2
#   z0:  standardized score of a logistic regression fitted on observed mortality (shared concept)
#   h_g: group-specific direction, decorrelated from z0 and then standardized (specific concept)
#   s:   standard deviation of the reference score, giving the same discrimination for any SHIFT_A
#   c_g: constant that sets each group's mortality to the observed overall mortality
# SHIFT_A = 0: same concept everywhere; SHIFT_A = 1: concept specific to each group.
# The retained hospitals are those of the real data (selection happens before this step).
# ============================================================
if LABEL_MODE == "rotation":
    rng_c = np.random.default_rng(1000 + GROUP_SEED)
    sites = np.array(sorted(df["site"].unique()))
    grp   = dict(zip(sites, rng_c.integers(0, N_CONCEPT_GROUPS, size=len(sites))))
    df["concept_group"] = df["site"].map(grp).astype(int)
    Xc  = df[FEATURES].astype(float).values
    Xc  = np.where(np.isnan(Xc), np.nanmedian(Xc, axis=0), Xc)
    Xc  = (Xc - Xc.mean(axis=0)) / (Xc.std(axis=0) + 1e-8)
    y_old = df["mortality"].values.copy()
    ref   = LogisticRegression(max_iter=2000).fit(Xc, y_old)
    f0    = Xc @ ref.coef_.ravel()
    s_ref = float(f0.std())
    z0    = (f0 - f0.mean()) / (s_ref + 1e-12)
    W_dir = rng_c.normal(0.0, 1.0, size=(N_CONCEPT_GROUPS, Xc.shape[1]))
    theta  = SHIFT_A * np.pi / 2
    target = float(y_old.mean())
    gid    = df["concept_group"].values
    logit  = np.zeros(len(df))
    for gg in range(N_CONCEPT_GROUPS):
        ix = np.where(gid == gg)[0]
        if len(ix) == 0:
            continue
        zg = z0[ix]
        hg = Xc[ix] @ W_dir[gg]
        zc = zg - zg.mean()
        hg = hg - hg.mean()
        hg = hg - (hg @ zc) / (zc @ zc + 1e-12) * zc          # decorrelate from the shared concept
        hg = hg / (hg.std() + 1e-12)
        v  = s_ref * (np.cos(theta) * zg + np.sin(theta) * hg)
        lo, hi = -30.0, 30.0                                     # find c_g by bisection
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if np.mean(1.0 / (1.0 + np.exp(-(v + mid)))) < target:
                lo = mid
            else:
                hi = mid
        logit[ix] = v + 0.5 * (lo + hi)
    p_new = 1.0 / (1.0 + np.exp(-logit))
    df["mortality"] = (rng_c.random(len(df)) < p_new).astype(int)
    auc_ref = roc_auc_score(y_old, f0)
    auc_new = roc_auc_score(df["mortality"].values, logit)
    print(f"  SEMI-SYNTHÉTIQUE (rotation) a = {SHIFT_A} | groupes seed {GROUP_SEED} : mortalité {y_old.mean():.2%} → "
          f"{df['mortality'].mean():.2%} | AUROC du concept : réel {auc_ref:.3f} → semi-synthétique {auc_new:.3f} "
          f"| {N_CONCEPT_GROUPS} groupes d'hôpitaux")
elif LABEL_MODE == "casemix":
    # Pure case-mix control: the same risk function for all hospitals,
    # logit = s·z0 + c with a single constant c (rotation at a = 0 without group constants).
    # Only patient characteristics (the case mix) differ between hospitals.
    rng_c = np.random.default_rng(1000 + GROUP_SEED)
    df["concept_group"] = -1
    Xc  = df[FEATURES].astype(float).values
    Xc  = np.where(np.isnan(Xc), np.nanmedian(Xc, axis=0), Xc)
    Xc  = (Xc - Xc.mean(axis=0)) / (Xc.std(axis=0) + 1e-8)
    y_old = df["mortality"].values.copy()
    ref   = LogisticRegression(max_iter=2000).fit(Xc, y_old)
    f0    = Xc @ ref.coef_.ravel()
    v     = f0 - f0.mean()                                   # = s_ref · z0
    target = float(y_old.mean())
    lo, hi = -30.0, 30.0                                     # find the single constant c by bisection
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if np.mean(1.0 / (1.0 + np.exp(-(v + mid)))) < target:
            lo = mid
        else:
            hi = mid
    logit = v + 0.5 * (lo + hi)
    p_new = 1.0 / (1.0 + np.exp(-logit))
    df["mortality"] = (rng_c.random(len(df)) < p_new).astype(int)
    auc_ref = roc_auc_score(y_old, f0)
    auc_new = roc_auc_score(df["mortality"].values, logit)
    print(f"  SEMI-SYNTHÉTIQUE (case-mix pur) | tirage seed {GROUP_SEED} : mortalité {y_old.mean():.2%} → "
          f"{df['mortality'].mean():.2%} | AUROC du concept : réel {auc_ref:.3f} → semi-synthétique {auc_new:.3f}")
else:
    df["concept_group"] = -1

# ============================================================
# 4. PER-HOSPITAL SPLITS
# ============================================================
def safe_split(X, y, train_size, seed):
    try:
        return train_test_split(X, y, train_size=train_size, stratify=y, random_state=seed)
    except ValueError:
        return train_test_split(X, y, train_size=train_size, stratify=None, random_state=seed)

raw = {}
for cid, (site, sub) in enumerate(df.sort_values("site").groupby("site", sort=True)):
    hid = sub["hospitalid"].iloc[0]
    X, y = sub[FEATURE_COLS].values.astype(float), sub["mortality"].values.astype(int)
    X_tr, X_rest, y_tr, y_rest = safe_split(X, y, TRAIN_SIZE, PREP_SEED)
    X_va, X_ho, y_va, y_ho     = safe_split(X_rest, y_rest, VAL_SIZE / (VAL_SIZE + HOLD_SIZE), PREP_SEED)
    first = sub.iloc[0]
    raw[cid] = dict(hospitalid=int(hid), site=str(site), unittype=str(first["unittype"]), region=str(first["region"]),
                    concept_group=int(first["concept_group"]),
                    numbedscategory=str(first["numbedscategory"]), teachingstatus=str(first["teachingstatus"]),
                    X_train=X_tr, y_train=y_tr, X_val=X_va, y_val=y_va, X_test=X_ho, y_test=y_ho)

# Imputation: each hospital's training median (fallback: median over all training sets)
n_cont = len(FEATURES)                       # columns to impute and normalize (excludes missingness indicators)
global_median = np.nanmedian(np.vstack([r["X_train"][:, :n_cont] for r in raw.values()]), axis=0)
for r in raw.values():
    med = np.nanmedian(r["X_train"][:, :n_cont], axis=0)
    med = np.where(np.isnan(med), global_median, med)
    for key in ["X_train", "X_val", "X_test"]:
        Xk = r[key]
        for j in range(n_cont):
            Xk[np.isnan(Xk[:, j]), j] = med[j]

# Normalization shared by all hospitals (mean and SD of the union of training sets).
# In a federated deployment, this comes from aggregated sums and sums of squares, without sharing data.
# Per-hospital normalization would make inputs incomparable between hospitals.
all_tr   = np.vstack([r["X_train"][:, :n_cont] for r in raw.values()])
mu_glob  = all_tr.mean(axis=0)
sd_glob  = all_tr.std(axis=0); sd_glob[sd_glob < 1e-8] = 1.0
for r in raw.values():
    for key in ["X_train", "X_val", "X_test"]:
        r[key][:, :n_cont] = (r[key][:, :n_cont] - mu_glob) / sd_glob

# ============================================================
# 5. SAVE CLIENT FILES
# ============================================================
clients = {}
for cid, r in raw.items():
    c = {"client_id": cid, "hospitalid": r["hospitalid"], "site": r["site"], "unittype": r["unittype"],
         "concept_group": r["concept_group"],
         "phenotype": (f"concept_{r['concept_group']:02d}" if LABEL_MODE == "rotation"
                       else (r["unittype"] if SITE_MODE == "hospital_unit" else r["region"])),
         "numbedscategory": r["numbedscategory"], "teachingstatus": r["teachingstatus"],
         "X_train": r["X_train"], "y_train": r["y_train"],
         "X_val": r["X_val"], "y_val": r["y_val"],
         "X_test": r["X_test"], "y_test": r["y_test"],
         "n_train": len(r["y_train"]), "n_val": len(r["y_val"]), "n_test": len(r["y_test"]),
         "n_deaths_total": int(r["y_train"].sum() + r["y_val"].sum() + r["y_test"].sum()),
         "mortality_rate_train": float(r["y_train"].mean())}
    clients[cid] = c
    with open(os.path.join(CLIENT_DIR, f"client_{cid:02d}.pkl"), "wb") as f:
        pickle.dump(c, f, protocol=pickle.HIGHEST_PROTOCOL)

metadata = {
    "n_clients": len(clients), "input_dim": INPUT_DIM, "feature_cols": FEATURE_COLS,
    "target_col": "mortality (réelle, actualhospitalmortality)",
    "phenotypes": sorted({c["phenotype"] for c in clients.values()}),
    "phenotype_meaning": "région de l'hôpital (eICU hospital.region) — regroupement descriptif, PAS un oracle",
    "split": {"train": TRAIN_SIZE, "val": VAL_SIZE, "holdout_cal_test": HOLD_SIZE},
    "min_stays": MIN_STAYS, "min_deaths": MIN_DEATHS, "seed": PREP_SEED,
    "normalization": {"mean": mu_glob.tolist(), "std": sd_glob.tolist(), "columns": FEATURES},
    "client_summary": {cid: {"hospitalid": c["hospitalid"], "region": c["phenotype"],
                             "n_train": c["n_train"], "n_val": c["n_val"], "n_test": c["n_test"],
                             "deaths_total": c["n_deaths_total"],
                             "mortality_rate_train": round(c["mortality_rate_train"], 4)}
                       for cid, c in clients.items()},
}
with open(os.path.join(CLIENT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
print(f"\n  {len(clients)} hôpitaux sauvegardés dans '{CLIENT_DIR}/' ({INPUT_DIM} variables)")

# ============================================================
# 6. DIAGNOSTIC A: HOSPITAL SIZES AND HETEROGENEITY
# ============================================================
print("\n" + "=" * 70); print("6 - DIAGNOSTIC A : TAILLES ET HÉTÉROGÉNÉITÉ"); print("=" * 70)
summ = pd.DataFrame([{"cid": cid, "n": c["n_train"] + c["n_val"] + c["n_test"], "deaths": c["n_deaths_total"],
                      "mort": c["n_deaths_total"] / (c["n_train"] + c["n_val"] + c["n_test"]),
                      "hold_pos": int(c["y_test"].sum()), "region": c["phenotype"]}
                     for cid, c in clients.items()])
q = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
print("  Quantiles (min, 10 %, 25 %, médiane, 75 %, 90 %, max)")
print(f"    séjours par hôpital    : {np.quantile(summ['n'], q).astype(int).tolist()}")
print(f"    décès par hôpital      : {np.quantile(summ['deaths'], q).astype(int).tolist()}")
print(f"    mortalité par hôpital  : {np.round(np.quantile(summ['mort'], q), 3).tolist()}")
print(f"    décès dans le holdout  : {np.quantile(summ['hold_pos'], q).astype(int).tolist()}")
print(f"  Hôpitaux dont la moitié calibration du holdout a < 9 décès "
      f"(calibration locale impossible à 90 %) : {(summ['hold_pos'] // 2 < 9).mean():.0%}")
print("  Hôpitaux par région :", summ["region"].value_counts().to_dict())

# ============================================================
# 7. DIAGNOSTIC B: CALIBRATION FEASIBILITY (no federated learning)
# Reference model: logistic regression on the union of training sets (diagnostic only).
# For each hospital, DIAG_SPLITS random 50/50 calibration / test splits of the holdout:
#   - local: threshold from the hospital's own calibration deaths only
#   - pool:  one threshold from the calibration deaths of all hospitals
# ============================================================
print("\n" + "=" * 70); print("7 - DIAGNOSTIC B : CALIBRATION LOCALE vs POOLÉE"); print("=" * 70)

def lower_threshold(p_cal, alpha=DIAG_ALPHA):
    """Conformal threshold t such that P(p_test < t) ≤ alpha for a death exchangeable with the calibration deaths."""
    p = np.sort(np.asarray(p_cal, float)); n = len(p)
    m = int(np.floor(alpha * (n + 1)))
    return -np.inf if m == 0 else float(p[m - 1])

X_all = np.vstack([c["X_train"] for c in clients.values()])
y_all = np.concatenate([c["y_train"] for c in clients.values()])
lr = LogisticRegression(max_iter=2000).fit(X_all, y_all)
P_hold = {cid: lr.predict_proba(c["X_test"])[:, 1] for cid, c in clients.items()}
auc_h  = {cid: (roc_auc_score(c["y_test"], P_hold[cid]) if len(np.unique(c["y_test"])) > 1 else np.nan)
          for cid, c in clients.items()}
print(f"  AUROC de la régression logistique poolée par hôpital : médiane={np.nanmedian(list(auc_h.values())):.3f} "
      f"| 10 %={np.nanquantile(list(auc_h.values()), 0.1):.3f} | 90 %={np.nanquantile(list(auc_h.values()), 0.9):.3f}")

rs  = np.random.RandomState(PREP_SEED)
acc = {cid: {"local_sens": [], "local_alert": [], "local_inf": [], "pool_sens": [], "pool_alert": []} for cid in clients}
for _ in range(DIAG_SPLITS):
    cal_pos, parts = [], {}
    for cid, c in clients.items():
        n = len(c["y_test"]); perm = rs.permutation(n); h = n // 2
        cal, te = perm[:h], perm[h:]
        parts[cid] = (cal, te)
        cal_pos.append(P_hold[cid][cal][c["y_test"][cal] == 1])
    t_pool = lower_threshold(np.concatenate(cal_pos))
    for cid, c in clients.items():
        cal, te = parts[cid]
        y_te, p_te = c["y_test"][te], P_hold[cid][te]
        if (y_te == 1).sum() == 0:
            continue
        t_loc = lower_threshold(P_hold[cid][cal][c["y_test"][cal] == 1])
        for name, t in [("local", t_loc), ("pool", t_pool)]:
            alert = p_te >= t
            acc[cid][f"{name}_sens"].append(alert[y_te == 1].mean())
            acc[cid][f"{name}_alert"].append(alert.mean())
        acc[cid]["local_inf"].append(np.isinf(t_loc))

res = pd.DataFrame([{"cid": cid, "deaths": summ.loc[summ.cid == cid, "deaths"].item(),
                     **{k: float(np.mean(v)) for k, v in a.items() if len(v) > 0}} for cid, a in acc.items()])
res["taille"] = pd.qcut(res["deaths"], 3, labels=["petits", "moyens", "grands"])
thr = 1 - DIAG_ALPHA - DIAG_TOL
print(f"\n  Cible : sensibilité ≥ {1 - DIAG_ALPHA:.2f} | hôpital en violation si sensibilité moyenne < {thr:.2f}")
print(f"  {'':<8} | {'POOL viol.':>10} | {'POOL sens':>9} | {'POOL alertes':>12} | "
      f"{'LOCAL viol.':>11} | {'LOCAL alertes':>13} | {'LOCAL seuil=-inf':>16}")
for grp, g in list(res.groupby("taille")) + [("TOUS", res)]:
    print(f"  {str(grp):<8} | {(g['pool_sens'] < thr).mean():>10.0%} | {g['pool_sens'].mean():>9.3f} | "
          f"{g['pool_alert'].mean():>12.3f} | {(g['local_sens'] < thr).mean():>11.0%} | "
          f"{g['local_alert'].mean():>13.3f} | {g['local_inf'].mean():>16.0%}")
res.to_csv(os.path.join(BASE_DIR, "diagnostic_calibration.csv"), index=False)
print(f"\n  Détail par hôpital : {os.path.join(BASE_DIR, 'diagnostic_calibration.csv')}")
print("\nDone.")
