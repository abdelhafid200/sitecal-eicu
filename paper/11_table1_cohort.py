# ================================================================
# 11_table1_cohort: values of Table 1 (cohort by size third) and counts quoted in the text
# Reads: data_real_hospitals/clients/ (patient-level files, requires running the pipeline first)
# ================================================================
import os, json, pickle
import numpy as np
import pandas as pd

CL_DIR = os.path.join("data_real_hospitals", "clients")
with open(os.path.join(CL_DIR, "metadata.json")) as fh:
    META = json.load(fh)
rows = []
for cid in range(META["n_clients"]):
    with open(os.path.join(CL_DIR, f"client_{cid:02d}.pkl"), "rb") as fh:
        c = pickle.load(fh)
    y_all = np.concatenate([np.asarray(v).ravel() for k, v in c.items() if k.startswith("y_")]).astype(int)
    rows.append({"stays": len(y_all), "deaths": int(y_all.sum()),
                 "deaths_meta": int(c["n_deaths_total"]),
                 "deaths_holdout": int(np.asarray(c["y_test"]).sum()),
                 "keys_y": ",".join(k for k in c if k.startswith("y_"))})
H = pd.DataFrame(rows)
print("Clés des labels lues :", H["keys_y"].iloc[0])
if not (H["deaths"] == H["deaths_meta"]).all():
    print("⚠ Le total des décès recalculé ne correspond pas à n_deaths_total : vérifiez les fichiers clients.")

# Size thirds defined as in src/03_threshold_study.py (rank of the total number of deaths)
rank = H["deaths_meta"].rank(method="first")
H["taille"] = pd.qcut(rank, 3, labels=["Small", "Medium", "Large"])
H["mort"] = 100 * H["deaths"] / H["stays"]
H["cal_lt9"] = (H["deaths_holdout"] // 2) < 9          # expected calibration half has < 9 deaths

def med(s, fmt="{:.0f}"):
    return f"{fmt.format(s.median())} [{fmt.format(s.quantile(0.25))}–{fmt.format(s.quantile(0.75))}]"

out = []
for name, g in list(H.groupby("taille", observed=True)) + [("All", H)]:
    out.append({"group": name, "Hospitals, n": len(g),
                "Deaths per hospital, range": f"{g['deaths_meta'].min()}–{g['deaths_meta'].max()}",
                "Stays, total": f"{int(g['stays'].sum()):,}",
                "Stays per hospital": med(g["stays"]),
                "Deaths per hospital": med(g["deaths"]),
                "Hospital mortality, %": med(g["mort"], "{:.1f}"),
                "Held-out deaths per hospital": med(g["deaths_holdout"]),
                "Fewer than 9 calibration deaths, %": f"{100 * g['cal_lt9'].mean():.0f}"})
T1 = pd.DataFrame(out).set_index("group").T
print(T1.to_string())
print(f"\nCohorte : {H['stays'].sum():,} séjours, {H['deaths'].sum():,} décès, "
      f"mortalité {100 * H['deaths'].sum() / H['stays'].sum():.2f} %")
