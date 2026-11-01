# ================================================================
# 07_run_main_plan: all conditions × seeds × models, following the pre-specified analysis plan
# Requires: 00_config already run; src/01, 01b, 02 and 03 present (run from the repository root).
# Resumable: completed runs are skipped, so this script can be interrupted and restarted.
# ================================================================
import os, json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

assert "make_base_dir" in globals(), "Lancez d'abord la cellule 00_Config"
CELL_FILES = {"01": "src/01_preprocessing.py", "01b": "src/01b_null_reference.py",
              "02": "src/02_score_models.py", "03b": "src/03_threshold_study.py"}
for _f in CELL_FILES.values():
    assert os.path.exists(_f), f"Fichier manquant : {_f} — lancez depuis la racine du dépôt"

QUICK         = False          # True: preview (seed 42, 1 group assignment), about 2–3 h; False: full plan, about 13 h
SHIFT_VALUES  = [0.0, 0.5, 1.0]
GROUP_SEEDS   = [1] if QUICK else [1, 2]
MODEL_SEEDS   = [42] if QUICK else [42, 123, 2025]
SCORE_MODELS  = ["personalized", "global"]
CFG_NULL_REPS = 3
FINAL_SUBDIR  = "frontier_final"      # kept separate from exploratory runs
MASTER_DIR    = "resultats_finaux"
os.makedirs(MASTER_DIR, exist_ok=True)

CONDITIONS = [("real", 0.0, 0)] + [("rotation", a, g) for a in SHIFT_VALUES for g in GROUP_SEEDS]
n_runs = len(CONDITIONS) * len(MODEL_SEEDS) * len(SCORE_MODELS)
print(f"Plan : {len(CONDITIONS)} conditions × {len(MODEL_SEEDS)} seeds × {len(SCORE_MODELS)} modèles = {n_runs} runs de 03b "
      f"(≈ {n_runs * (1 + CFG_NULL_REPS) * 4.5 / 60:.1f} h, hors runs déjà terminés)")

def run_cell(key):
    path = CELL_FILES[key]
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    exec(compile(src, path, "exec"), globals())

frontier_rows, cond_rows = [], []
t_start = time.time()
for LABEL_MODE, SHIFT_A, GROUP_SEED in CONDITIONS:
    COND         = "réel" if LABEL_MODE == "real" else f"rot a={SHIFT_A:g}"
    SEMI_SYNTH_A = SHIFT_A if LABEL_MODE == "rotation" else 0.0
    BASE_DIR     = make_base_dir(LABEL_MODE, SHIFT_A, GROUP_SEED)
    CLIENT_DIR   = os.path.join(BASE_DIR, "clients")
    MODEL_DIR    = os.path.join(BASE_DIR, "models")
    FRONTIER_DIR = os.path.join(BASE_DIR, FINAL_SUBDIR)
    for _d in (CLIENT_DIR, MODEL_DIR, FRONTIER_DIR):
        os.makedirs(_d, exist_ok=True)
    print(f"\n{'#' * 100}\n#  CONDITION {COND} | groupes seed {GROUP_SEED} | {BASE_DIR} | "
          f"{(time.time() - t_start) / 60:.0f} min écoulées\n{'#' * 100}")

    # --- 01 + 01b: once per condition ---
    diag_path = os.path.join(BASE_DIR, "diag_01b.json")
    if not (os.path.exists(diag_path) and os.path.exists(os.path.join(CLIENT_DIR, "metadata.json"))):
        run_cell("01")
        run_cell("01b")
        diag = {"mortalite": float(df["mortality"].mean()),
                "viol_pool_obs": float(obs), "viol_pool_hasard": float(np.mean(null_viol)),
                "exces_01b": float(obs - np.mean(null_viol)), "n_hopitaux": int(len(clients))}
        with open(diag_path, "w") as fh:
            json.dump(diag, fh, indent=2)
    with open(diag_path) as fh:
        diag = json.load(fh)

    for SEED in MODEL_SEEDS:
        # --- 02: once per condition and seed ---
        if not os.path.exists(os.path.join(MODEL_DIR, f"models_seed{SEED}.pt")):
            run_cell("02")
        auc_df = pd.read_csv(os.path.join(MODEL_DIR, f"auroc_holdout_seed{SEED}.csv"))
        cond_rows.append({"condition": COND, "label_mode": LABEL_MODE, "shift_a": SHIFT_A,
                          "group_seed": GROUP_SEED, "model_seed": SEED, **diag,
                          "auroc_global": float(auc_df["global"].median()),
                          "auroc_personalized": float(auc_df["personalized"].median()),
                          "auroc_local": float(auc_df["local_lr"].median())})
        # --- 03 (key "03b"): once per condition, seed and score model ---
        for CFG_SCORE_MODEL in SCORE_MODELS:
            summ_path = os.path.join(FRONTIER_DIR, f"frontier_summary_{CFG_SCORE_MODEL}_seed{SEED}.csv")
            if not os.path.exists(summ_path):
                run_cell("03b")
                plt.close("all")
            run_summary = pd.read_csv(summ_path)
            run_summary["condition"]   = COND
            run_summary["label_mode"]  = LABEL_MODE
            run_summary["shift_a"]     = SHIFT_A
            run_summary["group_seed"]  = GROUP_SEED
            run_summary["model_seed"]  = SEED
            run_summary["score_model"] = CFG_SCORE_MODEL
            frontier_rows.append(run_summary)
        # save after each seed so that partial results remain usable
        pd.concat(frontier_rows, ignore_index=True).to_csv(os.path.join(MASTER_DIR, "master_frontier.csv"), index=False)
        pd.DataFrame(cond_rows).to_csv(os.path.join(MASTER_DIR, "master_conditions.csv"), index=False)

print(f"\n✅ Plan terminé en {(time.time() - t_start) / 3600:.1f} h | résultats : {MASTER_DIR}")
