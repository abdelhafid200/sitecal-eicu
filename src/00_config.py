# ================================================================
# 00_config: experiment settings. Run first, and again after every kernel restart.
# Edit these values for a single manual run. The final plan (07, 07c) sets them automatically.
# ================================================================
import os

DATA_PATH   = "dataset/"   # eICU tables: patient, hospital, apacheApsVar, apachePatientResult (.csv.gz)
SITE_MODE   = "hospital"      # "hospital" (main study) | "hospital_unit" (robustness check)
LABEL_MODE  = "real"          # "real": observed mortality | "rotation": semi-synthetic concept shift | "casemix": pure case-mix control
SHIFT_A     = 0.0             # rotation: 0 = same risk function everywhere ... 1 = risk function specific to each hospital group
GROUP_SEED  = 1               # rotation: random assignment of hospitals to groups; casemix: outcome draw
SEED        = 42              # score model seed

SEMI_SYNTH_A = SHIFT_A if LABEL_MODE == "rotation" else 0.0     # legacy alias of SHIFT_A, no longer read by the pipeline

def make_base_dir(label_mode, shift_a, group_seed, site_mode=None):
    site_mode = SITE_MODE if site_mode is None else site_mode
    base = "data_real_hospitals" if site_mode == "hospital" else "data_real_hospital_units"
    if label_mode == "real":
        return base
    if label_mode == "casemix":
        return f"{base}_casemix_g{group_seed}"
    return f"{base}_rot{shift_a:g}_g{group_seed}"

BASE_DIR     = make_base_dir(LABEL_MODE, SHIFT_A, GROUP_SEED)
CLIENT_DIR   = os.path.join(BASE_DIR, "clients")
MODEL_DIR    = os.path.join(BASE_DIR, "models")
FRONTIER_DIR = os.path.join(BASE_DIR, "frontier")
for _d in (CLIENT_DIR, MODEL_DIR, FRONTIER_DIR):
    os.makedirs(_d, exist_ok=True)
print(f"✓ Expérience : {BASE_DIR} | LABEL_MODE = {LABEL_MODE} | a = {SHIFT_A} | groupes = {GROUP_SEED} | seed = {SEED}")
