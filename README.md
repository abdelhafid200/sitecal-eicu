# SiteCal — site-level calibration of federated ICU mortality alarms

Code for the paper *Borrowing alarm thresholds across hospitals: site-level sensitivity guarantees for federated intensive care mortality prediction* (submitted to the International Journal of Medical Informatics).

The study asks whether small hospitals in a federated network can borrow calibration data from other hospitals to set mortality alarm thresholds with a guaranteed sensitivity, and whether that guarantee still holds at each hospital. It compares local, pooled, shrinkage, donor-based and empirical-Bayes thresholds on 129 hospitals of the eICU Collaborative Research Database, with real and semi-synthetic outcomes.

**Main result:** pooled thresholds break the guarantee at individual hospitals, even when all hospitals share the same risk function. Shrinkage toward the pooled threshold with 10 pseudo-deaths keeps the guarantee in all 20 settings tested, and cuts the share of flagged patients in small hospitals from 99.5% to 66.5% (90% sensitivity target).

## Data access

The eICU Collaborative Research Database v2.0 is available on [PhysioNet](https://physionet.org/content/eicu-crd/2.0/) to credentialed users who sign its data use agreement. **No data are included in this repository**, and patient-level files produced by the code must never be published.

Download these four tables and put them in `dataset/` at the root of the repository:

```
dataset/patient.csv.gz
dataset/hospital.csv.gz
dataset/apacheApsVar.csv.gz
dataset/apachePatientResult.csv.gz
```

## Installation

```bash
git clone https://github.com/noreply/sitecal-eicu.git
cd sitecal-eicu
pip install -r requirements.txt
```

Python 3.10 or later. A GPU is not needed.

## Reproducing the results

Open Jupyter **from the repository root** (all paths are relative to it) and run the two notebooks in order.

| Notebook | What it does | Time |
|---|---|---|
| `run_pipeline.ipynb` | Preprocessing, federated score models and all calibration runs: 42 main runs (`07`) and 12 runs of the pure case-mix control (`07c`) | about 10 h + 2.5 h on CPU |
| `run_analysis.ipynb` | Hypothesis tests, tables and figures of the paper, from the saved results | a few minutes |

Runs that are already finished are detected and skipped, so the pipeline can be stopped and restarted.

The scripts are run with `%run -i` so that they share variables, as in a single notebook: `src/00_config.py` must run before `07` and `07c`, which call the four files in `src/` once per condition, seed and model.

## Where each result comes from

| Paper element | Script | Output |
|---|---|---|
| Table 1 (cohort) | `paper/11_table1_cohort.py` | printed |
| Table 2 (pooled thresholds) | `paper/12_tables_2_3.py` | `resultats_finaux/tables_article/table2_pooled_excess.tex` |
| Table 3 (methods) | `paper/12_tables_2_3.py` | `resultats_finaux/tables_article/table3_methods.tex` |
| Figure 1 | `paper/10_figures.py` | `resultats_finaux/figures_article/Fig1_pooling_excess.pdf` |
| Figure 2 | `paper/10_figures.py` | `resultats_finaux/figures_article/Fig2_frontier.pdf` |
| Hypotheses P1–P3 | `analysis/08_analysis_main.py` | printed |
| Amendment A1 (+ raw shortfalls, mean sensitivity) | `analysis/08b_analysis_amendment1.py` | printed |
| Amendment A2 (pure case-mix control) | `analysis/08c_analysis_casemix.py` | printed |
| Small-hospital excess | `analysis/08d_analysis_small_hospitals.py` | printed |

The pre-specified analysis plan and its two dated amendments are in [`docs/analysis_plan.md`](docs/analysis_plan.md).

## Repository structure

```
sitecal-eicu/
├── run_pipeline.ipynb        runs 00 → 07 → 07c
├── run_analysis.ipynb        runs 08 → 12
├── src/
│   ├── 00_config.py          paths, condition and seed (set automatically by 07 and 07c)
│   ├── 01_preprocessing.py   cohort, hospitals, semi-synthetic outcomes, splits
│   ├── 01b_null_reference.py random reassignment of patients to hospitals (diagnostic)
│   ├── 02_score_models.py    federated global and personalized models (FedAvg + fine-tuning)
│   └── 03_threshold_study.py the 23 calibration settings, real data and random-reassignment reference
├── experiments/
│   ├── 07_run_main_plan.py           real outcomes + concept shift, 3 seeds, 2 models
│   └── 07c_run_casemix_control.py    pure case-mix control (amendment A2)
├── analysis/                 hypothesis tests (08, 08b, 08c, 08d)
├── paper/                    figures and tables (10, 11, 12)
├── tools/09_extract_logs.py  collects checks printed in notebook outputs
├── docs/analysis_plan.md
└── resultats_finaux/         aggregated results (no patient-level data)
```

**Not** published: `data_real_hospital*/` (patient-level splits and models created by the pipeline) and `paper/manuscript/` (LaTeX manuscript).

Code comments are in English. Printed outputs and some variable and column names are in French (`exces` = excess, `alertes` = alerts, `petits` = small hospitals, `réel` = real outcomes).

## Citation

If you use this code, please cite the paper (reference to be added after publication).

## License

MIT — see `LICENSE`.
