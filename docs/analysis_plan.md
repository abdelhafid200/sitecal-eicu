# Pre-specified analysis plan and amendments

This file records the analysis plan as it was fixed during the study, with the dates of each step.
The original working notes were written in French; this is an English summary.

## Plan fixed on 13 October 2026 (before the confirmatory runs)

Fixed after exploratory analyses (one seed, an earlier semi-synthetic design that is not part of this code) and before any confirmatory run.

**Conditions**
- Real outcomes: eICU-CRD v2.0, 129 hospitals with at least 200 stays and 10 deaths, in-hospital mortality.
- Concept shift ("rotation"): logit = s·[cos(θ)·z0 + sin(θ)·h_g] + c_g, θ = a·π/2, a ∈ {0, 0.5, 1}, 8 groups of hospitals,
  two group assignments (`GROUP_SEED` 1 and 2). c_g sets each group's mortality to the overall rate.
- Score models: personalized (federated averaging + local fine-tuning) and global (federated averaging); seeds 42, 123, 2025.

**Evaluation**
- Each hospital's held-out set is split 50 times into calibration and test halves.
- Significant shortfall: mean miss rate > α with z > 1.645.
- Excess = share of hospitals with a significant shortfall minus the same share when patients are randomly reassigned to hospitals (3 repetitions).
- A setting is valid in a condition when its excess, averaged over seeds, is ≤ 2 points for all hospitals and for the smallest third of hospitals.
- Primary α = 0.10 (90% sensitivity); secondary α = 0.20.

**Primary settings**: local conformal; pooled conformal; shrinkage n0 = 10 (from the exploratory analysis) and n0 = 19; donors M = 10;
empirical Bayes η = 0.05 (all hospitals and donors). All other settings are secondary.

**Hypotheses (α = 0.10)**
- P1 (H1): local calibration flags ≥ 90% of patients in small hospitals on real outcomes.
- H2 (descriptive): raw shortfalls of pooled thresholds, real versus random reassignment.
- P2 (H3): under the global model, the pooled excess increases by ≥ 5 points from a = 0 to a = 1; under personalized models it is ≤ 2 points at a = 1.
- P3 (H4): shrinkage n0 = 10 is valid in every cell and flags fewer patients than local calibration in small hospitals on real outcomes.
- H5 (exploratory): empirical Bayes η = 0.05 compared with shrinkage n0 = 10.

## Amendment A1 — 14 October 2026 (after a one-seed preview, before analyzing group assignment 2)
- A1a: at a = 0, pooled thresholds are invalid under both models (excess > 2 points).
- A1b: under personalized models, the pooled excess falls by ≥ 5 points from a = 0 to a = 1 and is ≤ 2 points at a = 1, while it stays > 2 points under the global model.
- Tested on `GROUP_SEED = 2` only.

## Amendment A2 — 14 October 2026 (before any run of the pure case-mix condition)
- Reason: at a = 0, group constants c_g differ, so the a = 0 condition is not pure case-mix.
- New condition: logit = s·z0 + c with a single constant c (same risk function for every hospital); 2 label draws × 3 seeds × 2 models.
- A2a: pooled thresholds are invalid under both models. A2b: shrinkage n0 = 10 stays valid.
- Added on 16 October 2026, before the A2 results were seen: empirical Bayes η = 0.2 (found in the secondary sweep) is tested on this new condition, as an exploratory analysis.

## Outcomes
| Hypothesis | Outcome |
|---|---|
| P1 | Confirmed (99.5% of patients flagged) |
| H2 | 27% of hospitals below 85% sensitivity versus 19% after random reassignment |
| P2 | Not confirmed (increase of 1.8 points) |
| P3 | Confirmed (valid in 8/8 cells at both targets; 66.5% vs 99.5% flagged) |
| A1a | Confirmed (global 4.9, personalized 3.8 points) |
| A1b | Not confirmed (decrease of 4.8 points) |
| A2a | Confirmed (personalized 4.8 ± 0.8, global 5.3 ± 1.9 points) |
| A2b | Confirmed |
| H5 | Empirical Bayes η = 0.05 valid 10/10 (all hospitals) and 9/10 (donors), with more alarms |
| Exploratory | Empirical Bayes η = 0.2 valid in 20/20 combinations |
