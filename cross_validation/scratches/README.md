# SMILE — HistBoost Ensemble Experiments

This directory introduces an experimental suite for comparing **single-model HistBoost** with several **logistic-regression stacking** strategies for Parkinson's disease screening from smile-video features. The goal is to study how base-model prediction representation and meta-training class balance affect classification performance while keeping the underlying HistBoost configurations and evaluation splits consistent.

The experiments build on the established SMILE feature representation and model configurations. They are **additional modeling experiments**, not changes to the published SMILE findings.

## Experiments

| Script | Experiment | Meta-model inputs | Meta-training distribution |
|---|---|---|---|
| `01_single_histboost.py` | Single HistBoost reference model | — | — |
| `02_stack_binary.py` | Logistic regression stacking with class labels | Binary predictions (0 or 1) | Observed class distribution |
| `03_stack_binary_balanced.py` | Binary stacking with class-balanced meta-training | Binary predictions (0 or 1) | Random oversampling |
| `04_stack_probability.py` | Logistic regression stacking with confidence scores | Predicted probabilities | Observed class distribution |
| `05_stack_probability_balanced.py` | Probability stacking with class-balanced meta-training | Predicted probabilities | Random oversampling |
| `06_compare_all.py` | Evaluate all five variants in one run | As specified above | As specified above |

Each numbered script can be run independently. The shared modules, `smile_base.py` and `smile_variants_core.py`, implement data preparation, model construction, cross-validation, and evaluation consistently across experiments. The combined runner reuses the same fitted base models and folds for more directly comparable results.

## Experimental design

- **Input representation:** SMILE's structured action-unit and MediaPipe-derived facial features.
- **Base-model configurations:** The selected HistBoost configurations are read from `cross-validation-hist-configs.csv`. By default, the first 18 configurations are chosen according to the historical validation-accuracy ranking (including the CSV's original sorting behavior).
- **Base-model settings:** Saved HistBoost learning rates, tree settings, feature counts, and random seeds are retained from the corresponding configuration entries.
- **Meta-classifier:** Logistic regression with the previously used settings (`C=40.36631131136931`, `penalty=l1`, `solver=saga`, `max_iter=115`, `tol=0.0009070426942042604`, `random_state=6424`).
- **Evaluation:** By default, 10 outer folds and 3 inner folds are used. Fold-wise feature preparation, feature ranking, and SMOTE operate on the applicable training partitions. The stacking experiments use out-of-fold predictions to construct meta-training features. Final metrics are calculated on held-out participants.
- **Participant separation:** With one record per participant, stratified folds are used; when repeated participant IDs are present, group-aware splitting is used.
- **Comparison controls:** All variants use matched partitions and base-model configurations, allowing the effects of binary vs. probability inputs and balanced vs. observed meta-training distributions to be examined separately.

The suite uses a default `lbfgs` numerical solver for logistic-regression-based feature ranking. The `--ranking-solver statsmodels` option is available for experiments with the alternative implementation. Feature ranking may show convergence warnings, which should be reviewed when interpreting model comparisons.

## Installation and execution

Place this directory within `PD-Screening-SMILE/cross_validation/`, for example as `cross_validation/ensemble_experiments/`. Keep the existing experiment scripts and datasets in place.

```bash
cd PD-Screening-SMILE/cross_validation/ensemble_experiments
python -m pip install -r requirements.txt

python 06_compare_all.py \
  --dataset ../dataset-cross-validation.csv \
  --configs ../cross-validation-hist-configs.csv \
  --outer-folds 10 \
  --inner-folds 3 \
  --num-configs 18 \
  --jobs 4
```

To run a specific variant:

```bash
python 01_single_histboost.py --dataset ../dataset-cross-validation.csv --configs ../cross-validation-hist-configs.csv
python 02_stack_binary.py --dataset ../dataset-cross-validation.csv --configs ../cross-validation-hist-configs.csv
python 03_stack_binary_balanced.py --dataset ../dataset-cross-validation.csv --configs ../cross-validation-hist-configs.csv
python 04_stack_probability.py --dataset ../dataset-cross-validation.csv --configs ../cross-validation-hist-configs.csv
python 05_stack_probability_balanced.py --dataset ../dataset-cross-validation.csv --configs ../cross-validation-hist-configs.csv
```

For a quicker execution check:

```bash
python 06_compare_all.py \
  --dataset ../dataset-cross-validation.csv \
  --configs ../cross-validation-hist-configs.csv \
  --outer-folds 2 --inner-folds 2 --num-configs 2 --jobs 1
```

Results are written under `runs/<variant>/` (or `runs/all/` for the combined comparison), including `metrics.json`, `fold_metrics.csv`, `heldout_predictions.csv`, and `historical_configurations.csv`.

## Reference results

The `reference_results/` directory contains outputs from a previously completed **10-outer-fold / 3-inner-fold / 18-configuration** comparison on 827 available participant records. These files are archived experimental reference results, **not a fresh execution of the final packaged entry points**.

| Experiment | Pooled AUROC (%) | Accuracy (%) | Sensitivity (%) | Specificity (%) |
|---|---:|---:|---:|---:|
| Single HistBoost (historically top-ranked by accuracy) | 88.52 | 82.22 | 63.59 | 87.97 |
| Binary stacking | 84.73 | 85.13 | 63.08 | 91.93 |
| Binary stacking + balanced meta-training | 84.53 | 84.64 | 73.33 | 88.13 |
| Probability stacking | 89.03 | 84.52 | 63.08 | 91.14 |
| Probability stacking + balanced meta-training | 88.63 | 83.68 | 78.46 | 85.28 |

The results illustrate trade-offs among model discrimination, sensitivity, and specificity. In this run, probability-based stacking achieved higher AUROC than binary stacking, while balanced meta-training increased sensitivity with a corresponding specificity trade-off. All tabulated metrics use pooled outer-held-out predictions and a classification threshold of 0.5 for threshold-dependent metrics.

The *historically top-ranked* single model is defined by its earlier accuracy ranking, not by maximum AUROC in the new reference comparison. Different ranking criteria can select different configurations.

## Scope and interpretation

This package is intended for **controlled comparative experiments**, not as a claim of improved clinical screening performance. The candidate HistBoost configurations originated from prior development experiments on the same cohort, so these CV results should be treated as development-stage estimates. An untouched external cohort or a fully prespecified, nested model-selection study would provide a stronger independent performance estimate.

The uploaded feature CSV used here has 827 participant records; the supplementary manuscript describes a larger development cohort. Reproducing manuscript numbers requires verifying the exact analysis cohort and data version. The numerical feature-ranking implementation and the number of inner folds can also affect the observed metrics. See `SMOKE_TEST.md` for the package execution check.

## Potential follow-up experiments

- Evaluate alternative meta-learning representations, including probability calibration and threshold selection.
- Explore meta-classifier regularization and class-weighting strategies under a consistent outer evaluation protocol.
- Compare selected feature subsets and optimization convergence across feature-ranking solvers.
- Assess model robustness on independent Clinic and Home-BD cohorts.

**Project:** [PD-Screening-SMILE / cross_validation](https://github.com/tmadnan/PD-Screening-SMILE/tree/main/cross_validation)
