# SMILE — separate, Git-ready leakage-controlled evaluation variants

This folder contains five independently executable scripts with **one shared implementation** so folds, preprocessing, base configurations, and metric calculations stay consistent. It is a *re-evaluation* of the historical SMILE HistBoost experiments, not a replacement for the published study.

## Files and model versions

| Script | Model / change | Meta learner input | Meta training balance |
|---|---|---|---|
| `01_single_histboost.py` | Historical top-ranked HistBoost alone | N/A | N/A |
| `02_stack_binary.py` | Original-style stacking, corrected fold boundaries | Hard class predictions, 0/1 | Natural class ratio |
| `03_stack_binary_balanced.py` | Binary stacking with balanced meta training | Hard class predictions, 0/1 | RandomOverSampler |
| `04_stack_probability.py` | Probability stacking | Continuous probabilities | Natural class ratio |
| `05_stack_probability_balanced.py` | Probability stacking with balanced meta training | Continuous probabilities | RandomOverSampler |
| `06_compare_all.py` | All five variants **on the same fitted bases and folds** | Each as above | Each as above |

`smile_variants_core.py` is the common evaluator. `smile_base.py` is the earlier corrected SMILE feature-selection/SMOTE/HistBoost implementation reused by all variants. Each numbered Python file is intentionally a small, independent entry point; this avoids five subtly different versions of data splitting or preprocessing in Git.

`reference_results/` contains the **previously completed full 10-outer-fold, 3-inner-fold, 18-configuration real-data comparison**, not new metrics from the final packaged code. It is supplied for comparison/auditing. The current package passed a 2-outer-fold/2-inner-fold/2-config end-to-end smoke test on the uploaded real input files.

## Run from your Git repository

Place this directory inside `PD-Screening-SMILE/cross_validation/` (e.g. `cross_validation/leakage_controlled_versions/`). Keep your existing original files unchanged. Then:

```bash
cd PD-Screening-SMILE/cross_validation/leakage_controlled_versions
python -m pip install -r requirements.txt

python 06_compare_all.py \
  --dataset ../dataset-cross-validation.csv \
  --configs ../cross-validation-hist-configs.csv \
  --outer-folds 10 --inner-folds 3 --num-configs 18 --jobs 4
```

Each script also runs independently. For example:

```bash
python 01_single_histboost.py --dataset ../dataset-cross-validation.csv --configs ../cross-validation-hist-configs.csv
python 02_stack_binary.py --dataset ../dataset-cross-validation.csv --configs ../cross-validation-hist-configs.csv
python 03_stack_binary_balanced.py --dataset ../dataset-cross-validation.csv --configs ../cross-validation-hist-configs.csv
python 04_stack_probability.py --dataset ../dataset-cross-validation.csv --configs ../cross-validation-hist-configs.csv
python 05_stack_probability_balanced.py --dataset ../dataset-cross-validation.csv --configs ../cross-validation-hist-configs.csv
```

By default each variant writes `runs/<variant>/{metrics.json,fold_metrics.csv,heldout_predictions.csv,historical_configurations.csv}`. The combined evaluator writes `runs/all/` and shares base model fits, rather than wasting time fitting them separately for each variant. For a rapid check, add `--num-configs 2 --outer-folds 2 --inner-folds 2 --jobs 1`.

## What is intentionally preserved

- Original **top 18 configurations selected by historical `Accuracy`**, including pandas' default tie sorting (not random candidates).
- Exact saved HistBoost learning rate, depth, leaves, feature-count and random seed from each configuration row.
- HistGradientBoostingClassifier's otherwise default parameters and the original smile-only AU + MediaPipe feature subset.
- Original logistic-regression meta-classifier defaults: **C=40.36631131136931, max_iter=115, penalty=l1, solver=saga, tol=0.0009070426942042604, random_state=6424**.
- Original outer `StratifiedKFold` with shuffle and seed 42 when one row per participant, as in the uploaded dataset (827 records, 827 unique IDs). If IDs repeat, a participant-grouped splitter is used.
- Base-level SMOTE on **training** data only, as before. Final model performance is measured on real held-out participants without synthetic evaluation records.

## What changes (and why)

1. The feature selector, its scaling, and its SMOTE-based ranking data are fitted inside each relevant **training** fold only; the held-out fold is never used for fitting.
2. Base model features for the meta-classifier are generated **out of fold within each outer training partition**. Base models are then refit on that outer-training partition and evaluated on its untouched outer test fold. The meta-classifier does not train on in-sample predictions.
3. `04_*` and `05_*` use **probabilities** instead of the original hard predictions; this is an **experimental architecture change**, not merely a leakage fix.
4. `03_*` and `05_*` balance the **meta-feature** training data via `RandomOverSampler`; this is a new diagnostic choice. The original code obtained balanced training examples differently by SMOTE applied before its meta training prediction generation.
5. Feature-ranking numerical solver defaults to **sklearn unpenalized, no-intercept logistic regression (LBFGS, max_iter=200)**, replacing the original `statsmodels.Logit(...).fit(method='bfgs', maxiter=1000)`. This substitution was necessary for completing the earlier comparison but it can change which features are selected and can emit *ConvergenceWarning*. It is **not** an exact replication. Use `--ranking-solver statsmodels` to run the original implementation; it may be very slow or unstable. All warning messages should be retained and audited.
6. Inner folds default to **3**, the setting used for the completed diagnostic study. For a more extensive analysis, pass `--inner-folds 10` (with a greater computational cost). Inner folds only generate meta-training OOF features; historical model configuration selection remains frozen.

## Remaining sources of bias and interpretation

**Do not describe these results as fully leakage-free model selection.** Historical top-18 configuration IDs came from searches using the same 827-person development dataset. The current re-evaluation contains no direct held-out-fold preprocessing or stacking-fitting leakage **within each run**, but earlier use of development labels to select model configurations can still bias CV estimates. Truly independent performance needs either a fresh outer-fold configuration search using a prespecified candidate space or an untouched external test cohort.

**Do not equate the first-ranked configuration with the strongest present-day baseline.** The first row was selected by historical *accuracy*. Its pooled new AUROC was 88.52%. Among all 18 original configurations, the largest observed pooled AUROC was 89.97% (sixth historical-accuracy rank). This ranking is *post hoc* on held-out predictions and is **not valid as a fresh unbiased model selection**.

The historical config CSV has scores (Accuracy, ROC AUC) but does **not** encode all preprocessing choices (e.g., scaler/feature selection/SMOTE). Those omitted fields are filled with the documented selected defaults. This is an assumption, not exact reconstruction of every historical run.

The supplementary manuscript refers to 640 non-PD + 195 PD enrolled/development participants (835), while the uploaded *usable feature CSV* has 632 non-PD + 195 PD (827). That discrepancy merits a check against the exact cohort inclusion and video-validity criteria before treating the datasets as identical.

## Previously completed 10-fold diagnostic results (NOT a fresh run of these packaged scripts)

| Model | Pooled AUROC | Accuracy | Sensitivity | Specificity |
|---|---:|---:|---:|---:|
| Historical top-accuracy HistBoost only | 88.52 | 82.22 | 63.59 | 87.97 |
| Binary LR stack, no meta balancing | 84.73 | 85.13 | 63.08 | 91.93 |
| Binary LR stack, meta balancing | 84.53 | 84.64 | 73.33 | 88.13 |
| Probability LR stack, no meta balancing | 89.03 | 84.52 | 63.08 | 91.14 |
| Probability LR stack, meta balancing | 88.63 | 83.68 | 78.46 | 85.28 |

These are pooled AUROCs, not averages of fold AUROCs, at threshold 0.5. The full per-model results, fold metrics and participant-level OOF predictions are in `reference_results/`.

## Next scientific checks

1. Compare *identical* training/evaluation splits with original versus new feature-ranking solver and record selected feature overlaps, nonconvergence and coefficient variation.
2. Confirm the exact 827-person usable-video cohort and original code commit/data version.
3. Assess the candidate-selection effect with a proper outer-fold hyperparameter search (large computational budget) or freeze the resulting full pipeline and measure Clinic and Home-BD external cohorts.
4. Evaluate thresholds chosen only inside outer training folds, reporting AUROC separately from threshold-dependent sensitivity/specificity.

Original project source: https://github.com/tmadnan/PD-Screening-SMILE/tree/main/cross_validation
