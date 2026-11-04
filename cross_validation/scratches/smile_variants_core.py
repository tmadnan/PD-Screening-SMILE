#!/usr/bin/env python3
"""Shared evaluator for five SMILE model variants.

This reuses the prior SMILE BaseModel's feature extraction, fit-only
normalization, train-fold SMOTE, and original HistGradientBoosting hyperparameters.

The historical top configurations were selected using the same development
records. Therefore THIS IS NOT A FULLY UNBIASED MODEL SELECTION EXPERIMENT.
The nested training of fitted transforms / stacking does prevent the in-run
preprocessing and base-model stacking leakage previously identified.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import expit
from scipy.stats import norm
from imblearn.over_sampling import RandomOverSampler
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import smile_base

VARIANTS = {
    "single": None,
    "binary_none": ("binary", "none"),
    "binary_random": ("binary", "random"),
    "probability_none": ("probability", "none"),
    "probability_random": ("probability", "random"),
}


class FastLBFGSBaseModel(smile_base.BaseModel):
    """Approximate original unpenalized, no-intercept logistic feature ranking.

    sklearn LBFGS replaces original statsmodels BFGS. The 200-iteration cap can
    fail to converge, changing feature ranks. Warnings are intentionally shown.
    This is a numerical adaptation, NOT an exact code replication.
    """
    def _select(self, x: np.ndarray, y: np.ndarray) -> list[str]:
        method = self.cfg.feature_selection.strip().lower()
        if method not in ("logistic regression", "lr", "coeff"):
            return super()._select(x, y)
        n = min(len(self.columns), self.cfg.num_features)
        model = LogisticRegression(
            penalty=None, fit_intercept=False, solver="lbfgs",
            max_iter=200, tol=1e-5,
        )
        model.fit(x, y)
        coeff = np.asarray(model.coef_[0], dtype=float)
        if "pvalue" in self.cfg.sorting_metric.lower() or "p-value" in self.cfg.sorting_metric.lower():
            p = expit(x @ coeff)
            information = (x.T * (p * (1.0 - p))) @ x
            variance = np.diag(np.linalg.pinv(information))
            z = np.abs(coeff) / np.sqrt(np.maximum(variance, 1e-12))
            order = np.argsort(2.0 * norm.sf(z))
        else:
            order = np.argsort(-np.abs(coeff))
        return [self.columns[i] for i in order[:n]]


def new_base(config, cols, ranking_solver):
    factory = FastLBFGSBaseModel if ranking_solver == "lbfgs" else smile_base.BaseModel
    return factory(config, cols)


def splits(y, group, count, seed):
    """Preserve original stratified 10-fold split if IDs are unique.

    Fall back to group-stratified folds if there is more than one record per ID.
    """
    if len(np.unique(group)) == len(group):
        ans = list(StratifiedKFold(n_splits=count, shuffle=True, random_state=seed)
                   .split(np.zeros(len(y)), y))
    else:
        ans = smile_base.splits_for(y, group, count, seed, "group-aware CV")
    for tr, te in ans:
        assert set(group[tr]).isdisjoint(set(group[te])), "Participant ID leaked across fold"
    return ans


def original_meta(args):
    version = tuple(int(x) for x in sklearn_version.split(".")[:2])
    penalty_arg = (
        {"l1_ratio": 1.0 if args.penalty == "l1" else 0.0}
        if version >= (1, 8) else {"penalty": args.penalty}
    )
    return LogisticRegression(
        C=args.C, max_iter=args.max_iter, solver=args.solver,
        tol=args.tol, random_state=args.random_state, **penalty_arg,
    )


def score_summary(y, scores, folds):
    m = smile_base.metrics(y, scores)
    m["Mean_Fold_AUROC"] = float(np.mean([
        roc_auc_score(y[te], scores[te]) for _, te in folds
    ]))
    return {name: round(float(val) * 100, 4) for name, val in m.items()}


def run_fold(fold_number, tr, te, df, y, groups, cols, configs, required, args):
    base_predictions = np.full((len(te), len(configs)), np.nan)
    stack_features = None
    chosen_modes = [(v, VARIANTS[v]) for v in required if v != "single"]
    if chosen_modes:
        inner = splits(y[tr], groups[tr], args.inner_folds, args.cv_seed)
        stack_features = np.full((len(tr), len(configs)), np.nan)
    for j, cfg in enumerate(configs):
        if chosen_modes:
            for subtrain, subholdout in inner:
                model = new_base(cfg, cols, args.ranking_solver)
                model.fit(df.iloc[tr[subtrain]], y[tr[subtrain]])
                stack_features[subholdout, j] = model.predict_proba(df.iloc[tr[subholdout]])
        model = new_base(cfg, cols, args.ranking_solver)
        model.fit(df.iloc[tr], y[tr])
        base_predictions[:, j] = model.predict_proba(df.iloc[te])
    if not np.isfinite(base_predictions).all() or (chosen_modes and not np.isfinite(stack_features).all()):
        raise RuntimeError("Some base-model predictions are missing/nonfinite")
    stack_outputs = {}
    for name, (mode, balance) in chosen_modes:
        x_meta_train = (stack_features >= 0.5).astype(float) if mode == "binary" else stack_features
        x_meta_test = (base_predictions >= 0.5).astype(float) if mode == "binary" else base_predictions
        meta_y = y[tr]
        if balance == "random":
            x_meta_train, meta_y = RandomOverSampler(random_state=args.random_state).fit_resample(x_meta_train, meta_y)
        meta = original_meta(args)
        meta.fit(x_meta_train, meta_y)
        stack_outputs[name] = meta.predict_proba(x_meta_test)[:, 1]
    row = {"Fold": fold_number, "N_train": len(tr), "N_test": len(te)}
    for j in range(len(configs)):
        row[f"Base_{j+1}_AUROC"] = smile_base.metrics(y[te], base_predictions[:, j])["AUROC"] * 100
    for variant, p in stack_outputs.items():
        m = smile_base.metrics(y[te], p)
        row[f"{variant}_AUROC"] = m["AUROC"] * 100
        row[f"{variant}_Sensitivity"] = m["Sensitivity"] * 100
    return fold_number, te, base_predictions, stack_outputs, row


def main(force_variant=None, argv=None):
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=root / "dataset-cross-validation.csv")
    ap.add_argument("--configs", type=Path, default=root / "cross-validation-hist-configs.csv")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--variant", choices=["all", *VARIANTS], default="all")
    ap.add_argument("--num-configs", type=int, default=18)
    ap.add_argument("--outer-folds", type=int, default=10)
    ap.add_argument("--inner-folds", type=int, default=3,
                    help="3 reproduces audit; 10 is closer to original 10-fold protocol but slower")
    ap.add_argument("--cv-seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--ranking-solver", choices=["lbfgs", "statsmodels"], default="lbfgs")
    ap.add_argument("--C", type=float, default=40.36631131136931)
    ap.add_argument("--max-iter", type=int, default=115)
    ap.add_argument("--penalty", choices=["l1", "l2"], default="l1")
    ap.add_argument("--solver", default="saga")
    ap.add_argument("--tol", type=float, default=0.0009070426942042604)
    ap.add_argument("--random-state", type=int, default=6424)
    args = ap.parse_args(argv)
    if force_variant is not None:
        args.variant = force_variant
    if args.num_configs < 1 or args.inner_folds < 2 or args.outer_folds < 2:
        raise ValueError("num-configs >= 1 and folds >= 2 are required")
    if args.variant == "all":
        required = list(VARIANTS)
    else:
        required = [args.variant]
    outdir = args.output or (root / "runs" / args.variant)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.dataset)
    if df["ID"].isna().any() or df["Diagnosis"].isna().any():
        raise ValueError("Missing participant IDs or PD diagnoses")
    if (df.groupby("ID")["Diagnosis"].nunique() > 1).any():
        raise ValueError("Conflicting diagnosis for a participant ID")
    y = df["Diagnosis"].to_numpy(dtype=int)
    groups = df["ID"].to_numpy()
    if not set(np.unique(y)) == {0, 1}:
        raise ValueError("Expected binary Diagnosis (0, 1)")
    feature_cols = smile_base.feature_names(df, ["smile"], True, True)

    historical = pd.read_csv(args.configs).sort_values("Accuracy", ascending=False)
    historical_top = historical.head(args.num_configs)
    if len(historical_top) != args.num_configs:
        raise ValueError("Not enough saved configurations")
    config_by_index = {c.source_row: c for c in smile_base.read_candidates(args.configs, 0, args.random_state)}
    selected = [config_by_index[int(row_index)] for row_index in historical_top.index]
    if any(c.model_name != "HistGradientBoosting" for c in selected):
        raise ValueError("Original top-config selection included non-HistBoost; check the CSV")
    # A baseline run only needs top-ranked historical configuration.
    if required == ["single"]:
        selected = selected[:1]
        historical_top = historical_top.iloc[:1]

    outer = splits(y, groups, args.outer_folds, args.cv_seed)
    base_out = np.full((len(df), len(selected)), np.nan)
    stacked_out = {name: np.full(len(df), np.nan) for name in required if name != "single"}
    print(f"{args.variant}: {len(df)} rows, {len(np.unique(groups))} participants, {len(selected)} HistBoost configs, ",
          f"outer={args.outer_folds}, inner={args.inner_folds}, ranking_solver={args.ranking_solver}", flush=True)
    tasks = Parallel(n_jobs=args.jobs)(
        delayed(run_fold)(i, tr, te, df, y, groups, feature_cols, selected, required, args)
        for i, (tr, te) in enumerate(outer, start=1)
    )
    fold_rows = []
    test_folds = np.full(len(df), -1, dtype=int)
    for i, te, bp, sp, fold_row in sorted(tasks, key=lambda x: x[0]):
        test_folds[te] = i
        base_out[te, :] = bp
        for name, p in sp.items():
            stacked_out[name][te] = p
        fold_rows.append(fold_row)
        print(f"Fold {i:02d}/{args.outer_folds}: n_test={len(te)}, base_1_auc={fold_row['Base_1_AUROC']:.2f}", flush=True)
    if (test_folds < 0).any() or not np.isfinite(base_out).all():
        raise RuntimeError("Test coverage check failed")
    if any(not np.isfinite(score).all() for score in stacked_out.values()):
        raise RuntimeError("Meta predictions contain missing data")

    results = {
        "method": "Fixed historical configurations, in-run leakage-controlled folds",
        "variant": args.variant,
        "not_independent_hyperparameter_selection": True,
        "n_rows": len(df), "n_participants": len(np.unique(groups)),
        "outer_folds": args.outer_folds, "inner_folds": args.inner_folds,
        "selected_config_count": len(selected), "features_before_selection": len(feature_cols),
        "feature_ranking_solver": args.ranking_solver,
        "model_params": {"C": args.C, "max_iter": args.max_iter, "penalty": args.penalty,
                         "solver": args.solver, "tol": args.tol, "random_state": args.random_state},
        "base_models": {}, "stack_variants": {},
        "known_limitations": [
            "Top configurations were originally selected on this same development dataset.",
            "Fast LBFGS feature ranking differs from statsmodels.Logit and can be nonconverged.",
            "Balanced meta-training uses RandomOverSampler; original code's meta inputs were produced differently.",
            "No Clinic or Home-BD external validation performed in this run.",
        ],
    }
    pred = pd.DataFrame({"ID": groups, "Diagnosis": y, "Outer_Fold": test_folds})
    for j, c in enumerate(selected):
        key = f"Base_{j+1}"
        pred[f"{key}_Probability"] = base_out[:, j]
        results["base_models"][key] = {"source_row": c.source_row,
                                      "historical_WandB_name": str(historical_top.iloc[j].get("Name", "")),
                                      "metrics": score_summary(y, base_out[:, j], outer)}
    for name, s in stacked_out.items():
        pred[f"Stack_{name}_Probability"] = s
        results["stack_variants"][name] = score_summary(y, s, outer)
    pd.DataFrame(fold_rows).to_csv(outdir / "fold_metrics.csv", index=False)
    pred.to_csv(outdir / "heldout_predictions.csv", index=False)
    historical_top.to_csv(outdir / "historical_configurations.csv", index=False)
    (outdir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved", outdir, flush=True)
    for key, result in results["stack_variants"].items():
        print(key, result, flush=True)
    if results["base_models"]:
        print("historical-top-1", results["base_models"]["Base_1"]["metrics"], flush=True)
    return results


if __name__ == "__main__":
    main()
