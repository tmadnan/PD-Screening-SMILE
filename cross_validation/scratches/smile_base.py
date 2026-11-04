#!/usr/bin/env python3
"""Nested, group-aware evaluation of SMILE HistBoost -> LogisticRegression stacking.

Replacement for cross_validation/cross-validation-hist-ensemble-lr.py in
https://github.com/tmadnan/PD-Screening-SMILE

Core design:
  OUTER CV = entirely unseen people for evaluation.
  Inside each OUTER training portion:
    INNER CV = refit independently configured base models; produce OOF probabilities.
    Rank candidate configurations ONLY on inner OOF performance.
    Train logistic-regression meta-classifier using OOF probabilities.
    Refit selected base models on ALL outer-training participants.
  Predict outer-held-out participants once with that frozen stack.

Every learned feature preprocessing step (including feature selection and
SMOTE) is refit inside the applicable *base model's* training partition.

IMPORTANT: To obtain a defensible unbiased estimate, the candidate CONFIGURATIONS
and meta-learner search space must not themselves have been chosen based on the
entire dataset being evaluated. A configuration CSV consisting solely of prior
'winners' on that same dataset retains earlier selection bias, even though this
script does not read its historical score columns. Supply an independently
specified candidate grid, or evaluate with a genuinely untouched cohort.

Output metrics WILL differ from the original script. This is expected.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
from sklearn.svm import SVC


@dataclass(frozen=True)
class Candidate:
    source_row: int
    model_name: str
    learning_rate: float
    max_depth: int | None
    max_leaf_nodes: int | None
    random_state: int
    num_features: int
    feature_selection: str
    sorting_metric: str
    oversampling: bool
    scaler: str
    # For optional SVM candidates in the original CSV
    C: float = 1.0
    gamma: str | float = "scale"
    kernel: str = "rbf"


def scalar(v: Any, default: Any = None) -> Any:
    return default if pd.isna(v) else v


def bool_value(v: Any, default: bool) -> bool:
    if v is None or pd.isna(v):
        return default
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n"):
        return False
    raise ValueError(f"Cannot parse boolean: {v!r}")


def read_candidates(path: Path, candidate_limit: int, seed: int) -> list[Candidate]:
    """Never rank/filter by the CSV's historical performance columns."""
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError("Configuration CSV is empty")
    if "model_name" not in raw:
        raise ValueError("Configuration CSV must contain model_name")

    allowed = {"HistGradientBoosting", "Hist Gradient Boosting", "HistBoost", "SVM"}
    raw = raw[raw["model_name"].isin(allowed)].copy()
    if raw.empty:
        raise ValueError("No supported HistGradientBoosting or SVM candidates in CSV")

    # Candidate limiting is RNG-based, not based on historical Accuracy/AUROC.
    # Prefer --candidate-limit 0 (all) for best methodological assurance.
    if candidate_limit > 0 and len(raw) > candidate_limit:
        chosen = np.random.default_rng(seed).choice(len(raw), candidate_limit, replace=False)
        raw = raw.iloc[np.sort(chosen)]
        print(f"[info] Randomly sampled {candidate_limit} candidate rows (ignored prior score columns)")

    result: list[Candidate] = []
    for ix, row in raw.iterrows():
        lookup = lambda key, default=None: scalar(row.get(key, default), default)
        model_name = str(lookup("model_name"))
        if model_name in {"Hist Gradient Boosting", "HistBoost"}:
            model_name = "HistGradientBoosting"
        scale_name = str(lookup("scaler", "MinMaxScaler"))
        if scale_name.lower() in ("none", "no scaler", "false"):
            scale_name = "None"
        if scale_name not in {"MinMaxScaler", "StandardScaler", "None"}:
            raise ValueError(f"Unsupported scaler {scale_name!r} in config row {ix}")
        feature_sel = str(lookup("feature_selection", "Logistic Regression"))
        if feature_sel.lower() in {"nan", "none", "false"}:
            feature_sel = "None"
        nr = int(float(lookup("num_features", lookup("num_selected_features", 24))))
        if nr < 1:
            raise ValueError("num_features must be positive")
        result.append(Candidate(
            source_row=int(ix), model_name=model_name,
            learning_rate=float(lookup("learning_rate", 0.1)),
            max_depth=int(float(lookup("max_depth"))) if lookup("max_depth") is not None else None,
            max_leaf_nodes=int(float(lookup("max_leaf_nodes"))) if lookup("max_leaf_nodes") is not None else None,
            random_state=int(float(lookup("random_state", 42))),
            num_features=nr, feature_selection=feature_sel,
            sorting_metric=str(lookup("sorting_metric", "coeff")),
            oversampling=bool_value(lookup("oversampling", True), True),
            scaler=scale_name, C=float(lookup("C", 1.0)),
            gamma=lookup("gamma", "scale"), kernel=str(lookup("kernel", "rbf")),
        ))
    return result


def make_scaler(name: str):
    if name == "MinMaxScaler":
        return MinMaxScaler()
    if name == "StandardScaler":
        return StandardScaler()
    if name == "None":
        return None
    raise ValueError(f"Unknown scaling option: {name}")


def feature_names(df: pd.DataFrame, expressions: list[str], enable_au: bool, enable_mp: bool) -> list[str]:
    if not enable_au and not enable_mp:
        raise ValueError("Cannot disable both AU and MP features")
    cols = []
    for col in df.columns:
        if not any(ex in col.lower() for ex in expressions):
            continue
        au = "AU" in col  # Match original repo's feature filter.
        if (au and enable_au) or (not au and enable_mp):
            cols.append(col)
    if not cols:
        raise ValueError("No feature columns matched expressions and AU/MP selection")
    for col in cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Selected feature column {col!r} is not numeric")
    return cols


def smote_only_training(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(np.asarray(y, dtype=int), minlength=2)
    small = int(counts.min())
    if small < 2:
        raise ValueError("SMOTE requires >=2 minority-class training rows per fit")
    k = min(5, small - 1)
    return SMOTE(random_state=seed, k_neighbors=k).fit_resample(X, y)


class BaseModel:
    """Each instance owns its own fit-only imputer, scaler, selector and SMOTE."""

    def __init__(self, candidate: Candidate, columns: list[str]):
        self.cfg = candidate
        self.columns = columns

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> "BaseModel":
        if not set(self.columns).issubset(x.columns):
            raise ValueError("Training input is missing one or more facial feature columns")
        y = np.asarray(y, dtype=int)
        self.imputer = SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True)
        all_train = self.imputer.fit_transform(x[self.columns])

        # Selection ranking uses TRAINING ROWS ONLY. Preserve original script's
        # rank-time scaling + L2 normalization; no estimator ever sees the holdout.
        rank_scaler = make_scaler(self.cfg.scaler)
        ranking_x = rank_scaler.fit_transform(all_train) if rank_scaler else all_train
        ranking_x = normalize(ranking_x)
        rank_y = y
        if self.cfg.oversampling:
            ranking_x, rank_y = smote_only_training(ranking_x, y, self.cfg.random_state)
        self.selected = self._select(ranking_x, rank_y)
        self.selected_indices = [self.columns.index(f) for f in self.selected]

        # Fit a NEW scaler, as the original pipeline did on the final selected
        # feature subset, but now strictly within THIS training partition.
        final_x = all_train[:, self.selected_indices]
        self.scaler = make_scaler(self.cfg.scaler)
        if self.scaler is not None:
            final_x = self.scaler.fit_transform(final_x)
        final_y = y
        if self.cfg.oversampling:
            final_x, final_y = smote_only_training(final_x, y, self.cfg.random_state)

        if self.cfg.model_name == "HistGradientBoosting":
            self.model = HistGradientBoostingClassifier(
                learning_rate=self.cfg.learning_rate,
                max_depth=self.cfg.max_depth,
                max_leaf_nodes=self.cfg.max_leaf_nodes,
                random_state=self.cfg.random_state,
            )
        elif self.cfg.model_name == "SVM":
            self.model = SVC(C=self.cfg.C, gamma=self.cfg.gamma,
                             kernel=self.cfg.kernel, probability=True,
                             random_state=self.cfg.random_state)
        else:
            raise ValueError(f"Unsupported base model {self.cfg.model_name}")
        self.model.fit(final_x, final_y)
        return self

    def _select(self, x: np.ndarray, y: np.ndarray) -> list[str]:
        n = min(len(self.columns), self.cfg.num_features)
        method = self.cfg.feature_selection.strip().lower()
        if method in {"none", "no feature selection"}:
            return self.columns.copy()
        if "boost" in method:
            # These are non-default methods, but are supported for compatibility
            # with optional historical config rows. Never silently substitute
            # sklearn's RFE for the original project's SHAP-based BoostRFE/RFA.
            try:
                from shaphypetune import BoostRFE, BoostRFA, BoostBoruta
                from xgboost import XGBRegressor
                from lightgbm import LGBMRegressor
            except ImportError as exc:
                raise ImportError("Boost-based selection needs shaphypetune, xgboost, lightgbm") from exc
            terms = self.cfg.feature_selection.split()
            name = terms[0].lower()
            search = {"boostrfe": BoostRFE, "boostrfa": BoostRFA, "boostboruta": BoostBoruta}
            if name not in search:
                raise ValueError(f"Unknown boosted feature selector {name}")
            regressor = XGBRegressor() if len(terms) > 1 and terms[1].upper() == "XGB" else LGBMRegressor()
            sel = search[name](regressor)
            sel.fit(pd.DataFrame(x, columns=self.columns), y)
            indices = np.asarray(sel.ranking_).argsort()[:n]
            return [self.columns[i] for i in indices]
        if "logistic" not in method and method not in {"lr", "coeff"}:
            raise ValueError(f"Unrecognized feature-selection method: {self.cfg.feature_selection}")

        sorting = self.cfg.sorting_metric.strip().lower()
        # Original code used statsmodels.Logit (no added intercept) for feature
        # ranking. Retain that behavior, with explicit fallback on numerical
        # failures for coefficient ranking ONLY.
        try:
            import statsmodels.api as sm
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = sm.Logit(y, x).fit(method="bfgs", maxiter=1000, disp=False)
            values = np.asarray(result.pvalues if "p-value" in sorting or "pvalue" in sorting
                                else np.abs(result.params), dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("Non-finite feature ranking values")
            order = np.argsort(values) if "p-value" in sorting or "pvalue" in sorting else np.argsort(-values)
        except Exception as exc:
            if "p-value" in sorting or "pvalue" in sorting:
                raise RuntimeError("Statsmodels p-value ranking failed; do not silently use coefficients") from exc
            warnings.warn("Statsmodels ranking was unstable; falling back to sklearn LR coefficients")
            selector = LogisticRegression(max_iter=3000, solver="liblinear", random_state=self.cfg.random_state)
            selector.fit(x, y)
            order = np.argsort(-np.abs(selector.coef_[0]))
        return [self.columns[int(i)] for i in order[:n]]

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        z = self.imputer.transform(x[self.columns])[:, self.selected_indices]
        if self.scaler is not None:
            z = self.scaler.transform(z)
        return np.asarray(self.model.predict_proba(z)[:, 1], dtype=float)


def splits_for(y: np.ndarray, groups: np.ndarray, n: int, seed: int, stage: str):
    unique_groups = np.unique(groups)
    if n < 2 or len(unique_groups) < n:
        raise ValueError(f"{stage}: need at least {n} distinct participants for {n} folds")
    splitter = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.zeros(len(y)), y, groups))
    for tr, te in splits:
        if set(groups[tr]) & set(groups[te]):
            raise AssertionError(f"{stage}: participant ID leakage in splitter")
        if len(np.unique(y[tr])) != 2 or len(np.unique(y[te])) != 2:
            raise ValueError(f"{stage}: a fold lacks a class; reduce folds or inspect group balance")
    return splits


def metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "AUROC": float(roc_auc_score(y_true, prob)),
        "Accuracy": float(accuracy_score(y_true, pred)),
        "Sensitivity": float(recall_score(y_true, pred, zero_division=0)),
        "Specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "PPV": float(precision_score(y_true, pred, zero_division=0)),
        "NPV": float(tn / (tn + fn)) if tn + fn else float("nan"),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
    }


def ranking_score(y: np.ndarray, prob: np.ndarray, metric: str) -> float:
    m = metrics(y, prob)
    key = metric.strip().lower()
    names = {"rocauc": "AUROC", "roc auc": "AUROC", "auroc": "AUROC",
             "accuracy": "Accuracy", "sensitivity": "Sensitivity", "specificity": "Specificity",
             "ppv": "PPV", "precision": "PPV", "npv": "NPV", "f1": "F1"}
    if key not in names:
        raise ValueError(f"Unsupported --sorting_metric {metric!r}; use AUROC or Accuracy, etc.")
    return m[names[key]]


def build_meta(y: np.ndarray, groups: np.ndarray, inner_oof: np.ndarray,
               c_values: list[float], meta_folds: int, penalty: str,
               solver: str, max_iter: int, tol: float, seed: int):
    # sklearn 1.8 deprecated the penalty keyword in favor of l1_ratio.
    version = tuple(int(v) for v in sklearn_version.split(".")[:2])
    penalty_arg = ({"l1_ratio": 1.0 if penalty == "l1" else 0.0}
                   if version >= (1, 8) else {"penalty": penalty})
    estimator = LogisticRegression(solver=solver, max_iter=max_iter,
                                   tol=tol, random_state=seed, **penalty_arg)
    if len(c_values) == 1:
        estimator.set_params(C=c_values[0]); return estimator.fit(inner_oof, y), c_values[0]
    cv = splits_for(y, groups, meta_folds, seed + 899, "meta tuning")
    search = GridSearchCV(estimator, {"C": c_values}, scoring="roc_auc", cv=cv,
                          n_jobs=1, refit=True, error_score="raise")
    search.fit(inner_oof, y)
    return search.best_estimator_, float(search.best_params_["C"])


def train_stack(train_df: pd.DataFrame, candidates: list[Candidate], features: list[str],
                n_configs: int, inner_folds: int, meta_folds: int,
                score_metric: str, meta_cs: list[float], meta_penalty: str,
                meta_solver: str, meta_max_iter: int, meta_tol: float,
                seed: int, verbose: bool = True):
    y = train_df["Diagnosis"].to_numpy(dtype=int)
    groups = train_df["ID"].to_numpy()
    splits = splits_for(y, groups, inner_folds, seed, "inner base OOF")
    scores, oofs = [], []
    for ci, cfg in enumerate(candidates):
        oof = np.full(len(y), np.nan, dtype=float)
        for tr, te in splits:
            base = BaseModel(cfg, features).fit(train_df.iloc[tr], y[tr])
            oof[te] = base.predict_proba(train_df.iloc[te])
        if not np.isfinite(oof).all():
            raise AssertionError("Incomplete/non-finite OOF base predictions")
        oofs.append(oof)
        scores.append(ranking_score(y, oof, score_metric))
        if verbose:
            print(f"  Candidate {ci+1:>3}/{len(candidates)} source-row={cfg.source_row}, {score_metric}={scores[-1]:.4f}", flush=True)
    # Config selection never sees the OUTER held-out fold.
    top = sorted(range(len(candidates)), key=lambda j: (-scores[j], j))[:n_configs]
    meta_x = np.stack([oofs[j] for j in top], axis=1)
    meta, best_c = build_meta(y, groups, meta_x, meta_cs, meta_folds,
                              meta_penalty, meta_solver, meta_max_iter, meta_tol, seed)
    full_base = [BaseModel(candidates[j], features).fit(train_df, y) for j in top]
    return meta, full_base, top, scores, best_c


def stack_predict(df: pd.DataFrame, meta, bases: list[BaseModel]) -> np.ndarray:
    z = np.stack([base.predict_proba(df) for base in bases], axis=1)
    return np.asarray(meta.predict_proba(z)[:, 1], dtype=float)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False, default=str) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", default="dataset-cross-validation.csv")
    parser.add_argument("--config_file", default="cross-validation-hist-configs.csv")
    parser.add_argument("--external-csv", help="Optional truly independent external dataset, never used to tune")
    parser.add_argument("--outdir", default="smile_nested_cv_results")
    parser.add_argument("--num_configs", type=int, default=18,
                        help="Select this many base configurations inside EACH outer train split")
    parser.add_argument("--candidate-limit", type=int, default=40,
                        help="Number of randomly chosen CSV candidates to evaluate per outer fold; 0 = all")
    parser.add_argument("--sorting_metric", default="AUROC",
                        help="Score computed from INNER OOF predictions, not CSV's historical score")
    parser.add_argument("--outer-folds", type=int, default=10)
    parser.add_argument("--inner-folds", type=int, default=10)
    parser.add_argument("--meta-folds", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=6424)
    parser.add_argument("--expressions", default="smile", help="Comma separated: smile,disgust,surprise")
    parser.add_argument("--no-au", action="store_true")
    parser.add_argument("--no-mp", action="store_true")
    parser.add_argument("--meta-cs", default="0.01,0.1,1,10,40.36631131136931",
                        help="GridSearchCV C options tuned WITHIN each outer training subset")
    parser.add_argument("--penalty", choices=["l1", "l2"], default="l1")
    parser.add_argument("--solver", choices=["saga", "liblinear", "lbfgs"], default="saga")
    parser.add_argument("--max_iter", type=int, default=3000)
    parser.add_argument("--tol", type=float, default=0.0009070426942042604)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.penalty == "l1" and args.solver == "lbfgs":
        parser.error("lbfgs does not support l1 penalty")
    meta_cs = [float(c) for c in args.meta_cs.split(",")]
    if not meta_cs or any(c <= 0 or not math.isfinite(c) for c in meta_cs):
        parser.error("All meta Cs must be positive finite values")
    if args.num_configs < 1:
        parser.error("--num_configs must be >=1")
    exprs = [x.strip().lower() for x in args.expressions.split(",")]
    if not set(exprs).issubset({"smile", "disgust", "surprise"}):
        parser.error("--expressions must contain only smile,disgust,surprise")

    df = pd.read_csv(args.dataset)
    for field in ("Diagnosis", "ID"):
        if field not in df.columns:
            raise ValueError(f"Dataset must contain {field!r}; grouping requires ID")
    if df[["Diagnosis", "ID"]].isna().any().any():
        raise ValueError("Missing participant IDs or labels are not allowed")
    if not set(df.Diagnosis.unique()).issubset({0, 1}):
        raise ValueError("Diagnosis must be binary: 0=non-PD, 1=PD")
    if (df.groupby("ID")["Diagnosis"].nunique() > 1).any():
        raise ValueError("A participant ID appears with conflicting diagnosis labels")
    features = feature_names(df, exprs, not args.no_au, not args.no_mp)
    candidates = read_candidates(Path(args.config_file), args.candidate_limit, args.random_state)
    if len(candidates) < args.num_configs:
        print(f"[warning] Only {len(candidates)} candidates; using all rather than {args.num_configs}")
    chosen_n = min(args.num_configs, len(candidates))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Development rows={len(df)}, participants={df.ID.nunique()}, features={len(features)}, "
          f"candidate configs={len(candidates)}, stack size={chosen_n}", flush=True)
    print("[important] If configs/meta search space were optimized on THIS ENTIRE dataset before this run, "
          "outer CV retains historical selection bias. Use independent candidates or new external test data.", flush=True)

    all_y = df.Diagnosis.to_numpy(dtype=int)
    all_groups = df.ID.to_numpy()
    outer = splits_for(all_y, all_groups, args.outer_folds, args.random_state, "outer evaluation")
    oof_score = np.full(len(df), np.nan)
    prediction_rows, fold_rows, selection_rows = [], [], []
    for fold, (tr, te) in enumerate(outer, 1):
        print(f"\n[outer {fold}/{len(outer)}] train={len(tr)}, held-out={len(te)}", flush=True)
        train_df, test_df = df.iloc[tr].copy(), df.iloc[te].copy()
        meta, bases, selected, rank_scores, best_c = train_stack(
            train_df, candidates, features, chosen_n, args.inner_folds,
            args.meta_folds, args.sorting_metric, meta_cs,
            args.penalty, args.solver, args.max_iter, args.tol,
            args.random_state + fold * 1009, verbose=not args.quiet,
        )
        probs = stack_predict(test_df, meta, bases)
        oof_score[te] = probs
        fm = metrics(all_y[te], probs)
        fold_rows.append({"fold": fold, "train_rows": len(tr), "test_rows": len(te),
                          "meta_C": best_c, **fm})
        print(f"[outer {fold}] AUROC={fm['AUROC']:.4f} ACC={fm['Accuracy']:.4f} "
              f"SENS={fm['Sensitivity']:.4f}, SPEC={fm['Specificity']:.4f}", flush=True)
        for idx, prob in zip(te, probs):
            prediction_rows.append({"original_index": int(idx), "ID": df.iloc[idx].ID,
                                    "outer_fold": fold, "Diagnosis": int(all_y[idx]),
                                    "PD_probability": float(prob),
                                    "PD_prediction": int(prob >= 0.5)})
        for rank, candidate_i in enumerate(selected, 1):
            selection_rows.append({"outer_fold": fold, "rank": rank,
                                   "source_config_row": candidates[candidate_i].source_row,
                                   "inner_oof_score": rank_scores[candidate_i],
                                   **asdict(candidates[candidate_i])})
    if not np.isfinite(oof_score).all():
        raise AssertionError("Not every development participant received an outer-held-out prediction")
    final_metrics = metrics(all_y, oof_score)
    # If a participant has multiple rows, also report one score per participant.
    # The outer folds are group-disjoint, so this averaging is evaluation-only.
    participant_df = (pd.DataFrame({"ID": df.ID.to_numpy(),
                                    "Diagnosis": all_y, "PD_probability": oof_score})
                      .groupby("ID", as_index=False)
                      .agg(Diagnosis=("Diagnosis", "first"),
                           PD_probability=("PD_probability", "mean")))
    participant_df["PD_prediction"] = (participant_df.PD_probability >= 0.5).astype(int)
    participant_df.to_csv(outdir / "participant_oof_predictions.csv", index=False)
    participant_metrics = metrics(participant_df.Diagnosis.to_numpy(dtype=int),
                                  participant_df.PD_probability.to_numpy(dtype=float))
    pd.DataFrame(prediction_rows).sort_values("original_index").to_csv(outdir / "outer_oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(outdir / "outer_fold_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(outdir / "selected_configs_by_outer_fold.csv", index=False)
    write_json(outdir / "summary.json", {
        "method": "nested participant-grouped CV; inner OOF probability stacking",
        "n_rows": len(df), "n_participants": int(df.ID.nunique()),
        "n_candidates": len(candidates), "n_base_models": chosen_n,
        "outer_folds": args.outer_folds, "inner_folds": args.inner_folds,
        "meta_tuning_folds": args.meta_folds, "row_level_metrics": final_metrics,
        "participant_level_metrics": participant_metrics,
        "warning": "Prior candidate generation or tuning on complete evaluation data is not corrected by CV alone",
    })
    print("\nPooled OUTER held-out CV metrics (row level):")
    print(json.dumps(final_metrics, indent=2))
    print("Participant-aggregated OUTER held-out CV metrics:")
    print(json.dumps(participant_metrics, indent=2))

    if args.external_csv:
        external = pd.read_csv(args.external_csv)
        if not set(features).issubset(external.columns):
            raise ValueError("External CSV is missing selected facial feature columns")
        if "ID" in external.columns and set(df.ID) & set(external.ID):
            raise ValueError("External test set overlaps development participant IDs")
        if "Diagnosis" in external.columns and not set(external.Diagnosis.dropna().unique()).issubset({0,1}):
            raise ValueError("External Diagnosis labels must be binary 0/1")
        print("\n[external] Fit final stack using all DEVELOPMENT data; external untouched", flush=True)
        meta, bases, selected, _, _ = train_stack(
            df, candidates, features, chosen_n, args.inner_folds,
            args.meta_folds, args.sorting_metric, meta_cs,
            args.penalty, args.solver, args.max_iter, args.tol,
            args.random_state + 99999, verbose=not args.quiet,
        )
        p = stack_predict(external, meta, bases)
        ext_out = pd.DataFrame({"PD_probability": p, "PD_prediction": (p >= 0.5).astype(int)})
        for col in ("ID", "Diagnosis", "Pipeline"):
            if col in external:
                ext_out.insert(0, col, external[col].values)
        ext_out.to_csv(outdir / "external_predictions.csv", index=False)
        if "Diagnosis" in external.columns and external.Diagnosis.notna().all():
            ext_metrics = metrics(external.Diagnosis.to_numpy(dtype=int), p)
            write_json(outdir / "external_metrics.json", ext_metrics)
            print("External metrics:", json.dumps(ext_metrics, indent=2))

    print(f"\nResults written to {outdir.resolve()}")


if __name__ == "__main__":
    main()
