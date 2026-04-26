"""
train_gbm.py — Gradient-Boosted Ensemble (XGBoost + LightGBM + CatBoost)
=========================================================================
Language-robust: loại bỏ features bị OOD shift (|d| > 1.0) trước khi train.

Sử dụng:
    from train_gbm import run_gbm
    best_tau, val_proba = run_gbm(train_feat_path, val_feat_path, cfg)
"""
import logging
import os
import joblib
import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics import f1_score, accuracy_score
import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logging.warning("lightgbm không có, bỏ qua")

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    logging.warning("catboost không có, bỏ qua")

from config import CFG
from metrics import optimize_threshold, quantile_threshold, print_eval_report

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Feature preparation
# =============================================================================
def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm interaction features: entropy×compression, tokens/line, ..."""
    if "shannon_entropy" in df and "compression_ratio" in df:
        df["entropy_x_compression"] = df["shannon_entropy"] * df["compression_ratio"]
    if "token_count" in df and "line_count" in df:
        df["tokens_per_line"] = df["token_count"] / df["line_count"].replace(0, 1)
    if "comment_ratio" in df and "line_count" in df:
        df["comment_density"] = df["comment_ratio"] * df["line_count"]
    if "avg_line_len" in df and "line_length_std" in df:
        df["line_len_cv"] = df["line_length_std"] / df["avg_line_len"].replace(0, 1)
    if "indent_consistency" in df and "max_indent" in df:
        df["indent_ratio"] = df["indent_consistency"] / df["max_indent"].replace(0, 1)
    return df


def _detect_language(code: str) -> int:
    """Heuristic: 0=Python, 1=C++, 2=Java"""
    import re
    if re.search(r'\bdef\s+\w+\s*\(', code) or "print(" in code:
        return 0
    if re.search(r'#include\s*<', code) or "std::" in code:
        return 1
    if re.search(r'\bpublic\s+class\b', code) or "System.out" in code:
        return 2
    return 0


def prep_features(
    df: pd.DataFrame,
    lang_shifted: tuple,
    train_columns: list | None = None,
) -> pd.DataFrame:
    """
    Drop metadata, OOD-shifted features, align columns với train nếu có.
    """
    DROP_META = {"label", "ID", "code", "generator", "language", "__lang"}
    drop = list(DROP_META.intersection(df.columns))
    drop += [c for c in lang_shifted if c in df.columns]

    X = df.drop(columns=drop, errors="ignore")
    X = X.replace([float("inf"), float("-inf")], float("nan")).fillna(0)

    if train_columns is not None:
        X = X.reindex(columns=train_columns, fill_value=0)
    return X


# =============================================================================
# 2. Train single GBM models
# =============================================================================
def _train_xgb(X_train, y_train, X_val, y_val, cfg: CFG):
    model = xgb.XGBClassifier(
        n_estimators=cfg.gbm_n_estimators,
        learning_rate=cfg.gbm_lr,
        max_depth=cfg.gbm_max_depth,
        subsample=cfg.gbm_subsample,
        colsample_bytree=cfg.gbm_colsample,
        min_child_weight=5,
        reg_alpha=0.5,
        reg_lambda=3.0,
        gamma=0.3,
        random_state=cfg.seed,
        tree_method="hist",
        n_jobs=-1,
        early_stopping_rounds=100,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
    return model


def _train_lgb(X_train, y_train, X_val, y_val, cfg: CFG):
    model = lgb.LGBMClassifier(
        n_estimators=cfg.gbm_n_estimators,
        learning_rate=cfg.gbm_lr,
        max_depth=cfg.gbm_max_depth,
        num_leaves=63,
        min_child_samples=30,
        subsample=cfg.gbm_subsample,
        colsample_bytree=cfg.gbm_colsample,
        reg_alpha=0.5,
        reg_lambda=3.0,
        random_state=cfg.seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(100),
        ],
    )
    return model


def _train_cat(X_train, y_train, X_val, y_val, cfg: CFG):
    model = CatBoostClassifier(
        iterations=cfg.gbm_n_estimators,
        learning_rate=cfg.gbm_lr,
        depth=cfg.gbm_max_depth,
        l2_leaf_reg=3.0,
        random_seed=cfg.seed,
        verbose=100,
        early_stopping_rounds=100,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    return model


# =============================================================================
# 3. Ensemble weight optimization
# =============================================================================
def _optimize_weights(
    proba_dict: dict,
    y_val: np.ndarray,
    weight_options: list = [0.5, 1.0, 1.5, 2.0, 2.5],
):
    """Grid search trọng số tốt nhất cho soft-voting."""
    names = list(proba_dict.keys())
    probas = [proba_dict[k] for k in names]

    best_f1, best_weights, best_tau = 0.0, [1.0] * len(names), 0.5
    for combo in product(weight_options, repeat=len(names)):
        total_w = sum(combo)
        avg_p = sum(p * w for p, w in zip(probas, combo)) / total_w
        tau, f1 = optimize_threshold(avg_p, y_val)
        if f1 > best_f1:
            best_f1, best_weights, best_tau = f1, list(combo), tau

    logger.info("Trọng số tốt nhất:")
    for n, w in zip(names, best_weights):
        logger.info(f"  {n}: {w:.1f}")
    logger.info(f"  τ*={best_tau:.4f}  →  Val Macro F1={best_f1:.4f}")
    return best_weights, best_tau, best_f1


# =============================================================================
# 4. Main function
# =============================================================================
def run_gbm(
    train_feat_path: str,
    val_feat_path:   str,
    cfg: CFG,
    train_meta_path: str | None = None,
    val_meta_path:   str | None = None,
    test_feat_path:  str | None = None,
    test_data_path:  str | None = None,
) -> tuple[float, np.ndarray]:
    """
    Train gradient-boosted language-robust ensemble.

    Returns:
        (best_tau, val_proba)  — để dùng cho blending ensemble
    """
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── 1. Load features ──────────────────────────────────────────────────
    logger.info("Tải train features...")
    train = pd.read_parquet(train_feat_path)
    y_train = train["label"]

    # Thêm language feature
    if train_meta_path and os.path.exists(train_meta_path):
        meta = pd.read_parquet(train_meta_path, columns=["language"])
        train["language"] = meta["language"].values
        lang_map = {"Python": 0, "C++": 1, "Java": 2}
        train["lang_id"] = train["language"].map(lang_map).fillna(0).astype(int)

    train = _add_interaction_features(train)
    X_train = prep_features(train, cfg.lang_shifted_features)
    train_columns = X_train.columns.tolist()
    logger.info(f"Train: {X_train.shape} | Đã drop {len(cfg.lang_shifted_features)} shifted features")

    logger.info("Tải val features...")
    val = pd.read_parquet(val_feat_path)
    y_val = val["label"].values

    if val_meta_path and os.path.exists(val_meta_path):
        meta = pd.read_parquet(val_meta_path, columns=["language"])
        val["language"] = meta["language"].values
        val["lang_id"] = val["language"].map({"Python":0,"C++":1,"Java":2}).fillna(0).astype(int)

    val = _add_interaction_features(val)
    X_val = prep_features(val, cfg.lang_shifted_features, train_columns)
    logger.info(f"Val: {X_val.shape}")

    # ── 2. Train models ───────────────────────────────────────────────────
    models, val_probas = {}, {}

    logger.info("[1/3] Training XGBoost...")
    xgb_m = _train_xgb(X_train, y_train, X_val, y_val, cfg)
    models["xgb"] = xgb_m
    val_probas["xgb"] = xgb_m.predict_proba(X_val)[:, 1]

    if HAS_LGB:
        logger.info("[2/3] Training LightGBM...")
        lgb_m = _train_lgb(X_train, y_train, X_val, y_val, cfg)
        models["lgb"] = lgb_m
        val_probas["lgb"] = lgb_m.predict_proba(X_val)[:, 1]

    if HAS_CAT:
        logger.info("[3/3] Training CatBoost...")
        cat_m = _train_cat(X_train, y_train, X_val, y_val, cfg)
        models["cat"] = cat_m
        val_probas["cat"] = cat_m.predict_proba(X_val)[:, 1]

    # ── 3. Optimize ensemble weights ──────────────────────────────────────
    logger.info("Tối ưu ensemble weights...")
    best_weights, best_tau, best_f1 = _optimize_weights(val_probas, y_val)

    names = list(val_probas.keys())
    total_w = sum(best_weights)
    val_proba = sum(
        val_probas[n] * w for n, w in zip(names, best_weights)
    ) / total_w
    y_pred = (val_proba >= best_tau).astype(int)

    print_eval_report(y_val, y_pred, val_proba, "Val (GBM Ensemble)")

    # Feature importance
    imp = xgb_m.get_booster().get_score(importance_type="gain")
    top15 = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:15]
    logger.info("Top 15 features (XGBoost gain):")
    for feat, gain in top15:
        logger.info(f"  {feat}: {gain:.1f}")

    # ── 4. Save ───────────────────────────────────────────────────────────
    model_path = os.path.join(cfg.output_dir, "gbm_ensemble.pkl")
    joblib.dump({
        "models":           models,
        "weights":          best_weights,
        "model_names":      names,
        "threshold":        best_tau,
        "train_columns":    train_columns,
        "dropped_features": list(cfg.lang_shifted_features),
    }, model_path)
    logger.info(f"GBM model → {model_path}")

    val_proba_path = os.path.join(cfg.output_dir, "val_proba_gbm.npy")
    np.save(val_proba_path, val_proba)

    # ── 5. Test inference (tuỳ chọn) ─────────────────────────────────────
    if test_feat_path and os.path.exists(test_feat_path):
        logger.info("Inference GBM trên test...")
        test_df = pd.read_parquet(test_feat_path)

        if test_data_path and os.path.exists(test_data_path):
            test_raw = pd.read_parquet(test_data_path, columns=["code"])
            test_df["lang_id"] = test_raw["code"].apply(_detect_language)
        else:
            test_df["lang_id"] = 0

        test_df = _add_interaction_features(test_df)
        X_test = prep_features(test_df, cfg.lang_shifted_features, train_columns)

        test_proba_list = [
            models[n].predict_proba(X_test)[:, 1] for n in names
        ]
        test_proba = sum(
            p * w for p, w in zip(test_proba_list, best_weights)
        ) / total_w
        np.save(os.path.join(cfg.output_dir, "test_proba_gbm.npy"), test_proba)

        y_test = (test_proba >= best_tau).astype(int)
        test_ids = test_df["ID"] if "ID" in test_df.columns else range(len(test_df))
        out = cfg.submission_out.replace(".csv", "_gbm.csv")
        pd.DataFrame({"ID": test_ids, "label": y_test}).to_csv(out, index=False)
        logger.info(f"Submission GBM → {out}")

    return best_tau, val_proba


# =============================================================================
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("--train_feat",  required=True)
    p.add_argument("--val_feat",    required=True)
    p.add_argument("--test_feat",   default=None)
    p.add_argument("--output_dir",  default=CFG.output_dir)
    args = p.parse_args()

    cfg = CFG()
    cfg.output_dir = args.output_dir
    run_gbm(args.train_feat, args.val_feat, cfg, test_feat_path=args.test_feat)
