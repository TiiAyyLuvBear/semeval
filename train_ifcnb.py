"""
train_ifcnb.py — IsolationForest + ComplementNaiveBayes Hybrid
==============================================================
Dùng 20 style-only features (ít bị OOD shift nhất).
IsolationForest detect code "quá hoàn hảo" như outlier.
ComplementNB phân loại dựa trên phân phối style.

Sử dụng:
    from train_ifcnb import run_ifcnb
    val_proba = run_ifcnb(train_df, val_df, cfg)
"""
import logging
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

from config import CFG
from feature_extractor import extract_style_features
from metrics import optimize_threshold, print_eval_report

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Trích xuất 20 style features từ raw code
# =============================================================================
def build_style_matrix(codes: pd.Series, show_progress: bool = True) -> pd.DataFrame:
    """
    Áp dụng extract_style_features() lên mỗi code string.
    Returns DataFrame (n_samples × 20).
    """
    try:
        from tqdm import tqdm
        it = tqdm(codes.items(), total=len(codes), desc="Style features") if show_progress else codes.items()
    except ImportError:
        it = codes.items()

    records = []
    for _, code in it:
        try:
            records.append(extract_style_features(str(code)))
        except Exception:
            records.append({k: 0.0 for k in extract_style_features("")})
    return pd.DataFrame(records, index=codes.index)


# =============================================================================
# 2. IsolationForest score → normalize → AI signal
# =============================================================================
def _if_score(if_model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """
    IsolationForest.score_samples trả về giá trị âm (càng âm = càng outlier).
    Chuyển thành xác suất AI ∈ [0,1]:
      - AI code có style "hoàn hảo" → outlier → score thấp → proba cao
    """
    raw = if_model.score_samples(X)          # shape (N,), khoảng [-0.7, 0]
    # Đảo dấu: càng outlier → điểm càng cao
    flipped = -raw
    # Min-max normalize về [0, 1]
    mn, mx = flipped.min(), flipped.max()
    if mx - mn < 1e-10:
        return np.full(len(X), 0.5)
    return (flipped - mn) / (mx - mn)


# =============================================================================
# 3. Main function
# =============================================================================
def run_ifcnb(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    cfg: CFG,
    test_df:  pd.DataFrame | None = None,
) -> np.ndarray:
    """
    Train IsolationForest + ComplementNB trên 20 style features.

    Args:
        train_df: DataFrame có cột ['code', 'label']
        val_df:   DataFrame có cột ['code', 'label']
        cfg:      CFG instance
        test_df:  DataFrame có cột ['code'] (không có label)

    Returns:
        val_proba (xác suất AI) — để blend vào ensemble
    """
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── 1. Trích xuất style features ──────────────────────────────────────
    logger.info("Trích xuất 20 style features (train)...")
    X_train_raw = build_style_matrix(train_df["code"])
    y_train     = train_df["label"].values

    logger.info("Trích xuất 20 style features (val)...")
    X_val_raw   = build_style_matrix(val_df["code"])
    y_val       = val_df["label"].values

    # ── 2. Scale về [0,1] cho ComplementNB (cần giá trị không âm) ─────────
    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val   = scaler.transform(X_val_raw)

    # ── 3. IsolationForest ─────────────────────────────────────────────────
    logger.info(f"Training IsolationForest (n={cfg.if_n_estimators}, contamination={cfg.if_contamination})...")
    if_model = IsolationForest(
        n_estimators=cfg.if_n_estimators,
        contamination=cfg.if_contamination,
        random_state=cfg.seed,
        n_jobs=-1,
    )
    if_model.fit(X_train)

    if_proba_train = _if_score(if_model, X_train)
    if_proba_val   = _if_score(if_model, X_val)

    # ── 4. ComplementNB ────────────────────────────────────────────────────
    logger.info("Training ComplementNB...")
    cnb = ComplementNB(alpha=1.0)
    cnb.fit(X_train, y_train)

    cnb_proba_train = cnb.predict_proba(X_train)[:, 1]
    cnb_proba_val   = cnb.predict_proba(X_val)[:, 1]

    # ── 5. Blend IF + CNB ─────────────────────────────────────────────────
    # Grid search trọng số tốt nhất
    best_f1, best_alpha, best_tau = 0.0, 0.5, 0.5
    for alpha in np.arange(0.0, 1.01, 0.1):
        blend = alpha * if_proba_val + (1 - alpha) * cnb_proba_val
        tau, f1 = optimize_threshold(blend, y_val)
        if f1 > best_f1:
            best_f1, best_alpha, best_tau = f1, alpha, tau

    logger.info(f"IF+CNB blend: α(IF)={best_alpha:.1f}, α(CNB)={1-best_alpha:.1f}")
    logger.info(f"  τ*={best_tau:.3f}  →  Val Macro F1={best_f1:.4f}")

    val_proba  = best_alpha * if_proba_val + (1 - best_alpha) * cnb_proba_val
    y_pred_val = (val_proba >= best_tau).astype(int)
    print_eval_report(y_val, y_pred_val, val_proba, "Val (IF+CNB)")

    # ── 6. Save ───────────────────────────────────────────────────────────
    model_path = os.path.join(cfg.output_dir, "ifcnb_model.pkl")
    joblib.dump({
        "if_model":   if_model,
        "cnb":        cnb,
        "scaler":     scaler,
        "alpha_if":   best_alpha,
        "threshold":  best_tau,
    }, model_path)
    logger.info(f"IF+CNB model → {model_path}")

    np.save(os.path.join(cfg.output_dir, "val_proba_ifcnb.npy"), val_proba)

    # ── 7. Test inference ─────────────────────────────────────────────────
    if test_df is not None:
        logger.info("Inference IF+CNB trên test...")
        X_test_raw  = build_style_matrix(test_df["code"])
        X_test      = scaler.transform(X_test_raw)

        if_proba_test  = _if_score(if_model, X_test)
        cnb_proba_test = cnb.predict_proba(X_test)[:, 1]
        test_proba     = best_alpha * if_proba_test + (1 - best_alpha) * cnb_proba_test

        np.save(os.path.join(cfg.output_dir, "test_proba_ifcnb.npy"), test_proba)

        y_test = (test_proba >= best_tau).astype(int)
        test_ids = test_df["ID"] if "ID" in test_df.columns else range(len(test_df))
        out = cfg.submission_out.replace(".csv", "_ifcnb.csv")
        pd.DataFrame({"ID": test_ids, "label": y_test}).to_csv(out, index=False)
        logger.info(f"Submission IF+CNB → {out}")

    return val_proba


# =============================================================================
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("--train_data", default=CFG.train_data)
    p.add_argument("--val_data",   default=CFG.val_data)
    p.add_argument("--test_data",  default=CFG.test_data)
    p.add_argument("--output_dir", default=CFG.output_dir)
    args = p.parse_args()

    from data_utils import load_dataframe
    cfg = CFG()
    cfg.output_dir = args.output_dir

    train_df = load_dataframe(args.train_data, cfg.max_train, "Train")
    val_df   = load_dataframe(args.val_data,   cfg.max_val,   "Val")
    test_df  = None
    if args.test_data:
        test_df = pd.read_parquet(args.test_data)[["code"]].reset_index(drop=True)

    run_ifcnb(train_df, val_df, cfg, test_df=test_df)
