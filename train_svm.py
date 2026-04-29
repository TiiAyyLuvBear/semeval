"""
train_svm.py — Raw-Text SVM Specialist (TF-IDF + LinearSVC)
===========================================================
Khác với train_tfidf.py (chạy trên syntax pattern), script này 
chạy trực tiếp trên mã nguồn gốc (raw code) để bắt các "chữ ký từ vựng"
(vocabulary signatures). Nhanh hơn CodeBERT rất nhiều.
"""
import logging
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from config import CFG
from metrics import optimize_threshold, print_eval_report

logger = logging.getLogger(__name__)

def run_svm(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    cfg: CFG,
    test_df:  pd.DataFrame | None = None,
) -> np.ndarray:
    """
    Train TF-IDF + LinearSVC trên raw code.

    Args:
        train_df: DataFrame có cột ['code', 'label']
        val_df:   DataFrame có cột ['code', 'label']
        cfg:      CFG instance
        test_df:  DataFrame có cột ['code'] (không có label)

    Returns:
        val_proba (xác suất AI)
    """
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── 1. Chuẩn bị dữ liệu ───────────────────────────────────────────────
    logger.info("Sử dụng raw code để huấn luyện SVM...")
    train_text = train_df["code"].fillna("").astype(str)
    y_train    = train_df["label"].values

    val_text   = val_df["code"].fillna("").astype(str)
    y_val      = val_df["label"].values

    # ── 2. Huấn luyện TF-IDF ──────────────────────────────────────────────
    # Sử dụng word n-grams để bắt các token, từ khóa hệ thống và tên biến
    logger.info("Huấn luyện TfidfVectorizer (word n-grams 1-3)...")
    vectorizer = TfidfVectorizer(
        analyzer='word', 
        ngram_range=(1, 3), 
        max_features=50000,
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(train_text)
    X_val_vec   = vectorizer.transform(val_text)

    # ── 3. Huấn luyện LinearSVC ───────────────────────────────────────────
    logger.info("Huấn luyện LinearSVC (Calibrated)...")
    # LinearSVC nhanh và phù hợp nhất cho bài toán TF-IDF nhiều chiều
    base_svc = LinearSVC(C=0.5, random_state=cfg.seed, max_iter=2000, dual=True)
    
    # CalibratedClassifierCV giúp chuyển output của SVM thành xác suất [0, 1]
    model = CalibratedClassifierCV(base_svc, cv=5, method='isotonic')
    model.fit(X_train_vec, y_train)

    train_proba = model.predict_proba(X_train_vec)[:, 1]
    val_proba   = model.predict_proba(X_val_vec)[:, 1]

    # ── 4. Đánh giá Val ───────────────────────────────────────────────────
    tau, val_f1 = optimize_threshold(val_proba, y_val)
    logger.info(f"Raw-Text SVM Model: τ*={tau:.3f}  →  Val Macro F1={val_f1:.4f}")

    y_pred_val = (val_proba >= tau).astype(int)
    print_eval_report(y_val, y_pred_val, val_proba, "Val (Raw-Text SVM)")

    # ── 5. Lưu model ──────────────────────────────────────────────────────
    model_path = os.path.join(cfg.output_dir, "svm_model.pkl")
    joblib.dump({
        "vectorizer": vectorizer,
        "model":      model,
        "threshold":  tau,
    }, model_path)
    logger.info(f"SVM model → {model_path}")

    np.save(os.path.join(cfg.output_dir, "val_proba_svm.npy"), val_proba)

    # ── 6. Test inference ─────────────────────────────────────────────────
    if test_df is not None:
        logger.info("Inference SVM trên test...")
        test_text = test_df["code"].fillna("").astype(str)
        X_test_vec = vectorizer.transform(test_text)
        test_proba = model.predict_proba(X_test_vec)[:, 1]
        
        np.save(os.path.join(cfg.output_dir, "test_proba_svm.npy"), test_proba)

        y_test = (test_proba >= tau).astype(int)
        test_ids = test_df["ID"] if "ID" in test_df.columns else range(len(test_df))
        out = cfg.submission_out.replace(".csv", "_svm.csv")
        pd.DataFrame({"ID": test_ids, "label": y_test}).to_csv(out, index=False)
        logger.info(f"Submission SVM → {out}")

    return val_proba


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
    if args.test_data and os.path.exists(args.test_data):
        raw_test = pd.read_parquet(args.test_data)
        cols = ["code", "ID"] if "ID" in raw_test.columns else ["code"]
        test_df = raw_test[cols].reset_index(drop=True)

    run_svm(train_df, val_df, cfg, test_df=test_df)
