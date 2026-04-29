"""
train_tfidf.py — Textual Specialist (TF-IDF + Logistic Regression)
===================================================================
Trích xuất cú pháp (syntax pattern) từ mã nguồn thô và sử dụng TF-IDF
(character N-grams) để huấn luyện một mô hình Logistic Regression.
"""
import logging
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from config import CFG
from data_utils import get_syntax_pattern
from metrics import optimize_threshold, print_eval_report

logger = logging.getLogger(__name__)


def build_syntax_corpus(codes: pd.Series, show_progress: bool = True) -> list[str]:
    """Áp dụng get_syntax_pattern lên từng dòng code."""
    try:
        from tqdm import tqdm
        it = tqdm(codes, desc="Syntax patterns") if show_progress else codes
    except ImportError:
        it = codes
    
    return [get_syntax_pattern(str(c)) for c in it]


def run_tfidf(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    cfg: CFG,
    test_df:  pd.DataFrame | None = None,
) -> np.ndarray:
    """
    Train TF-IDF + LogisticRegression.

    Args:
        train_df: DataFrame có cột ['code', 'label']
        val_df:   DataFrame có cột ['code', 'label']
        cfg:      CFG instance
        test_df:  DataFrame có cột ['code'] (không có label)

    Returns:
        val_proba (xác suất AI)
    """
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── 1. Tạo Syntax Corpus ──────────────────────────────────────────────
    logger.info("Chuyển đổi code sang syntax pattern (train)...")
    train_syntax = build_syntax_corpus(train_df["code"])
    y_train      = train_df["label"].values

    logger.info("Chuyển đổi code sang syntax pattern (val)...")
    val_syntax   = build_syntax_corpus(val_df["code"])
    y_val        = val_df["label"].values

    # ── 2. Huấn luyện TF-IDF ──────────────────────────────────────────────
    logger.info("Huấn luyện TfidfVectorizer (char n-gram: 3-6)...")
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 6), max_features=30000)
    X_train_vec = vectorizer.fit_transform(train_syntax)
    X_val_vec   = vectorizer.transform(val_syntax)

    # ── 3. Huấn luyện Logistic Regression ─────────────────────────────────
    logger.info("Huấn luyện Logistic Regression...")
    model = LogisticRegression(C=0.1, max_iter=1000, random_state=cfg.seed, n_jobs=-1)
    model.fit(X_train_vec, y_train)

    train_proba = model.predict_proba(X_train_vec)[:, 1]
    val_proba   = model.predict_proba(X_val_vec)[:, 1]

    # ── 4. Đánh giá Val ───────────────────────────────────────────────────
    tau, val_f1 = optimize_threshold(val_proba, y_val)
    logger.info(f"TF-IDF Model: τ*={tau:.3f}  →  Val Macro F1={val_f1:.4f}")

    y_pred_val = (val_proba >= tau).astype(int)
    print_eval_report(y_val, y_pred_val, val_proba, "Val (TF-IDF)")

    # ── 5. Lưu model ──────────────────────────────────────────────────────
    model_path = os.path.join(cfg.output_dir, "tfidf_model.pkl")
    joblib.dump({
        "vectorizer": vectorizer,
        "model":      model,
        "threshold":  tau,
    }, model_path)
    logger.info(f"TF-IDF model → {model_path}")

    np.save(os.path.join(cfg.output_dir, "val_proba_tfidf.npy"), val_proba)

    # ── 6. Test inference ─────────────────────────────────────────────────
    if test_df is not None:
        logger.info("Chuyển đổi code sang syntax pattern (test)...")
        test_syntax = build_syntax_corpus(test_df["code"])
        
        logger.info("Inference TF-IDF trên test...")
        X_test_vec = vectorizer.transform(test_syntax)
        test_proba = model.predict_proba(X_test_vec)[:, 1]
        
        np.save(os.path.join(cfg.output_dir, "test_proba_tfidf.npy"), test_proba)

        y_test = (test_proba >= tau).astype(int)
        test_ids = test_df["ID"] if "ID" in test_df.columns else range(len(test_df))
        out = cfg.submission_out.replace(".csv", "_tfidf.csv")
        pd.DataFrame({"ID": test_ids, "label": y_test}).to_csv(out, index=False)
        logger.info(f"Submission TF-IDF → {out}")

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

    run_tfidf(train_df, val_df, cfg, test_df=test_df)
