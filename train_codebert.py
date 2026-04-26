"""
train_codebert.py — Fine-tune CodeBERT bằng HuggingFace Trainer
================================================================
Sử dụng:
    from train_codebert import run_codebert
    tau, val_proba = run_codebert(cfg)

Hoặc chạy trực tiếp:
    python train_codebert.py
"""
import logging
import os
import numpy as np
import torch
from scipy.special import softmax as sp_softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import CFG
from data_utils import load_dataframe, build_hf_dataset, load_test_dataset
from trainer_utils import build_trainer
from metrics import optimize_threshold, quantile_threshold, print_eval_report

logger = logging.getLogger(__name__)


def run_codebert(cfg: CFG) -> tuple[float, np.ndarray]:
    """
    Fine-tune CodeBERT → trả về (best_tau, val_proba).

    best_tau:  threshold tốt nhất trên toàn bộ val set
    val_proba: xác suất AI trên val set (dùng cho ensemble blending)

    Side effects:
        - Lưu model tốt nhất tại cfg.output_dir
        - Lưu val_proba.npy và test_proba.npy tại cfg.output_dir
        - Ghi submission CSV nếu cfg.test_data tồn tại
    """
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── 1. Tokenizer & Model ───────────────────────────────────────────────
    logger.info(f"Tải model: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_labels,
        ignore_mismatched_sizes=True,
    )

    # ── 2. Dataset ─────────────────────────────────────────────────────────
    train_df = load_dataframe(cfg.train_data, cfg.max_train, "Train", cfg.seed)
    val_df   = load_dataframe(cfg.val_data,   cfg.max_val,   "Val",   cfg.seed)

    train_ds = build_hf_dataset(train_df, tokenizer, cfg.max_length)
    val_ds   = build_hf_dataset(val_df,   tokenizer, cfg.max_length)

    # ── 3. Build Trainer ───────────────────────────────────────────────────
    trainer = build_trainer(model, tokenizer, train_ds, val_ds, cfg)

    # ── 4. Train ───────────────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("BẮT ĐẦU TRAINING CODEBERT")
    logger.info("═" * 60)
    train_result = trainer.train()
    logger.info(f"  Thời gian: {train_result.metrics['train_runtime']:.1f}s")
    logger.info(f"  Samples/s: {train_result.metrics['train_samples_per_second']:.1f}")

    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    logger.info(f"Model lưu tại: {cfg.output_dir}")

    # ── 5. Evaluate toàn bộ val set ────────────────────────────────────────
    logger.info("Evaluate toàn bộ validation set...")
    full_val_df = load_dataframe(cfg.val_data, None, "Full-Val", cfg.seed)
    full_val_ds = build_hf_dataset(full_val_df, tokenizer, cfg.max_length)

    preds_out  = trainer.predict(full_val_ds)
    val_proba  = sp_softmax(preds_out.predictions, axis=1)[:, 1]
    y_true     = preds_out.label_ids
    y_pred_05  = np.argmax(preds_out.predictions, axis=1)

    # Tìm threshold tốt nhất
    best_tau, best_f1 = optimize_threshold(val_proba, y_true)
    y_pred_best = (val_proba >= best_tau).astype(int)

    print_eval_report(y_true, y_pred_best, val_proba, split_name="Validation")
    logger.info(f"Best τ={best_tau:.3f}  →  Macro F1={best_f1:.4f}")

    # Lưu val probabilities
    val_proba_path = os.path.join(cfg.output_dir, "val_proba_codebert.npy")
    np.save(val_proba_path, val_proba)
    logger.info(f"Val proba → {val_proba_path}")

    # ── 6. Inference trên test set (tuỳ chọn) ─────────────────────────────
    if cfg.test_data and os.path.exists(cfg.test_data):
        logger.info("═" * 60)
        logger.info(f"INFERENCE: {cfg.test_data}")
        logger.info("═" * 60)

        test_ds, test_ids = load_test_dataset(cfg.test_data, tokenizer, cfg.max_length)
        test_preds   = trainer.predict(test_ds)
        test_proba   = sp_softmax(test_preds.predictions, axis=1)[:, 1]

        # Lưu test probabilities (dùng cho ensemble blending)
        test_proba_path = os.path.join(cfg.output_dir, "test_proba_codebert.npy")
        np.save(test_proba_path, test_proba)
        logger.info(f"Test proba → {test_proba_path}")

        # Submission với best_tau từ val
        y_test_best = (test_proba >= best_tau).astype(int)

        # Safety net: quantile threshold
        q_tau = quantile_threshold(test_proba, ai_ratio=0.52)
        y_test_q  = (test_proba >= q_tau).astype(int)

        import pandas as pd
        pd.DataFrame({"ID": test_ids, "label": y_test_best}).to_csv(
            cfg.submission_out, index=False
        )
        pd.DataFrame({"ID": test_ids, "label": y_test_q}).to_csv(
            cfg.submission_out.replace(".csv", "_qthresh.csv"), index=False
        )
        logger.info(
            f"Submission (τ={best_tau:.3f}): {cfg.submission_out}\n"
            f"Submission (q-τ={q_tau:.3f}): *_qthresh.csv"
        )
        logger.info(
            f"Test dist (best_tau): Human={(y_test_best==0).sum()}, AI={(y_test_best==1).sum()}"
        )

    return best_tau, val_proba


# =============================================================================
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("--train_data",  default=CFG.train_data)
    p.add_argument("--val_data",    default=CFG.val_data)
    p.add_argument("--test_data",   default=CFG.test_data)
    p.add_argument("--model_name",  default=CFG.model_name)
    p.add_argument("--output_dir",  default=CFG.output_dir)
    p.add_argument("--epochs",      type=int,   default=CFG.epochs)
    p.add_argument("--batch_size",  type=int,   default=CFG.batch_size)
    p.add_argument("--lr",          type=float, default=CFG.lr)
    p.add_argument("--max_train",   type=int,   default=CFG.max_train)
    p.add_argument("--use_focal",   action="store_true")
    args = p.parse_args()

    cfg = CFG()
    for k, v in vars(args).items():
        if v is not None:
            setattr(cfg, k, v)

    run_codebert(cfg)
