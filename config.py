"""
config.py — Cấu hình tập trung, tự nhận diện môi trường Kaggle / Colab / Local
"""
import os
import torch
from dataclasses import dataclass, field
from typing import Optional


def _detect_env() -> str:
    if os.path.exists("/kaggle/working"):
        return "kaggle"
    if "COLAB_GPU" in os.environ or os.path.exists("/content"):
        return "colab"
    return "local"


ENV = _detect_env()

# ── Kaggle dataset slug — chỉnh thành tên dataset của bạn ─────────────────
# Ví dụ: competition data ở /kaggle/input/semeval2026task13/
KAGGLE_INPUT = "/kaggle/input/competitions/sem-eval-2026-task-13-subtask-a/Task_A"   # <- đổi slug này


@dataclass
class CFG:
    # ── Đường dẫn — tự map theo môi trường ────────────────────────────────
    train_data:     str = (
        f"{KAGGLE_INPUT}/train.parquet"      if ENV == "kaggle"
        else "/content/data/train.parquet"   if ENV == "colab"
        else "data/train.parquet"
    )
    val_data:       str = (
        f"{KAGGLE_INPUT}/validation.parquet" if ENV == "kaggle"
        else "/content/data/validation.parquet" if ENV == "colab"
        else "data/validation.parquet"
    )
    test_data: Optional[str] = (
        f"{KAGGLE_INPUT}/test.parquet"       if ENV == "kaggle"
        else "/content/data/test.parquet"    if ENV == "colab"
        else "data/test.parquet"
    )
    output_dir:     str = (
        "/kaggle/working/outputs"  if ENV == "kaggle"
        else "/content/outputs"    if ENV == "colab"
        else "outputs"
    )
    submission_out: str = (
        "/kaggle/working/submission.csv" if ENV == "kaggle"
        else "/content/submission.csv"   if ENV == "colab"
        else "outputs/submission.csv"
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model_name:   str  = "microsoft/codebert-base"
    max_length:   int  = 256
    num_labels:   int  = 2

    # ── Sampling ───────────────────────────────────────────────────────────
    max_train: Optional[int] = 100_000
    max_val:   Optional[int] = 20_000

    # ── Training ───────────────────────────────────────────────────────────
    epochs:          int   = 3
    batch_size:      int   = 32 if ENV == "kaggle" else 16
    grad_accum:      int   = 1  if ENV == "kaggle" else 2
    lr:              float = 2e-5
    warmup_ratio:    float = 0.06
    weight_decay:    float = 0.01
    label_smoothing: float = 0.0

    # ── Focal Loss ─────────────────────────────────────────────────────────
    use_focal:    bool  = False
    focal_gamma:  float = 2.0

    # ── Eval ───────────────────────────────────────────────────────────────
    eval_per_epoch:      int = 3
    early_stop_patience: int = 3
    save_total_limit:    int = 1   # tiết kiệm disk trên Kaggle

    # ── GBM ────────────────────────────────────────────────────────────────
    gbm_n_estimators: int   = 2000
    gbm_lr:           float = 0.03
    gbm_max_depth:    int   = 7
    gbm_subsample:    float = 0.8
    gbm_colsample:    float = 0.7

    # ── IsolationForest ────────────────────────────────────────────────────
    if_contamination:  float = 0.1
    if_n_estimators:   int   = 200

    # ── OOD-shifted features (loại bỏ khi train GBM) ──────────────────────
    lang_shifted_features: tuple = (
        "avg_identifier_len", "id_avg_len", "id_short_ratio",
        "line_entropy_std", "camel_case_ratio", "burstiness", "punct_density",
        "overall_ppl", "line_ppl_mean", "line_ppl_std",
        "line_ppl_max", "line_ppl_min", "ppl_variance",
    )

    # ── System ─────────────────────────────────────────────────────────────
    seed:               int  = 42
    fp16:               bool = field(default_factory=lambda: torch.cuda.is_available())
    dataloader_workers: int  = 4 if ENV == "kaggle" else 2
    log_level:          str  = "INFO"

    # ── Kaggle-specific ────────────────────────────────────────────────────
    env:                str  = field(default_factory=_detect_env)


if __name__ == "__main__":
    cfg = CFG()
    print(f"Environment : {cfg.env}")
    print(f"train_data  : {cfg.train_data}")
    print(f"output_dir  : {cfg.output_dir}")
    print(f"fp16        : {cfg.fp16}")
    print(f"batch_size  : {cfg.batch_size}")
