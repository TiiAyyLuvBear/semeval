"""
config.py — Cấu hình tập trung cho toàn bộ pipeline
=====================================================
Chỉnh tham số ở ĐÂY, không rải rác trong từng file.
Notebook gọi: from config import CFG
"""
from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class CFG:
    # ── Đường dẫn dữ liệu ─────────────────────────────────────────────────
    train_data:      str           = "data/train.parquet"
    val_data:        str           = "data/validation.parquet"
    test_data:       Optional[str] = "data/test.parquet"
    output_dir:      str           = "outputs"
    submission_out:  str           = "outputs/submission.csv"

    # ── Model ──────────────────────────────────────────────────────────────
    model_name:      str  = "microsoft/codebert-base"
    max_length:      int  = 256          # tokens per sample
    num_labels:      int  = 2            # 0=Human, 1=AI

    # ── Sampling ───────────────────────────────────────────────────────────
    max_train:       Optional[int] = 100_000   # None → dùng toàn bộ
    max_val:         Optional[int] = 20_000

    # ── Training ───────────────────────────────────────────────────────────
    epochs:          int   = 3
    batch_size:      int   = 16           # per device
    grad_accum:      int   = 2            # effective = batch_size × grad_accum
    lr:              float = 2e-5
    warmup_ratio:    float = 0.06
    weight_decay:    float = 0.01
    label_smoothing: float = 0.0          # 0.0 = tắt

    # ── Focal Loss ─────────────────────────────────────────────────────────
    use_focal:       bool  = False
    focal_gamma:     float = 2.0

    # ── Eval & Early Stop ──────────────────────────────────────────────────
    eval_per_epoch:      int = 3          # eval N lần / epoch
    early_stop_patience: int = 3
    save_total_limit:    int = 2

    # ── GBM Ensemble ───────────────────────────────────────────────────────
    gbm_n_estimators:  int   = 2000
    gbm_lr:            float = 0.03
    gbm_max_depth:     int   = 7
    gbm_subsample:     float = 0.8
    gbm_colsample:     float = 0.7

    # ── IsolationForest + CNB ──────────────────────────────────────────────
    if_contamination: float = 0.1        # expected fraction of outliers
    if_n_estimators:  int   = 200

    # ── Features bị loại bỏ do OOD shift (|d| > 1.0) ──────────────────────
    lang_shifted_features: tuple = (
        "avg_identifier_len",
        "id_avg_len",
        "id_short_ratio",
        "line_entropy_std",
        "camel_case_ratio",
        "burstiness",
        "punct_density",
        "overall_ppl",
        "line_ppl_mean",
        "line_ppl_std",
        "line_ppl_max",
        "line_ppl_min",
        "ppl_variance",
    )

    # ── Khác ───────────────────────────────────────────────────────────────
    seed:               int  = 42
    fp16:               bool = field(default_factory=lambda: torch.cuda.is_available())
    dataloader_workers: int  = 2
    log_level:          str  = "INFO"
