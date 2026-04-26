"""
trainer_utils.py — Build HuggingFace Trainer (standard & Focal Loss)
=====================================================================
Exports:
    build_trainer(model, tokenizer, train_ds, val_ds, cfg) -> Trainer
    FocalLoss
    FocalTrainer
"""
import logging
import math
import os
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from metrics import compute_metrics

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Focal Loss
# =============================================================================
class FocalLoss(torch.nn.Module):
    """
    Focal Loss: FL = −(1 − p_t)^γ · log(p_t)

    γ = 0  → Cross-Entropy thông thường
    γ = 2  → down-weight easy samples (Python code quen thuộc),
             tập trung vào hard samples (ngôn ngữ mới — OOD)

    Tại sao hữu ích khi bị language shift:
      - Model tự tin đúng trên Python (easy) → weight ≈ 0, bị bỏ qua
      - Model bất định trên JS/Go/Ruby (hard) → weight cao → học features tổng quát hơn
    """

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce = torch.nn.functional.cross_entropy(
            logits,
            labels,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)                      # xác suất của class đúng
        focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * ce).mean()


# =============================================================================
# 2. Trainer con — override compute_loss với Focal Loss
# =============================================================================
class FocalTrainer(Trainer):
    def __init__(self, *args, focal_gamma: float = 2.0,
                 label_smoothing: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = FocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing)
        logger.info(
            f"FocalTrainer: gamma={focal_gamma}, label_smoothing={label_smoothing}"
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


# =============================================================================
# 3. Factory function — build Trainer từ CFG
# =============================================================================
def build_trainer(model, tokenizer, train_ds, val_ds, cfg) -> Trainer:
    """
    Tạo Trainer (hoặc FocalTrainer) hoàn chỉnh từ config.

    Args:
        model:     AutoModelForSequenceClassification
        tokenizer: AutoTokenizer
        train_ds:  HuggingFace Dataset (đã tokenize, có cột 'labels')
        val_ds:    HuggingFace Dataset (đã tokenize, có cột 'labels')
        cfg:       CFG instance

    Returns:
        Trainer (hoặc FocalTrainer nếu cfg.use_focal=True)
    """
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── Tính eval_steps ────────────────────────────────────────────────────
    effective_batch = cfg.batch_size * cfg.grad_accum
    steps_per_epoch = max(len(train_ds) // effective_batch, 1)
    eval_steps      = max(steps_per_epoch // cfg.eval_per_epoch, 1)
    log_steps       = max(eval_steps // 5, 10)

    logger.info(
        f"Batch hiệu dụng: {effective_batch} | "
        f"Steps/epoch: {steps_per_epoch} | "
        f"Eval every: {eval_steps} steps"
    )

    # ── TrainingArguments ──────────────────────────────────────────────────
    training_args = TrainingArguments(
        # Cơ bản
        output_dir                  = cfg.output_dir,
        num_train_epochs            = cfg.epochs,
        per_device_train_batch_size = cfg.batch_size,
        per_device_eval_batch_size  = cfg.batch_size * 2,
        gradient_accumulation_steps = cfg.grad_accum,
        learning_rate               = cfg.lr,
        warmup_ratio                = cfg.warmup_ratio,
        weight_decay                = cfg.weight_decay,
        # Label smoothing chỉ áp dụng khi KHÔNG dùng FocalTrainer
        label_smoothing_factor      = cfg.label_smoothing if not cfg.use_focal else 0.0,

        # Precision
        fp16                        = cfg.fp16,

        # Logging
        logging_dir                 = os.path.join(cfg.output_dir, "logs"),
        logging_steps               = log_steps,
        report_to                   = "none",   # đổi "tensorboard" nếu muốn

        # Eval & save
        eval_strategy               = "steps",
        eval_steps                  = eval_steps,
        save_strategy               = "steps",
        save_steps                  = eval_steps,
        load_best_model_at_end      = True,
        metric_for_best_model       = "macro_f1",
        greater_is_better           = True,
        save_total_limit            = cfg.save_total_limit,

        # Tốc độ
        dataloader_num_workers      = cfg.dataloader_workers,
        dataloader_pin_memory       = cfg.fp16,

        # Reproduce
        seed                        = cfg.seed,
    )

    # ── Tham số dùng chung ────────────────────────────────────────────────
    trainer_kwargs = dict(
        model            = model,
        args             = training_args,
        train_dataset    = train_ds,
        eval_dataset     = val_ds,
        processing_class = tokenizer,
        data_collator    = DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics  = compute_metrics,
        callbacks        = [
            EarlyStoppingCallback(
                early_stopping_patience=cfg.early_stop_patience
            )
        ],
    )

    if cfg.use_focal:
        logger.info("Dùng FocalTrainer (Focal Loss)")
        return FocalTrainer(
            **trainer_kwargs,
            focal_gamma     = cfg.focal_gamma,
            label_smoothing = cfg.label_smoothing,
        )
    else:
        logger.info(
            f"Dùng Trainer chuẩn (CE + label_smoothing={cfg.label_smoothing})"
        )
        return Trainer(**trainer_kwargs)
