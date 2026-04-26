"""
metrics.py — Evaluation metrics & threshold utilities
======================================================
Exports:
    compute_metrics(eval_pred)              -> dict  (dùng cho Trainer)
    optimize_threshold(proba, y_true)       -> (best_tau, best_f1)
    quantile_threshold(proba, ai_ratio)     -> float
    print_eval_report(y_true, y_pred, proba)
"""
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report


# =============================================================================
# 1. Metric callback cho HuggingFace Trainer
# =============================================================================
def compute_metrics(eval_pred):
    """
    Được truyền vào Trainer(compute_metrics=compute_metrics).
    eval_pred: EvalPrediction(predictions=logits, label_ids=y_true)

    Returns dict — tất cả key xuất hiện trong training log.
    """
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=1)

    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_human": f1_score(y_true, y_pred, pos_label=0, average="binary"),
        "f1_ai":    f1_score(y_true, y_pred, pos_label=1, average="binary"),
    }


# =============================================================================
# 2. Tìm threshold tốt nhất trên validation set
# =============================================================================
def optimize_threshold(
    proba: np.ndarray,
    y_true: np.ndarray,
    lo: float = 0.20,
    hi: float = 0.95,
    step: float = 0.01,
) -> tuple[float, float]:
    """
    Grid search threshold τ ∈ [lo, hi] để maximize Macro F1.

    Args:
        proba:  xác suất dự đoán là AI (class 1), shape (N,)
        y_true: nhãn thực, shape (N,)
        lo, hi, step: khoảng tìm kiếm

    Returns:
        (best_tau, best_f1)
    """
    best_f1, best_tau = 0.0, 0.5
    for tau in np.arange(lo, hi, step):
        f1 = f1_score(y_true, (proba >= tau).astype(int), average="macro")
        if f1 > best_f1:
            best_f1, best_tau = f1, round(float(tau), 4)
    return best_tau, best_f1


# =============================================================================
# 3. Quantile threshold — khớp prior distribution
# =============================================================================
def quantile_threshold(
    proba: np.ndarray,
    ai_ratio: float = 0.52,
) -> float:
    """
    Đặt threshold sao cho tỉ lệ AI dự đoán ≈ ai_ratio.
    Tránh trường hợp model bị shift và predict 90%+ AI.

    Args:
        proba:    xác suất AI, shape (N,)
        ai_ratio: tỉ lệ mong muốn (train prior ≈ 0.52)

    Returns:
        threshold float
    """
    return float(np.quantile(proba, 1.0 - ai_ratio))


# =============================================================================
# 4. In báo cáo đầy đủ
# =============================================================================
def print_eval_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray | None = None,
    split_name: str = "Validation",
):
    """In Macro F1, Accuracy, classification report và phân phối threshold."""
    sep = "═" * 60
    print(f"\n{sep}")
    print(f"KẾT QUẢ {split_name.upper()}")
    print(sep)
    print(f"Macro F1:  {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, target_names=["Human", "AI"]))

    if proba is not None:
        print("─ Phân phối xác suất AI ─")
        for tau in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            pct = (proba >= tau).mean() * 100
            print(f"  τ={tau:.1f} → {pct:.1f}% AI")
