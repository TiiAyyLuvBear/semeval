"""
ensemble.py — Rank-Average Soft Voting Ensemble
================================================
Kết hợp probabilities từ GBM, CodeBERT, IF+CNB bằng rank normalization.

Sử dụng:
    from ensemble import blend_probas, run_ensemble
"""
import logging
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from metrics import optimize_threshold, quantile_threshold, print_eval_report
from data_utils import detect_language

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Rank normalize → [0, 1]
# =============================================================================
def rank_normalize(proba: np.ndarray) -> np.ndarray:
    """
    Chuyển raw probability → rank ∈ [0, 1].
    Tại sao: mỗi model có scale khác nhau (CodeBERT overconfident hơn GBM).
    Rank normalize đưa về cùng scale mà không mất thứ tự tương đối.
    """
    r = rankdata(proba)                    # rank từ 1 đến N
    return (r - 1) / max(len(proba) - 1, 1)


# =============================================================================
# 2. Blend probabilities
# =============================================================================
def blend_probas(
    proba_dict: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
    use_rank: bool = True,
) -> np.ndarray:
    """
    Soft voting: weighted average của các probability.

    Args:
        proba_dict: {"gbm": arr, "codebert": arr, "ifcnb": arr}
        weights:    {"gbm": 1.0, ...} — None → equal weights
        use_rank:   True → rank-normalize trước khi average

    Returns:
        blended proba array, shape (N,)
    """
    names = list(proba_dict.keys())
    if weights is None:
        weights = {n: 1.0 for n in names}

    total_w = sum(weights[n] for n in names)
    # Verify shapes
    lengths = {name: len(arr) for name, arr in proba_dict.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"Mismatched prediction lengths: {lengths}. "
                         "All models must be evaluated on the same data subset.")

    blend = np.zeros(next(iter(lengths.values())))

    for name in names:
        p = rank_normalize(proba_dict[name]) if use_rank else proba_dict[name]
        blend += p * weights[name]

    return blend / total_w


    return blend / total_w


# =============================================================================
# 3. Apply Language Routing (Heuristic based on scientific notebook)
# =============================================================================
def apply_language_routing(proba: np.ndarray, codes: pd.Series) -> np.ndarray:
    """
    Điều chỉnh xác suất dựa trên ngôn ngữ:
      - JavaScript: +5% AI signal (capping at 0.99)
      - PHP: -5% AI signal (capping at 0.01)
      - Java/C#: +2% AI signal
    """
    logger.info("Applying Language Routing heuristic...")
    adjusted = proba.copy()
    
    try:
        from tqdm import tqdm
        it = tqdm(codes, desc="Language Routing")
    except ImportError:
        it = codes

    for i, code in enumerate(it):
        lang = detect_language(code)
        if lang == 'javascript':
            adjusted[i] = min(adjusted[i] * 1.05, 0.99)
        elif lang == 'php':
            adjusted[i] = max(adjusted[i] * 0.95, 0.01)
        elif lang == 'javaish':
            adjusted[i] = min(adjusted[i] * 1.02, 0.99)
    return adjusted


# =============================================================================
# 4. Main ensemble function
# =============================================================================
def run_ensemble(
    val_proba_gbm:     np.ndarray | None = None,
    val_proba_codebert: np.ndarray | None = None,
    val_proba_svm:     np.ndarray | None = None,
    val_proba_ifcnb:   np.ndarray | None = None,
    val_proba_tfidf:   np.ndarray | None = None,
    y_val:             np.ndarray = None,
    val_codes:         pd.Series | None = None,
    test_proba_gbm:     np.ndarray | None = None,
    test_proba_codebert: np.ndarray | None = None,
    test_proba_svm:     np.ndarray | None = None,
    test_proba_ifcnb:   np.ndarray | None = None,
    test_proba_tfidf:   np.ndarray | None = None,
    test_codes:         pd.Series | None = None,
    test_ids:          list | None = None,
    output_dir:        str = "outputs",
    submission_out:    str = "outputs/submission_ensemble.csv",
    ai_ratio:          float = 0.52,
) -> np.ndarray:
    """
    Tổng hợp 3 model → submission cuối cùng.

    Returns:
        val_proba_blend (để log/phân tích)
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Val: tìm trọng số tốt nhất ─────────────────────────────────────
    logger.info("Tối ưu ensemble weights trên val set...")

    proba_dict = {}
    if val_proba_gbm is not None: proba_dict["gbm"] = val_proba_gbm
    if val_proba_codebert is not None: proba_dict["codebert"] = val_proba_codebert
    if val_proba_svm is not None: proba_dict["svm"] = val_proba_svm
    if val_proba_ifcnb is not None: proba_dict["ifcnb"] = val_proba_ifcnb
    if val_proba_tfidf is not None: proba_dict["tfidf"] = val_proba_tfidf
    
    if len(proba_dict) == 0:
        raise ValueError("Ít nhất một model probability phải được truyền vào.")

    # Verify y_val shape matches probas
    first_len = len(next(iter(proba_dict.values())))
    if len(y_val) != first_len:
        raise ValueError(
            f"y_val length ({len(y_val)}) does not match prediction length ({first_len}). "
            "Ensure y_val is loaded with the same sampling limit (max_val) as the models."
        )

    best_f1, best_weights, best_tau = 0.0, None, 0.5
    weight_options = [0.5, 1.0, 1.5, 2.0]

    from itertools import product as ip
    for combo in ip(weight_options, repeat=len(proba_dict)):
        w = dict(zip(proba_dict.keys(), combo))
        blend = blend_probas(proba_dict, weights=w, use_rank=True)
        
        # Apply language routing on val to evaluate properly
        if val_codes is not None:
            blend = apply_language_routing(blend, val_codes)
            
        tau, f1 = optimize_threshold(blend, y_val)
        if f1 > best_f1:
            best_f1, best_weights, best_tau = f1, w, tau

    logger.info(f"Trọng số tốt nhất: {best_weights}")
    logger.info(f"  τ*={best_tau:.3f}  →  Val Macro F1={best_f1:.4f}")

    # ── 2. Val blend & report ──────────────────────────────────────────────
    val_blend  = blend_probas(proba_dict, weights=best_weights, use_rank=True)
    if val_codes is not None:
        val_blend = apply_language_routing(val_blend, val_codes)
        
    y_pred_val = (val_blend >= best_tau).astype(int)
    print_eval_report(y_val, y_pred_val, val_blend, "Val (Ensemble cuối)")

    np.save(os.path.join(output_dir, "val_proba_ensemble.npy"), val_blend)

    # ── 3. So sánh từng model ─────────────────────────────────────────────
    print("\n─ So sánh từng model trên val ─")
    for name, proba in proba_dict.items():
        tau, f1 = optimize_threshold(proba, y_val)
        print(f"  {name:12s}: Macro F1={f1:.4f}  (best τ={tau:.3f})")
    print(f"  {'ensemble':12s}: Macro F1={best_f1:.4f}  (τ={best_tau:.3f})")

    # ── 4. Test inference ─────────────────────────────────────────────────
    test_proba_dict = {}
    if test_proba_gbm is not None: test_proba_dict["gbm"] = test_proba_gbm
    if test_proba_codebert is not None: test_proba_dict["codebert"] = test_proba_codebert
    if test_proba_svm is not None: test_proba_dict["svm"] = test_proba_svm
    if test_proba_ifcnb is not None: test_proba_dict["ifcnb"] = test_proba_ifcnb
    if test_proba_tfidf is not None: test_proba_dict["tfidf"] = test_proba_tfidf
    
    if len(test_proba_dict) == len(proba_dict):
        logger.info("Blend test probabilities...")
        test_blend = blend_probas(test_proba_dict, weights=best_weights, use_rank=True)
        if test_codes is not None:
            test_blend = apply_language_routing(test_blend, test_codes)
            
        np.save(os.path.join(output_dir, "test_proba_ensemble.npy"), test_blend)

        # Primary: val-tuned threshold
        y_test_best = (test_blend >= best_tau).astype(int)

        # Safety net: quantile threshold
        q_tau = quantile_threshold(test_blend, ai_ratio=ai_ratio)
        y_test_q = (test_blend >= q_tau).astype(int)

        print(f"\nTest dist (τ={best_tau:.3f}): Human={(y_test_best==0).sum()}, AI={(y_test_best==1).sum()}")
        print(f"Test dist (q_τ={q_tau:.3f}): Human={(y_test_q==0).sum()}, AI={(y_test_q==1).sum()}")

        if test_ids is not None:
            pd.DataFrame({"ID": test_ids, "label": y_test_best}).to_csv(
                submission_out, index=False
            )
            pd.DataFrame({"ID": test_ids, "label": y_test_q}).to_csv(
                submission_out.replace(".csv", "_qthresh.csv"), index=False
            )
            logger.info(f"Submission (primary)  → {submission_out}")
            logger.info(f"Submission (q-thresh) → *_qthresh.csv")

    return val_blend
