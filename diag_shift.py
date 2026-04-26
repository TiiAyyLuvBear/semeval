"""
diag_shift.py — Chẩn đoán OOD feature shift (Cohen's d)
=========================================================
Tính standardized mean difference giữa train và test cho từng feature.
Features có |d| > 1.0 là "language proxy" → cần loại bỏ.

Sử dụng:
    from diag_shift import compute_shift_report, plot_cohens_d
    df_shift = compute_shift_report(train_feat_path, test_feat_path)
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

META_COLS = {"label", "ID", "code", "generator", "language", "__lang"}


# =============================================================================
# 1. Tính Cohen's d cho từng feature
# =============================================================================
def compute_cohens_d(
    train: pd.DataFrame,
    test:  pd.DataFrame,
    exclude_cols: set = META_COLS,
) -> pd.DataFrame:
    """
    Tính |d| = |mean_train − mean_test| / pooled_std cho từng feature.

    Returns:
        DataFrame với cột [feature, mean_train, mean_test, std_train, std_test, abs_d]
        Sắp xếp giảm dần theo abs_d.
    """
    feat_cols = [c for c in train.columns if c not in exclude_cols and c in test.columns]
    rows = []

    for c in feat_cols:
        tr = train[c].replace([np.inf, -np.inf], np.nan).dropna()
        te = test[c].replace([np.inf, -np.inf], np.nan).dropna()

        if len(tr) == 0 or len(te) == 0:
            continue

        m_tr, m_te = tr.mean(), te.mean()
        s_tr, s_te = tr.std(), te.std()
        pooled = np.sqrt((s_tr**2 + s_te**2) / 2)
        d = abs(m_tr - m_te) / pooled if pooled > 1e-10 else 0.0

        rows.append({
            "feature":    c,
            "mean_train": round(m_tr, 4),
            "mean_test":  round(m_te, 4),
            "std_train":  round(s_tr, 4),
            "std_test":   round(s_te, 4),
            "abs_d":      round(d, 4),
        })

    df = pd.DataFrame(rows).sort_values("abs_d", ascending=False).reset_index(drop=True)
    return df


# =============================================================================
# 2. Main report function
# =============================================================================
def compute_shift_report(
    train_feat_path: str,
    test_feat_path:  str,
    top_n:           int = 25,
    save_csv:        str | None = None,
) -> pd.DataFrame:
    """
    Tải features train & test → tính Cohen's d → in báo cáo.

    Returns:
        DataFrame shift report đầy đủ.
    """
    logger.info(f"Tải train features: {train_feat_path}")
    train = pd.read_parquet(train_feat_path)

    logger.info(f"Tải test features:  {test_feat_path}")
    test  = pd.read_parquet(test_feat_path)

    logger.info(f"Train: {train.shape} | Test: {test.shape}")

    shift_df = compute_cohens_d(train, test)

    # ── In top N bị shift nhất ─────────────────────────────────────────────
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"TOP {top_n} FEATURES BỊ SHIFT NHẤT (train vs test)")
    print(sep)
    print(f"{'Feature':<30} {'train_mean':>12} {'test_mean':>12} {'|d|':>8}")
    print("-" * 65)
    for _, row in shift_df.head(top_n).iterrows():
        flag = " ⚠️  OOD" if row["abs_d"] > 1.0 else (" !" if row["abs_d"] > 0.5 else "")
        print(f"{row['feature']:<30} {row['mean_train']:>12.4f} {row['mean_test']:>12.4f} {row['abs_d']:>8.3f}{flag}")

    # ── In features ít shift nhất ──────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("FEATURES ÍT SHIFT NHẤT (robust với OOD):")
    for _, row in shift_df.tail(10).iterrows():
        print(f"  {row['feature']:<30}  |d|={row['abs_d']:.3f}")

    # ── Tóm tắt ───────────────────────────────────────────────────────────
    n_severe = (shift_df["abs_d"] > 1.0).sum()
    n_moderate = ((shift_df["abs_d"] > 0.5) & (shift_df["abs_d"] <= 1.0)).sum()
    n_ok = (shift_df["abs_d"] <= 0.5).sum()
    print(f"\n{'─'*65}")
    print(f"Tóm tắt: {n_severe} severe (|d|>1.0) | {n_moderate} moderate (0.5-1.0) | {n_ok} stable (≤0.5)")
    print(f"Khuyến nghị: loại bỏ {n_severe} features có |d|>1.0")

    if save_csv:
        shift_df.to_csv(save_csv, index=False)
        logger.info(f"Shift report → {save_csv}")

    return shift_df


# =============================================================================
# 3. (Tuỳ chọn) Plot Cohen's d
# =============================================================================
def plot_cohens_d(shift_df: pd.DataFrame, top_n: int = 30, save_path: str | None = None):
    """Vẽ bar chart Cohen's d — cần matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib không có, bỏ qua plot.")
        return

    df_plot = shift_df.head(top_n).sort_values("abs_d")
    colors = ["#e74c3c" if d > 1.0 else "#f39c12" if d > 0.5 else "#2ecc71"
              for d in df_plot["abs_d"]]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    ax.barh(df_plot["feature"], df_plot["abs_d"], color=colors)
    ax.axvline(1.0, color="red",    linestyle="--", linewidth=1.5, label="|d|=1.0 (drop)")
    ax.axvline(0.5, color="orange", linestyle="--", linewidth=1.0, label="|d|=0.5 (warn)")
    ax.set_xlabel("Cohen's |d| (train vs test)", fontsize=12)
    ax.set_title("Feature OOD Shift — Train vs Test", fontsize=14)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot → {save_path}")
    else:
        plt.show()


# =============================================================================
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("--train_feat", required=True)
    p.add_argument("--test_feat",  required=True)
    p.add_argument("--top_n",  type=int, default=25)
    p.add_argument("--save_csv", default=None)
    p.add_argument("--plot",     action="store_true")
    args = p.parse_args()

    df = compute_shift_report(args.train_feat, args.test_feat,
                               top_n=args.top_n, save_csv=args.save_csv)
    if args.plot:
        plot_cohens_d(df, top_n=args.top_n,
                      save_path=args.save_csv.replace(".csv", ".png") if args.save_csv else None)
