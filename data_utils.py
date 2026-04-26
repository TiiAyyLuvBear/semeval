"""
data_utils.py — Load, subsample, tokenize dataset
===================================================
Exports:
    load_dataframe(path, max_samples, split_name) -> pd.DataFrame
    build_hf_dataset(df, tokenizer, max_length)   -> datasets.Dataset
"""
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset

logger = logging.getLogger(__name__)


def load_dataframe(
    path: str,
    max_samples: int | None = None,
    split_name: str = "data",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Tải .parquet, giữ cột 'code' & 'label', subsample stratified nếu cần.

    Args:
        path:        Đường dẫn .parquet
        max_samples: Số mẫu tối đa (None = tất cả)
        split_name:  Tên hiển thị trong log ("Train", "Val", ...)
        seed:        Random seed cho subsample

    Returns:
        DataFrame với cột ['code', 'label'], index reset
    """
    logger.info(f"[{split_name}] Đang tải: {path}")
    df = pd.read_parquet(path)

    # Kiểm tra cột bắt buộc
    missing = [c for c in ("code", "label") if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{split_name}] Thiếu cột {missing}. "
            f"Cột hiện có: {df.columns.tolist()}"
        )

    df = df[["code", "label"]].dropna()
    df["label"] = df["label"].astype(int)

    if max_samples and max_samples < len(df):
        df, _ = train_test_split(
            df,
            train_size=max_samples,
            stratify=df["label"],
            random_state=seed,
        )
        logger.info(f"  Subsample → {max_samples:,} mẫu (stratified)")

    n_human = (df["label"] == 0).sum()
    n_ai    = (df["label"] == 1).sum()
    logger.info(f"  [{split_name}] {len(df):,} mẫu | Human={n_human:,} | AI={n_ai:,}")
    return df.reset_index(drop=True)


def build_hf_dataset(
    df: pd.DataFrame,
    tokenizer,
    max_length: int = 256,
) -> Dataset:
    """
    Chuyển DataFrame → HuggingFace Dataset đã tokenize.

    Args:
        df:         DataFrame có cột ['code', 'label']
        tokenizer:  HuggingFace tokenizer
        max_length: Token length tối đa

    Returns:
        Dataset có cột [input_ids, attention_mask, labels]
    """
    ds = Dataset.from_pandas(df)

    def tokenize_fn(batch):
        return tokenizer(
            batch["code"],
            truncation=True,
            max_length=max_length,
            # padding=False — DataCollatorWithPadding tự pad từng batch
        )

    ds = ds.map(tokenize_fn, batched=True, remove_columns=["code"])
    # Trainer yêu cầu tên cột là 'labels' (số nhiều)
    ds = ds.rename_column("label", "labels")
    return ds


def load_test_dataset(
    path: str,
    tokenizer,
    max_length: int = 256,
):
    """
    Tải test set (không có cột label) → Dataset tokenize + list ID.

    Returns:
        (test_ds, test_ids)
    """
    logger.info(f"[Test] Đang tải: {path}")
    df = pd.read_parquet(path)
    test_ids = df["ID"] if "ID" in df.columns else list(range(len(df)))

    ds = Dataset.from_pandas(df[["code"]].reset_index(drop=True))

    def tokenize_fn(batch):
        return tokenizer(
            batch["code"],
            truncation=True,
            max_length=max_length,
        )

    ds = ds.map(tokenize_fn, batched=True, remove_columns=["code"])
    logger.info(f"  [Test] {len(ds):,} mẫu")
    return ds, test_ids
