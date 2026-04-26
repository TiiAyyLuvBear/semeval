"""
pipeline_notebook_cells.py — Script tạo notebook pipeline.ipynb
================================================================
Chạy: python pipeline_notebook_cells.py
→ Tạo ra pipeline.ipynb có thể mở trong Jupyter/Kaggle/Colab
"""
import json, os

CELLS = [
    # ── Cell 0: Title & Setup ────────────────────────────────────────────
    {
        "type": "markdown",
        "source": [
            "# SemEval-2026 Task 13A — AI Code Detection Pipeline\n",
            "> **Tác giả**: 25C11066  \n",
            "> Mỗi cell gọi một module Python riêng biệt. Chỉ cần chỉnh `config.py`.\n",
            "\n",
            "## Flow\n",
            "```\n",
            "0. Setup & Config\n",
            "1. OOD Diagnosis  (diag_shift.py)\n",
            "2. Feature Extraction  (feature_extractor.py)\n",
            "3. Train GBM Ensemble  (train_gbm.py)\n",
            "4. Fine-tune CodeBERT  (train_codebert.py)\n",
            "5. Train IF+CNB  (train_ifcnb.py)\n",
            "6. Soft-Voting Ensemble  (ensemble.py)\n",
            "```"
        ]
    },
    # ── Cell 1: Install & import ─────────────────────────────────────────
    {
        "type": "code",
        "source": [
            "# ── Cài đặt (chỉ cần chạy lần đầu) ─────────────────────────────────\n",
            "# !pip install -r requirements.txt -q\n",
            "\n",
            "import os, sys, logging\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "\n",
            "# Thêm thư mục semeval/ vào path\n",
            "sys.path.insert(0, os.path.dirname(os.path.abspath('__file__')))\n",
            "\n",
            "logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')\n",
            "print('✓ Imports OK')"
        ]
    },
    # ── Cell 2: Config ───────────────────────────────────────────────────
    {
        "type": "code",
        "source": [
            "# ── Config — Chỉnh tất cả tham số ở ĐÂY ────────────────────────────\n",
            "from config import CFG\n",
            "\n",
            "cfg = CFG()\n",
            "\n",
            "# Override nếu chạy trên Kaggle\n",
            "IS_KAGGLE = os.path.exists('/kaggle')\n",
            "if IS_KAGGLE:\n",
            "    cfg.train_data  = '/kaggle/input/semeval2026/train.parquet'\n",
            "    cfg.val_data    = '/kaggle/input/semeval2026/validation.parquet'\n",
            "    cfg.test_data   = '/kaggle/input/semeval2026/test.parquet'\n",
            "    cfg.output_dir  = '/kaggle/working/outputs'\n",
            "    cfg.max_train   = 100_000\n",
            "    cfg.epochs      = 3\n",
            "\n",
            "os.makedirs(cfg.output_dir, exist_ok=True)\n",
            "print(f'Model: {cfg.model_name}')\n",
            "print(f'Output: {cfg.output_dir}')\n",
            "print(f'FP16: {cfg.fp16}')"
        ]
    },
    # ── Cell 3: OOD Diagnosis ────────────────────────────────────────────
    {
        "type": "markdown",
        "source": ["## 1. OOD Feature Shift Diagnosis\n",
                   "Tính Cohen's |d| cho từng feature giữa train và test.\n",
                   "Features có **|d| > 1.0** là ngôn ngữ-proxy — cần loại bỏ.\n"]
    },
    {
        "type": "code",
        "source": [
            "from diag_shift import compute_shift_report, plot_cohens_d\n",
            "\n",
            "# ⚠️  Cần có train_features và test_features đã extract sẵn\n",
            "TRAIN_FEAT = cfg.train_data.replace('.parquet', '_features.parquet')\n",
            "TEST_FEAT  = cfg.test_data.replace('.parquet', '_features.parquet') if cfg.test_data else None\n",
            "\n",
            "if TEST_FEAT and os.path.exists(TRAIN_FEAT) and os.path.exists(TEST_FEAT):\n",
            "    shift_df = compute_shift_report(\n",
            "        TRAIN_FEAT, TEST_FEAT,\n",
            "        top_n=25,\n",
            "        save_csv=os.path.join(cfg.output_dir, 'shift_report.csv')\n",
            "    )\n",
            "    # Vẽ biểu đồ\n",
            "    plot_cohens_d(shift_df, top_n=30,\n",
            "                  save_path=os.path.join(cfg.output_dir, 'cohens_d.png'))\n",
            "    display(shift_df.head(20))\n",
            "else:\n",
            "    print('Bỏ qua: chưa có feature files. Chạy feature extraction trước.')"
        ]
    },
    # ── Cell 4: Feature Extraction ───────────────────────────────────────
    {
        "type": "markdown",
        "source": ["## 2. Feature Extraction (75+ handcrafted features)\n",
                   "Trích xuất song song trên train / val / test.\n"]
    },
    {
        "type": "code",
        "source": [
            "from feature_extractor import extract_all_features\n",
            "from data_utils import load_dataframe\n",
            "\n",
            "# Định nghĩa đường dẫn feature files\n",
            "TRAIN_FEAT = os.path.join(cfg.output_dir, 'train_features.parquet')\n",
            "VAL_FEAT   = os.path.join(cfg.output_dir, 'val_features.parquet')\n",
            "TEST_FEAT  = os.path.join(cfg.output_dir, 'test_features.parquet') if cfg.test_data else None\n",
            "\n",
            "# ── Train ─────────────────────────────────────────────────────────────────\n",
            "if not os.path.exists(TRAIN_FEAT):\n",
            "    print('Extracting train features...')\n",
            "    train_df = load_dataframe(cfg.train_data, cfg.max_train, 'Train', cfg.seed)\n",
            "    train_feat = extract_all_features(train_df['code'], show_progress=True)\n",
            "    train_feat['label'] = train_df['label'].values\n",
            "    train_feat.to_parquet(TRAIN_FEAT, index=False)\n",
            "    print(f'Train: {train_feat.shape} -> {TRAIN_FEAT}')\n",
            "else:\n",
            "    print(f'[SKIP] Train features exist: {TRAIN_FEAT}')\n",
            "\n",
            "# ── Val ───────────────────────────────────────────────────────────────────\n",
            "if not os.path.exists(VAL_FEAT):\n",
            "    print('Extracting val features...')\n",
            "    val_df = load_dataframe(cfg.val_data, cfg.max_val, 'Val', cfg.seed)\n",
            "    val_feat = extract_all_features(val_df['code'], show_progress=True)\n",
            "    val_feat['label'] = val_df['label'].values\n",
            "    val_feat.to_parquet(VAL_FEAT, index=False)\n",
            "    print(f'Val: {val_feat.shape} -> {VAL_FEAT}')\n",
            "else:\n",
            "    print(f'[SKIP] Val features exist: {VAL_FEAT}')\n",
            "\n",
            "# ── Test (không có cột label) ─────────────────────────────────────────────\n",
            "# Test set chỉ có 'code' và 'ID', không có 'label'\n",
            "# Không dùng load_dataframe() vì hàm đó yêu cầu cột label\n",
            "if TEST_FEAT and not os.path.exists(TEST_FEAT):\n",
            "    if cfg.test_data and os.path.exists(cfg.test_data):\n",
            "        print('Extracting test features...')\n",
            "        test_raw = pd.read_parquet(cfg.test_data)\n",
            "        test_feat = extract_all_features(test_raw['code'], show_progress=True)\n",
            "        if 'ID' in test_raw.columns:\n",
            "            test_feat['ID'] = test_raw['ID'].values  # giữ ID để join submission\n",
            "        test_feat.to_parquet(TEST_FEAT, index=False)\n",
            "        print(f'Test: {test_feat.shape} -> {TEST_FEAT}')\n",
            "    else:\n",
            "        print(f'WARNING: test_data not found at {cfg.test_data}')\n",
            "        TEST_FEAT = None\n",
            "elif TEST_FEAT and os.path.exists(TEST_FEAT):\n",
            "    print(f'[SKIP] Test features exist: {TEST_FEAT}')\n",
            "\n",
            "print(f'\\nTRAIN_FEAT : {TRAIN_FEAT}')\n",
            "print(f'VAL_FEAT   : {VAL_FEAT}')\n",
            "print(f'TEST_FEAT  : {TEST_FEAT}')"
        ]
    },
    # ── Cell 5: Train GBM ────────────────────────────────────────────────
    {
        "type": "markdown",
        "source": ["## 3. Train Language-Robust GBM Ensemble\n",
                   "XGBoost + LightGBM + CatBoost với 75 features (loại 7 OOD features).\n"]
    },
    {
        "type": "code",
        "source": [
            "from train_gbm import run_gbm\n",
            "\n",
            "gbm_tau, val_proba_gbm = run_gbm(\n",
            "    train_feat_path = TRAIN_FEAT,\n",
            "    val_feat_path   = VAL_FEAT,\n",
            "    cfg             = cfg,\n",
            "    test_feat_path  = TEST_FEAT,\n",
            ")\n",
            "print(f'GBM best τ = {gbm_tau:.3f}')\n",
            "print(f'val_proba_gbm: mean={val_proba_gbm.mean():.3f}, std={val_proba_gbm.std():.3f}')"
        ]
    },
    # ── Cell 6: Fine-tune CodeBERT ───────────────────────────────────────
    {
        "type": "markdown",
        "source": ["## 4. Fine-tune CodeBERT\n",
                   "microsoft/codebert-base với Trainer API. Early stopping trên Macro F1.\n"]
    },
    {
        "type": "code",
        "source": [
            "from train_codebert import run_codebert\n",
            "\n",
            "codebert_tau, val_proba_codebert = run_codebert(cfg)\n",
            "print(f'CodeBERT best τ = {codebert_tau:.3f}')\n",
            "print(f'val_proba_codebert: mean={val_proba_codebert.mean():.3f}')"
        ]
    },
    # ── Cell 7: Train IF+CNB ─────────────────────────────────────────────
    {
        "type": "markdown",
        "source": ["## 5. Train IsolationForest + ComplementNB\n",
                   "20 style-only features — ít bị OOD shift nhất.\n",
                   "IF phát hiện code 'quá hoàn hảo' như outlier (ngôn ngữ-bất biến).\n"]
    },
    {
        "type": "code",
        "source": [
            "from train_ifcnb import run_ifcnb\n",
            "from data_utils import load_dataframe\n",
            "\n",
            "train_df_full = load_dataframe(cfg.train_data, cfg.max_train, 'Train', cfg.seed)\n",
            "val_df_full   = load_dataframe(cfg.val_data,   cfg.max_val,   'Val',   cfg.seed)\n",
            "test_df_full  = pd.read_parquet(cfg.test_data) if cfg.test_data else None\n",
            "\n",
            "val_proba_ifcnb = run_ifcnb(train_df_full, val_df_full, cfg, test_df=test_df_full)\n",
            "print(f'val_proba_ifcnb: mean={val_proba_ifcnb.mean():.3f}')"
        ]
    },
    # ── Cell 8: Ensemble ─────────────────────────────────────────────────
    {
        "type": "markdown",
        "source": ["## 6. Soft-Voting Ensemble\n",
                   "Rank-normalize → weighted average → quantile threshold calibration.\n"]
    },
    {
        "type": "code",
        "source": [
            "from ensemble import run_ensemble\n",
            "from data_utils import load_dataframe\n",
            "\n",
            "# Load val labels (full)\n",
            "val_full = load_dataframe(cfg.val_data, None, 'Full-Val', cfg.seed)\n",
            "y_val    = val_full['label'].values\n",
            "\n",
            "# Load test proba files\n",
            "def load_npy(path):\n",
            "    return np.load(path) if os.path.exists(path) else None\n",
            "\n",
            "test_gbm      = load_npy(os.path.join(cfg.output_dir, 'test_proba_gbm.npy'))\n",
            "test_codebert = load_npy(os.path.join(cfg.output_dir, 'test_proba_codebert.npy'))\n",
            "test_ifcnb    = load_npy(os.path.join(cfg.output_dir, 'test_proba_ifcnb.npy'))\n",
            "\n",
            "# Load test IDs\n",
            "test_ids = None\n",
            "if cfg.test_data and os.path.exists(cfg.test_data):\n",
            "    test_df_tmp = pd.read_parquet(cfg.test_data)\n",
            "    test_ids = test_df_tmp['ID'].tolist() if 'ID' in test_df_tmp.columns else list(range(len(test_df_tmp)))\n",
            "\n",
            "val_blend = run_ensemble(\n",
            "    val_proba_gbm      = val_proba_gbm,\n",
            "    val_proba_codebert = val_proba_codebert,\n",
            "    val_proba_ifcnb    = val_proba_ifcnb,\n",
            "    y_val              = y_val,\n",
            "    test_proba_gbm     = test_gbm,\n",
            "    test_proba_codebert= test_codebert,\n",
            "    test_proba_ifcnb   = test_ifcnb,\n",
            "    test_ids           = test_ids,\n",
            "    output_dir         = cfg.output_dir,\n",
            "    submission_out     = cfg.submission_out,\n",
            ")\n",
            "print(f'\\n✓ Submission → {cfg.submission_out}')"
        ]
    },
    # ── Cell 9: Summary ──────────────────────────────────────────────────
    {
        "type": "markdown",
        "source": ["## 7. Tóm tắt kết quả\n"]
    },
    {
        "type": "code",
        "source": [
            "from sklearn.metrics import f1_score\n",
            "from metrics import optimize_threshold\n",
            "\n",
            "results = {\n",
            "    'GBM':      val_proba_gbm,\n",
            "    'CodeBERT': val_proba_codebert,\n",
            "    'IF+CNB':   val_proba_ifcnb,\n",
            "    'Ensemble': val_blend,\n",
            "}\n",
            "\n",
            "print('\\n' + '='*55)\n",
            "print(f'{\"Model\":<15} {\"Best τ\":>8} {\"Val Macro F1\":>14}')\n",
            "print('-'*55)\n",
            "for name, proba in results.items():\n",
            "    tau, f1 = optimize_threshold(proba, y_val)\n",
            "    print(f'{name:<15} {tau:>8.3f} {f1:>14.4f}')\n",
            "print('='*55)"
        ]
    },
]

# =============================================================================
# Build notebook JSON
# =============================================================================
def make_cell(cell_type, source, cell_id):
    if cell_type == "markdown":
        return {
            "cell_type": "markdown",
            "id": f"cell-{cell_id}",
            "metadata": {},
            "source": source,
        }
    else:
        return {
            "cell_type": "code",
            "execution_count": None,
            "id": f"cell-{cell_id}",
            "metadata": {},
            "outputs": [],
            "source": source,
        }


notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
        },
    },
    "cells": [
        make_cell(c["type"], c["source"], i)
        for i, c in enumerate(CELLS)
    ],
}

out_path = os.path.join(os.path.dirname(__file__) or ".", "pipeline.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"[OK] Notebook created: {out_path}")
print(f"  Cells: {len(CELLS)}")
