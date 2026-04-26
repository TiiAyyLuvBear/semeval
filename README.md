# SemEval-2026 Task 13A — AI Code Detection Pipeline

## Cấu trúc project

```
semeval/
├── config.py              # ⚙️  Tất cả hyperparameters (chỉnh tại đây)
├── data_utils.py          # 📦  Load, subsample, tokenize dataset
├── feature_extractor.py   # 🔬  75+ handcrafted features (whitespace, entropy, naming...)
├── metrics.py             # 📊  compute_metrics, threshold optimization
├── trainer_utils.py       # 🏋️  Build HuggingFace Trainer (CE / Focal Loss)
├── train_codebert.py      # 🤖  Fine-tune CodeBERT
├── train_gbm.py           # 🌲  Language-robust GBM Ensemble (XGB+LGB+CAT)
├── train_ifcnb.py         # 🧩  IsolationForest + ComplementNB
├── ensemble.py            # 🗳️  Rank-average Soft Voting
├── diag_shift.py          # 🩺  OOD Feature Shift Diagnosis (Cohen's d)
├── pipeline.ipynb         # 📓  Notebook orchestrator
├── requirements.txt
└── README.md
```

## Pipeline tổng quan

```
Raw Code
  ├── [feature_extractor.py] → 75+ handcrafted features
  │      ├── [train_gbm.py]    → GBM Ensemble     → val_proba_gbm.npy
  │      └── [train_ifcnb.py]  → IF + CNB         → val_proba_ifcnb.npy
  └── [train_codebert.py]     → CodeBERT fine-tune → val_proba_codebert.npy
                                        │
                               [ensemble.py]
                          Rank Normalize + Soft Voting
                                        │
                               Submission CSV
```

## Xử lý OOD (Language Shift)

| Bước | File | Mô tả |
|---|---|---|
| 1. Chẩn đoán | `diag_shift.py` | Tính Cohen's d mỗi feature giữa train/test |
| 2. Drop features | `config.py → lang_shifted_features` | Loại bỏ features có `|d| > 1.0` |
| 3. Focal Loss | `trainer_utils.py → FocalTrainer` | Down-weight easy samples (Python) |
| 4. Robust features | `train_ifcnb.py` | Chỉ dùng 20 style features ít bị shift |
| 5. Threshold cal. | `ensemble.py` | Quantile threshold khớp prior train |

## Cách chạy

### 1. Cài đặt
```bash
pip install -r requirements.txt
```

### 2. Notebook (khuyến nghị)
Mở `pipeline.ipynb` và chạy từng cell theo thứ tự.

### 3. Command line
```bash
# Chẩn đoán OOD shift
python diag_shift.py --train_feat data/train_features.parquet \
                     --test_feat  data/test_features.parquet --plot

# Train GBM
python train_gbm.py --train_feat data/train_features.parquet \
                    --val_feat   data/val_features.parquet

# Fine-tune CodeBERT
python train_codebert.py --epochs 3 --batch_size 16

# Train IF+CNB
python train_ifcnb.py
```

## Kết quả (SemEval-2026 Task 13A)

| System | Val F1 | Test F1 |
|---|---:|---:|
| GBM Ensemble (83 feat) | 0.989 | 0.254 |
| GBM Ensemble (75 feat, lang-robust) | 0.973 | **0.472** |
| CodeBERT (τ=0.93) | 0.954 | 0.451 |
| **IF+CNB** | 0.823 | **0.535** |
| **Soft-Voting Ensemble** | **0.986** | **0.527** |

> Val F1 cao **KHÔNG** đảm bảo Test F1 cao khi có OOD shift. Luôn chạy `diag_shift.py` trước.
