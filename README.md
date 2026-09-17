# Lead Conversion Prediction — End-to-End ML Pipeline

> **DigiHust Assignment A02** — Build a complete, reproducible machine-learning
> classification pipeline that predicts whether an education-platform lead will
> convert into a paying customer.

---

## 📂 Project Structure

```
Leads Conversion Prediction/
├── .gitignore
├── README.md                    ← Project documentation & results
├── requirements.txt             ← Python dependencies
├── data/
│   └── raw/
│       ├── Leads.csv            ← Original leads dataset
│       └── Leads Data Dictionary.xlsx ← Field descriptions
├── notebooks/
│   └── 01_lead_conversion_ml_pipeline.ipynb   ← Primary executed learning notebook
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py    ← Loading, cleaning, column management
│   ├── feature_engineering.py   ← Derived features, outlier capping
│   ├── train.py                 ← Pipeline construction, GridSearchCV
│   ├── evaluate.py              ← Metrics, plots, model comparison
│   └── utils.py                 ← Seed management, model I/O
├── models/                      ← Saved model artefacts
├── outputs/
│   ├── figures/                 ← Exported visualisations
│   └── metrics/                 ← Exported evaluation metrics
└── report/                      ← Assignment reports
```

---

## 🚀 Quick Start

### Option A — Google Colab (recommended for learning)

1. Open **Google Colab** → *File → Upload notebook*.
2. Upload `notebooks/01_lead_conversion_ml_pipeline.ipynb`.
3. When prompted, upload `data/raw/Leads.csv`.
4. Execute cells **one by one** and read the explanations.

### Option B — Local environment

```bash
# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook notebooks/01_lead_conversion_ml_pipeline.ipynb
```

---

## 📊 Dataset

| Property          | Value                           |
|-------------------|---------------------------------|
| Source             | X Education CRM leads export   |
| Rows               | 9 240                           |
| Columns            | 37                              |
| Target             | `Converted` (0 = No, 1 = Yes)  |
| Class balance      | 61.5 % / 38.5 %                |

---

## 🔬 Pipeline Overview

1. **Data loading & inspection**
2. **Data quality assessment** — missing values, "Select" placeholders, constant columns
3. **Exploratory Data Analysis** — distributions, relationships, correlations
4. **Outlier handling** — percentile capping (thresholds learned from training data only)
5. **Feature engineering** — Activity Channel Count
6. **Train / Test split** — 80 / 20, stratified
7. **Preprocessing pipeline** — `ColumnTransformer` (median imputation + scaling for
   numerical; most-frequent imputation + one-hot encoding for categorical)
8. **Baseline models** — Logistic Regression, Random Forest
9. **Hyperparameter tuning** — `GridSearchCV` (5-fold CV, F1 scoring)
10. **Final evaluation** — Precision, Recall, F1, ROC-AUC, Confusion Matrix
11. **Feature importance & business interpretation**

---

## 📈 Model Performance (Untouched Test Set: 1,848 Leads)
 
| Model | Precision | Recall | F1-Score | ROC-AUC | Accuracy |
|---|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression (Baseline)** | 0.8490 | 0.8764 | 0.8625 | 0.9491 | 0.8923 |
| **Random Forest (Baseline)** | 0.8644 | 0.8413 | 0.8527 | 0.9478 | 0.8880 |
| **Tuned Logistic Regression** | 0.8490 | 0.8764 | 0.8625 | 0.9491 | 0.8923 |
| **Tuned Random Forest (Best)** | **0.8635** | **0.8708** | **0.8671** | **0.9526** | **0.8972** |

---

## 📝 Key ML Concepts Covered

- Missing-value imputation strategies
- Categorical encoding (One-Hot)
- Feature scaling (StandardScaler)
- Data leakage prevention (Pipeline / ColumnTransformer)
- Stratified train/test splitting
- Cross-validation
- Hyperparameter tuning (GridSearchCV)
- Evaluation metrics: Precision, Recall, F1, ROC-AUC
- Feature importance & model interpretability
- Correlation ≠ Causation

---

## 📜 License

Academic assignment — not intended for commercial use.
