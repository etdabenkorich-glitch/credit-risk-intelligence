# 💳 Credit Risk Classifier

> **End-to-end credit default prediction pipeline** using Logistic Regression, Naive Bayes, Random Forest, and XGBoost — with SHAP-based explainability aligned to Basel III / IFRS 9 requirements.

---

## 📌 Business Problem

Banks and financial institutions lose billions annually from clients who default on credit card payments. The challenge is not just *prediction* — it's **explainability**: a model that says "this client will default" is worthless in a regulated environment unless it can explain *why*.

This project builds a full ML pipeline that:
- Predicts credit default probability with high AUC
- Engineers domain-specific financial features
- Explains every prediction at the individual client level via SHAP
- Outputs a business-ready PDF report

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [UCI ML Repository — Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) |
| Clients | ~30,000 Taiwanese credit card holders |
| Target | `DEFAULT` — binary (1 = default next month) |
| Class imbalance | ~30% default / 70% pay |
| Features | 23 raw + 3 engineered |

> ⚠️ **Data not included** in this repo. Download from the link above and place the CSV in `data/` as `credit_risk_data.csv`.

---

## 🧠 Engineered Features

Three financial indicators derived from domain knowledge:

| Feature | Formula | Financial Meaning |
|---|---|---|
| `CREDIT_UTILIZATION` | `mean(BILL_AMT1–6) / LIMIT_BAL` | Ratio of credit used vs limit — proxy for financial stress |
| `REPAYMENT_RATIO` | `mean(PAY_AMT1–6) / (mean(BILL_AMT1–6) + 1)` | Fraction of bill actually paid — early warning signal |
| `CONSECUTIVE_DELAYS` | `sum(PAY_0–6 > 0)` | Number of months with payment delays in last 6 months |

---

## 🤖 Models & Results

All models evaluated on the **same 20% test split** (random_state=17, stratified) for a valid comparison.

| Model | AUC-ROC | Notes |
|---|---|---|
| Logistic Regression | ~0.77 | Linear baseline, highly interpretable |
| Naive Bayes (tuned) | ~0.73 | Fast probabilistic classifier |
| Random Forest (tuned) | ~0.79 | Ensemble, captures non-linearity |
| **XGBoost ★** | **~0.79+** | **Best AUC + SHAP explainability** |

> *Exact values depend on your dataset version. Run the notebook to reproduce.*

---

## 🔍 SHAP Explainability

Three levels of explanation — from global to individual:

**1. Beeswarm plot** — distribution of feature impact across all test clients  
**2. Mean |SHAP| bar chart** — global feature ranking (comparable to LR coefficients)  
**3. Waterfall plot** — single client explanation for the highest-risk profile

![SHAP explainability is regulatory-grade — required by Basel III Art. 22 and GDPR]

---

## 📁 Project Structure

```
credit-risk-classifier/
│
├── notebooks/
│   └── Project1_Complete.ipynb     # Full pipeline: EDA → Features → Models → SHAP
│
├── reports/
│   └── Credit_Risk_Final_Report_v2.pdf   # Auto-generated business report
│
├── data/
│   └── README.md                   # Data source instructions
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/credit-risk-classifier.git
cd credit-risk-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add data (see data/README.md)

# 4. Launch notebook
jupyter notebook notebooks/Project1_Complete.ipynb
```

---

## 💡 Key Findings

- **PAY_0** (most recent payment delay) is the strongest predictor — a client delayed by 2+ months has ~75% default probability
- **REPAYMENT_RATIO < 0.10** (paying less than 10% of the bill) is a critical early warning signal
- **CREDIT_UTILIZATION > 80%** combined with recent delays identifies the highest-risk segment
- XGBoost confirms **H7**: non-linear models outperform linear classifiers on this dataset
- Estimated **financial savings** vs no-model baseline: computed in notebook based on Basel III LGD = 45%

---

## 🏦 Regulatory Context

| Framework | Relevance |
|---|---|
| Basel III (IRB approach) | PD (Probability of Default) estimation |
| IFRS 9 | Expected credit loss staging |
| GDPR Art. 22 | Right to explanation for automated decisions |

SHAP values provide the audit trail required by all three frameworks.

---

## 🛠️ Tech Stack

`Python` · `scikit-learn` · `XGBoost` · `SHAP` · `pandas` · `matplotlib` · `seaborn` · `ReportLab` · `statsmodels`

---

## 👤 Author

**Your Name**  
MSc student — Finance & Business Administration | HEC  
[LinkedIn](https://linkedin.com/in/YOUR_PROFILE) · [GitHub](https://github.com/YOUR_USERNAME)

---

*This project is part of a portfolio combining quantitative finance and machine learning.*
