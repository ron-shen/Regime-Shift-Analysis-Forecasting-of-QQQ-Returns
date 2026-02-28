# Regime-Shift Analysis & Forecasting of QQQ Returns

This repository contains a quantitative research project focused on predicting the **1-day forward open-to-open return of the QQQ ETF**. The project emphasizes the transition of market mechanics in the post-COVID era, specifically diagnosing model decay through the lens of **Concept Drift**.

## 🚀 Project Overview
The core objective was to determine if alpha decay in a predictive model was caused by changes in feature distributions (**Covariate Shift**) or changes in the underlying relationship between features and the target (**Concept Drift**).

### Key Highlights:
* **Target:** 1-day forward open-to-open returns ($\frac{Open_{t+2}}{Open_t+1} - 1$).
* **Feature Set:** Multi-frequency data including Equities (QQQ), Macro indicators (Gold, Treasury Yields, DIX, GEX, Fed Reserve Repo, VIX), Forex (GBP/USD, USD/JPY), Treasury (2 year yield and 10 year - 2 year yield)
* **Primary Finding:** Model performs better in post-covid market (profit factor > 1) over pre-covid market. Post-COVID market regimes favored mid-term trend indicators over short-term mean-reversion signals.

## 🛠️ Technical Implementation

### Data Pipeline & Orchestration
* **Feature Engineering:** Automated generation of technical indicators (ROC, Volatility) and time-series features.
* **Pipeline:** Utilized `ColumnTransformer` and `StandardScaler` to ensure rigorous data preprocessing without leakage.
* **Cross-Validation:** Implemented a custom `SlidingWindowCV` (1,000 days training / 100 days testing) to simulate a real-world walk-forward trading environment.

### Modeling Approach
The study compared several machine learning architectures:
* **Linear Models:** ElasticNet (EN) for regularized baseline performance.
* **Ensemble Methods:** Gradient Boosting (GBM), Random Forest (RF), and XGBoost to capture non-linearities.

## 📊 Diagnostics: The "Concept Drift" Proof
The most significant part of this research was the diagnosis of model failure in specific folds (e.g., Fold 5).

* **Correlation Instability:** The relationship between `qqq_roc_4` and the target shifted from **-0.0426** (training) to **-0.192** (testing).
* **Sign Flips:** Identified 4 key features where the relationship with market returns completely reversed direction.
* **Drift Impact:** The average absolute change in correlation reached **0.1646**, proving that the market regime had fundamentally "re-wired" its logic, leading to degraded model performance.
