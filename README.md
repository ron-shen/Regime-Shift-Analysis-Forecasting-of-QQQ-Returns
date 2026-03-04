# Regime-Shift Analysis & Forecasting of QQQ Returns

This repository contains a quantitative research project focused on predicting the **1-day forward open-to-open return of the QQQ ETF**. The project emphasizes the transition of market mechanics in the post-COVID era, specifically diagnosing model decay through the lens of **Concept Drift**.

## Project Overview
The core objective was to determine if alpha decay in a predictive model was caused by changes in feature distributions (**Covariate Shift**) or changes in the underlying relationship between features and the target (**Concept Drift**).

### Key Highlights:
* **Target:** 1-day forward open-to-open returns ($\frac{Open_{t+2}}{Open_{t+1}} - 1$).
* **Feature Set:** Multi-frequency data including Equities (QQQ), Macro indicators (Gold, Treasury Yields, DIX, GEX, Fed Reserve Repo, VIX), Forex (GBP/USD, USD/JPY), Treasury (2 year yield and 10 year - 2 year yield)
* **Performance:** Model performs better in post-covid market (profit factor = 1.05) over pre-covid market. Post-COVID market regimes favored mid-term trend indicators over short-term mean-reversion signals.
* **Risk Management:** Developed an Adversarial Filter that successfully reduced Max Drawdown by 5% in the 2023-2024 test regime.

## Technical Implementation

### Data Pipeline & Orchestration
* **Feature Engineering:** Automated generation of technical indicators (ROC, Volatility) and time-series features.
* **Pipeline:** Utilized `ColumnTransformer` and `StandardScaler` to ensure rigorous data preprocessing without leakage.
* **Cross-Validation:** Implemented a custom `SlidingWindowCV` (1,000 days training / 100 days testing) to simulate a real-world walk-forward trading environment.

### Modeling Approach
The study compared several machine learning architectures:
* **Linear Models:** ElasticNet (EN) for regularized baseline performance.
* **Ensemble Methods:** Gradient Boosting (GBM), Random Forest (RF), and XGBoost to capture non-linearities.

## Key Research Findings

### 1. Concept Drift vs. Covariate Shift
Through a comparative analysis of "Collapsed" vs. "Stable" folds, this project identifies that performance degradation is rarely caused by **Covariate Shift** ($P(X)$ change). Instead, it is driven by **Adversarial Concept Drift** ($P(y|X)$ change).

* **Logic Inversion (Sign Flip Ratio):** In failed folds, the Sign Flip Ratio reached **39%**, indicating a structural shift where the model’s learned logic was actively inverted by the market:
    $$\text{sgn}(\rho_{\text{train}}) \neq \text{sgn}(\rho_{\text{val}})$$

* **Systemic Decay (Adversarial Feature Ratio):** The average **Adversarial Feature Ratio reached 57%**, implying that 57% of the top 15 predictive features experienced a significant weakening or total reversal of their correlation with the target during the test period.

### 2. The Adversarial Filter (Kill Switch)
I implemented an filter to halt trading when adversarial concept drift is detected, functioning as a real-time risk management layer. The filter acts as a drawdown limiter. In the 2023-2024 test regime, it successfully reduced Max Drawdown (MDD) by 5%.

