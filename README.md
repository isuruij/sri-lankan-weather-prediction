# 🌧️ Sri Lanka Rain Prediction — Machine Learning Assignment

> **Goal:** Predict whether it will rain tomorrow in Sri Lanka using the XGBoost algorithm, with full explainability using XAI techniques and a Streamlit front-end.

---

## Project Structure

| File | Purpose |
|------|---------|
| `01_data_preprocessing.py` | Data loading, cleaning, EDA, feature engineering, splitting |
| `02_model_training.py` | XGBoost model training, hyperparameter tuning, evaluation |
| `03_explainability.py` | SHAP explainability analysis |
| `app.py` | Streamlit front-end for predictions & explanations |
| `SriLanka_Weather_Dataset.csv` | Raw dataset (Kaggle) |
| `processed_data/` | Cleaned & split data, scaler, encoders |
| `models/` | Trained XGBoost model & hyperparameters |
| `plots/` | All generated EDA and evaluation plots |

---

## How to Run

```bash
# Step 1: Install dependencies
py -m pip install pandas numpy scikit-learn matplotlib seaborn shap lime xgboost joblib streamlit

# Step 2: Run preprocessing
py 01_data_preprocessing.py

# Step 3: Train the model
py 02_model_training.py

# Step 4: Run explainability analysis
py 03_explainability.py

# Step 5: Launch the front-end (Bonus)
py -m streamlit run app.py
```

---

## Assignment Requirements & Implementation

### 1. Problem Definition & Dataset Collection (15 marks)

**Requirement** | **How It Was Implemented**
---|---
Clearly describe the problem and its relevance | The problem is **binary classification**: predicting whether it will rain the next day in Sri Lanka. This is relevant for agriculture, disaster preparedness, and daily planning in a tropical island nation heavily affected by monsoons.
Data source | **Sri Lanka Weather Dataset** from Kaggle, originally sourced from the Open-Meteo historical weather API. Covers **30 cities** across Sri Lanka from **2010-01-01 to 2023-06-16**.
Features and target variable | **32 features** including temperature (max, min, mean, apparent), precipitation, wind speed/gusts/direction, solar radiation, evapotranspiration, elevation, plus engineered features (season, temporal, rain indicators). **Target:** `will_rain_tomorrow` (1 = rain, 0 = no rain).
Size of the dataset | **147,480 rows × 24 columns** (raw), **147,450 rows × 32 features** after preprocessing. Split into Train (103,215 / 70%), Validation (22,117 / 15%), Test (22,118 / 15%).
Preprocessing done | ✅ Missing value handling (dropped NaN rows) ✅ Datetime conversion (time → year, month, day_of_year, day_of_week, quarter, is_weekend) ✅ Feature engineering (temperature ranges, heat index, wind gust ratio, monsoon season encoding) ✅ Label encoding for categorical variables (city, season) ✅ StandardScaler normalization (mean=0, std=1)
Ethical data use | Dataset is publicly available on Kaggle (Open-Meteo API weather data). Contains no personal or sensitive information — only meteorological measurements and geographic location data.

**Implementation:** See `01_data_preprocessing.py` (Steps 1–10)

**Plots Generated:**
- `plots/01_missing_values.png` — Missing value analysis
- `plots/02_target_distribution.png` — Class balance visualization
- `plots/03_temperature_distributions.png` — Feature distributions
- `plots/04_monthly_precipitation.png` — Monthly rainfall patterns
- `plots/05_rain_by_season.png` — Seasonal rain patterns
- `plots/06_rain_by_city.png` — City-wise rain probability
- `plots/07_correlation_heatmap.png` — Feature correlations
- `plots/08_boxplots_by_target.png` — Feature distributions by class

---

### 2. Selection of a New Machine Learning Algorithm (15 marks)

**Requirement** | **How It Was Implemented**
---|---
Choose an algorithm (avoid deep learning) | **XGBoost (Extreme Gradient Boosting)** — a gradient boosting framework that builds an ensemble of decision trees sequentially. It is NOT a deep learning model.
Why was this algorithm selected | XGBoost was selected because: **(1)** It excels on tabular/structured data like weather measurements, **(2)** It provides built-in feature importance for interpretability, **(3)** It includes L1/L2 regularization to prevent overfitting, **(4)** It handles missing values natively, **(5)** It supports parallel processing for fast training, and **(6)** It is widely used in industry and competitions for classification tasks.
How it differs from standard models | **vs. Decision Trees:** XGBoost uses an *ensemble* of many weak decision trees combined via gradient boosting, rather than a single tree. It also adds regularization (L1, L2, gamma) to prevent overfitting. **vs. Logistic Regression:** XGBoost captures complex *non-linear relationships* between features, while logistic regression assumes a linear decision boundary. **vs. k-NN:** XGBoost doesn't suffer from the *curse of dimensionality* and doesn't require distance computations across all training samples at prediction time. **vs. Random Forest:** XGBoost uses *sequential boosting* (each tree corrects errors from the previous one) instead of independent bagging. It also uses second-order gradient information (Hessian) for more precise optimization.

**Implementation:** See `02_model_training.py` (docstring at top of file explains algorithm choice)

---

### 3. Model Training and Evaluation (20 marks)

**Requirement** | **How It Was Implemented**
---|---
Train/validation/test split | **70% / 15% / 15%** split with stratification to preserve class proportions (Rain: 81.9%, No Rain: 18.1% in all sets). Implemented using `train_test_split` with `stratify=y`.
Hyperparameter choices | **GridSearchCV** with 3-fold cross-validation was used to search over: `n_estimators` (100, 200), `max_depth` (4, 6), `learning_rate` (0.05, 0.1), `min_child_weight` (1, 3), `subsample` (0.8), `colsample_bytree` (0.8). Additional regularization: `reg_alpha=0.1` (L1), `reg_lambda=1.0` (L2), `gamma=0.1` (minimum loss reduction).
Performance metrics used | **Accuracy**, **Precision**, **Recall**, **F1-Score**, **AUC-ROC**, **Average Precision**, **Confusion Matrix**, and **5-fold Cross-Validation**. Metrics evaluated on all three sets (train, validation, test) to check for overfitting.
Results obtained | **Test Set:** Accuracy ≈ 0.89, AUC-ROC ≈ 0.94. The small gap between train and test performance indicates the model generalizes well without significant overfitting. The 5-fold CV F1 score is consistent across folds, confirming robust performance.
Tables, graphs, and plots | ✅ Confusion Matrix (raw + normalized) ✅ ROC Curves (train/val/test overlay) ✅ Precision-Recall Curves ✅ Feature Importance bar chart ✅ Performance Comparison bar chart across all metrics and sets

**Implementation:** See `02_model_training.py` (Steps 1–7)

**Plots Generated:**
- `plots/09_confusion_matrix.png` — Confusion matrix (raw + normalized)
- `plots/10_roc_curve.png` — ROC curves for all sets
- `plots/11_precision_recall_curve.png` — Precision-Recall curves
- `plots/12_feature_importance.png` — XGBoost feature importance
- `plots/13_performance_comparison.png` — Metric comparison across sets

---

### 4. Explainability & Interpretation (20 marks)

The explainability method applied is **SHAP (SHapley Additive exPlanations)**.

SHAP uses `TreeExplainer` to compute Shapley values for 1,000 test samples, providing both global and local explanations of the model's behavior. The following visualizations were generated: a summary bar plot, a beeswarm plot, dependence plots for the top 4 features, and waterfall plots for individual predictions.

**What the model has learned (SHAP):**
SHAP analysis reveals that the model has learned strong precipitation persistence patterns — if it rained today, tomorrow is very likely to be rainy as well. The model also learned that specific temperature differentials between actual and apparent ("feels like") temperatures serve as effective humidity indicators. Seasonal patterns based on the day of the year capture Sri Lanka's monsoon cycles, and wind conditions signal the approach of weather systems that bring rainfall.

**Most influential features (SHAP):**
- `precipitation_hours` is the single most influential feature with the highest mean absolute SHAP value, far exceeding all other features.
- `precipitation_sum` is the second most important, reinforcing the precipitation persistence pattern.
- `temp_diff_apparent` ranks third — the gap between actual and apparent temperature acts as a humidity proxy that strongly influences rain probability.
- `day_of_year` captures monsoon seasonality. Higher SHAP values during May–September (SW Monsoon) and December–February (NE Monsoon) reflect real-world rainfall patterns.
- `weathercode`, `daylight_hours`, `rain_sum`, and `elevation` provide additional predictive signals.

**Domain knowledge alignment (SHAP):**
- ✅ The dominance of precipitation-related features aligns with the well-established meteorological principle that rainfall is often persistent — weather systems don't change abruptly.
- ✅ The model correctly identifies Sri Lanka's two monsoon seasons through the `day_of_year` feature, matching known NE Monsoon (Dec–Feb) and SW Monsoon (May–Sep) periods.
- ✅ Higher elevation cities (e.g., Badulla at 680m) show different rainfall patterns than coastal cities, which the model captures through the `elevation` feature — consistent with orographic rainfall effects.
- ✅ Solar radiation is inversely correlated with rain prediction, reflecting the physical reality that cloudy/rainy days receive less solar radiation.

**Implementation:** See `03_explainability.py`

**SHAP Plots Generated:**
- `plots/14_shap_summary_bar.png` — Global feature importance ranking
- `plots/15_shap_beeswarm.png` — Feature impact direction and magnitude
- `plots/16_shap_dependence.png` — Interaction effects for top features
- `plots/17_shap_waterfall_rain.png` — Individual rain prediction breakdown
- `plots/18_shap_waterfall_norain.png` — Individual no-rain prediction breakdown

---

### 5. Critical Discussion (10 marks)

#### Limitations of the Model
- **Class Imbalance:** The dataset is imbalanced (81.9% rain vs. 18.1% no rain), which may bias the model toward predicting rain more often. Techniques like SMOTE or class weighting could improve recall for the minority class.
- **Temporal Leakage Risk:** Features like `is_rainy` and `precipitation_sum` are derived from the same-day data. In a real deployment, these would need to come from real-time sensor readings.
- **Limited Feature Set:** The model doesn't include humidity, cloud cover, or atmospheric pressure, which are strong rainfall predictors. These weren't available in the dataset.

#### Data Quality Issues
- **Missing Values:** Some rows contained NaN values and were dropped. This could introduce a slight bias if missing data isn't random (e.g., sensors failing during extreme weather).
- **Temporal Coverage:** Data spans 2010–2023. Climate change may shift rainfall patterns, meaning the model could become less accurate over time without retraining.
- **Geographic Resolution:** The dataset covers 30 cities, but doesn't capture microclimates between cities, particularly in the central highlands.

#### Risks of Bias or Unfairness
- **Geographic Bias:** Cities in the wet zone (e.g., Colombo, Ratnapura) have more rain events, so the model may perform better for these locations than for dry-zone cities with fewer training examples.
- **Seasonal Bias:** The dataset may have unequal representation of different monsoon seasons depending on the exact date range.

#### Potential Real-World Impact and Ethical Considerations
- **Positive Impact:** Accurate rain prediction can help farmers plan irrigation and harvesting, reduce weather-related crop losses, improve disaster preparedness during monsoons, and assist urban planning for flood-prone areas.
- **Ethical Considerations:** If deployed for agricultural insurance or disaster relief allocation, prediction errors could disproportionately affect vulnerable communities. The model should be transparent about its confidence levels and should not be the sole basis for critical decisions.
- **Environmental Responsibility:** The model uses publicly available weather data and doesn't require additional resource-intensive data collection.

---

### 6. Report Quality & Technical Clarity (10 marks)

- All code is thoroughly documented with docstrings and step-by-step comments
- Each script prints clear progress indicators and formatted result summaries
- Plots use professional formatting with titles, labels, legends, and appropriate color schemes
- This README provides a comprehensive overview mapping each requirement to its implementation
- Code follows a logical pipeline: preprocessing → training → explainability → front-end

---

### 7. Bonus: Front-End Integration (10 marks)

**Requirement** | **How It Was Implemented**
---|---
Integrating the model into a front-end | ✅ Built a **Streamlit web application** (`app.py`) with 4 pages
Allowing users to input data | ✅ The **"Make Prediction"** page provides interactive sliders and dropdowns for all weather features (temperature, precipitation, wind, city, month)
View predictions | ✅ Predictions are displayed with a **confidence percentage** and a clear Rain/No Rain result with visual styling
View explanations | ✅ Each prediction includes **real-time SHAP explanations** showing which features drove the prediction, plus full access to all SHAP and LIME plots

**App Pages:**
1. **🌦️ Make Prediction** — Interactive input form with instant SHAP-based explanations
2. **📊 Model Performance** — Metrics dashboard with all evaluation plots
3. **🔍 Explainability** — Tabbed view of SHAP and LIME analyses
4. **📈 Data Insights** — All EDA visualizations from preprocessing

**Run command:** `py -m streamlit run app.py` → Opens at `http://localhost:8501`

---

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python 3.9 | Programming language |
| Pandas / NumPy | Data manipulation |
| Scikit-learn | Preprocessing, metrics, evaluation |
| XGBoost | Machine learning algorithm |
| SHAP | Explainability (Shapley values) |
| LIME | Local interpretability |
| Matplotlib / Seaborn | Visualization |
| Streamlit | Front-end web application |

---

## Dataset Information

- **Name:** Sri Lanka Weather Dataset
- **Source:** Kaggle (Open-Meteo historical weather API)
- **Coverage:** 30 cities across Sri Lanka
- **Time Range:** January 2010 – June 2023
- **Records:** 147,480 daily observations
- **Features:** Weather code, temperatures (actual + apparent), solar radiation, precipitation, wind speed/gusts/direction, evapotranspiration, sunrise/sunset, elevation, latitude/longitude, city
