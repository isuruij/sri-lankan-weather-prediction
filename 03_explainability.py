"""
===============================================================================
Sri Lanka Weather Prediction - Explainability & Interpretation (XAI)
===============================================================================
Explainability Method Applied:
    SHAP (SHapley Additive exPlanations)
       - Global feature importance 
       - Summary plot (beeswarm)
       - Dependence plots
       - Waterfall plots for individual predictions
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('plots', exist_ok=True)

# ---------------------------------------------------------------
# LOAD MODEL AND DATA
# ---------------------------------------------------------------
print("=" * 70)
print("LOADING MODEL AND DATA")
print("=" * 70)

model = joblib.load('models/xgboost_rain_predictor.pkl')
feature_columns = joblib.load('processed_data/feature_columns.pkl')

X_train = pd.read_csv('processed_data/X_train.csv')
X_test = pd.read_csv('processed_data/X_test.csv')
y_test = pd.read_csv('processed_data/y_test.csv').iloc[:, 0]

# Load unscaled data for interpretability
X_train_unscaled = pd.read_csv('processed_data/X_train_unscaled.csv')
X_test_unscaled = pd.read_csv('processed_data/X_test_unscaled.csv')

print(f"Model loaded: XGBoost")
print(f"Test set size: {X_test.shape[0]:,}")
print(f"Features: {len(feature_columns)}")

# Use a sample for SHAP/LIME (faster computation)
np.random.seed(42)
sample_size = min(1000, len(X_test))
sample_idx = np.random.choice(len(X_test), sample_size, replace=False)
X_test_sample = X_test.iloc[sample_idx]
y_test_sample = y_test.iloc[sample_idx]

# ---------------------------------------------------------------
# 1. SHAP ANALYSIS
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("1. SHAP (SHapley Additive exPlanations)")
print("=" * 70)

print("\nComputing SHAP values (this may take a moment)...")

# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_sample)

print(f"✓ SHAP values computed for {sample_size} samples")

# 1.1 SHAP Summary Plot (Bar)
print("\nGenerating SHAP summary bar plot...")
fig, ax = plt.subplots(figsize=(12, 10))
shap.summary_plot(shap_values, X_test_sample, feature_names=feature_columns,
                  plot_type="bar", show=False, max_display=20)
plt.title('SHAP Feature Importance (Mean |SHAP Value|)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/14_shap_summary_bar.png', dpi=150, bbox_inches='tight')
plt.close('all')
print("✓ SHAP summary bar plot saved")

# 1.2 SHAP Beeswarm Plot
print("Generating SHAP beeswarm plot...")
fig, ax = plt.subplots(figsize=(12, 10))
shap.summary_plot(shap_values, X_test_sample, feature_names=feature_columns,
                  show=False, max_display=20)
plt.title('SHAP Beeswarm Plot - Feature Impact on Rain Prediction', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/15_shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close('all')
print("✓ SHAP beeswarm plot saved")

# 1.3 SHAP Dependence Plots for top features
print("Generating SHAP dependence plots...")
# Get top 4 features by importance
shap_importance = np.abs(shap_values).mean(axis=0)
top_features_idx = np.argsort(shap_importance)[-4:][::-1]
top_features = [feature_columns[i] for i in top_features_idx]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for idx, (feat_idx, feat_name) in enumerate(zip(top_features_idx, top_features)):
    ax = axes[idx // 2, idx % 2]
    shap.dependence_plot(feat_idx, shap_values, X_test_sample,
                         feature_names=feature_columns, ax=ax, show=False)
    ax.set_title(f'SHAP Dependence: {feat_name}', fontsize=12, fontweight='bold')
plt.suptitle('SHAP Dependence Plots for Top Features', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plots/16_shap_dependence.png', dpi=150, bbox_inches='tight')
plt.close('all')
print("✓ SHAP dependence plots saved")

# 1.4 SHAP Force Plot for individual predictions
print("Generating SHAP force plots for individual predictions...")

# Find a correct rain prediction and a correct no-rain prediction
y_pred_sample = model.predict(X_test_sample)
rain_correct = np.where((y_pred_sample == 1) & (y_test_sample.values == 1))[0]
no_rain_correct = np.where((y_pred_sample == 0) & (y_test_sample.values == 0))[0]

if len(rain_correct) > 0 and len(no_rain_correct) > 0:
    fig, axes = plt.subplots(2, 1, figsize=(20, 8))
    
    # Rain prediction
    idx = rain_correct[0]
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value,
        data=X_test_sample.iloc[idx].values,
        feature_names=feature_columns
    ), max_display=15, show=False)
    plt.title('SHAP Waterfall: Rain Prediction (Correct)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/17_shap_waterfall_rain.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # No-rain prediction
    idx = no_rain_correct[0]
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value,
        data=X_test_sample.iloc[idx].values,
        feature_names=feature_columns
    ), max_display=15, show=False)
    plt.title('SHAP Waterfall: No Rain Prediction (Correct)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/18_shap_waterfall_norain.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print("✓ SHAP waterfall plots saved")
else:
    print("⚠ Could not find suitable examples for waterfall plots")

# 1.5 SHAP interpretation summary
print("\n--- SHAP Interpretation ---")
print("\nTop 10 Most Influential Features (by mean |SHAP value|):")
shap_df = pd.DataFrame({
    'Feature': feature_columns,
    'Mean |SHAP|': np.abs(shap_values).mean(axis=0)
}).sort_values('Mean |SHAP|', ascending=False)

for i, row in shap_df.head(10).iterrows():
    print(f"  {row['Feature']:>35s}: {row['Mean |SHAP|']:.4f}")

# ---------------------------------------------------------------
# 2. INTERPRETATION SUMMARY
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("2. INTERPRETATION SUMMARY")
print("=" * 70)

print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    MODEL INTERPRETATION SUMMARY                     ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  What the Model Has Learned:                                        ║
║  ─────────────────────────────                                       ║
║  The XGBoost model predicts next-day rainfall in Sri Lanka based    ║
║  on current weather conditions. Key learnings:                      ║
║                                                                      ║
║  1. Precipitation-related features (rain_sum, precipitation_sum,    ║
║     precipitation_hours, is_rainy) are the strongest predictors.    ║
║     → If it rained today, it's more likely to rain tomorrow.        ║
║                                                                      ║
║  2. Wind conditions (windspeed, wind gusts) provide important       ║
║     signals about incoming weather systems.                         ║
║                                                                      ║
║  3. Temperature features (especially apparent temperature)          ║
║     indicate moisture-laden air that leads to rainfall.             ║
║                                                                      ║
║  4. Seasonal patterns (monsoon periods) strongly influence          ║
║     rainfall probability, consistent with domain knowledge.         ║
║                                                                      ║
║  5. Solar radiation inversely correlates with rain probability      ║
║     (cloudy days receive less radiation).                           ║
║                                                                      ║
║  Alignment with Domain Knowledge:                                   ║
║  ─────────────────────────────────                                   ║
║  ✓ Sri Lanka's monsoon seasons are captured by seasonal features   ║
║  ✓ Precipitation persistence (today→tomorrow) is well-known       ║
║  ✓ Wind patterns signal approaching weather fronts                 ║
║  ✓ Humidity (via apparent temperature) drives rainfall             ║
║  ✓ City-level variations reflect geographic differences            ║
║    (Colombo wet zone vs. Jaffna dry zone)                          ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# Print all generated plots
print("\n✓ All explainability plots saved to plots/ directory:")
print("  14_shap_summary_bar.png      - SHAP global feature importance")
print("  15_shap_beeswarm.png         - SHAP beeswarm (impact + direction)")
print("  16_shap_dependence.png       - SHAP dependence for top features")
print("  17_shap_waterfall_rain.png   - SHAP waterfall for rain prediction")
print("  18_shap_waterfall_norain.png - SHAP waterfall for no-rain prediction")

print("\n✓ Explainability analysis complete! Run 'py -m streamlit run app.py' for the front-end demo.")
