"""
===============================================================================
Sri Lanka Weather Prediction - Streamlit Front-End Application
===============================================================================
A web application that allows users to:
    1. Input weather conditions 
    2. Get rain predictions with confidence
    3. View SHAP-based explanations
    4. Explore model performance and feature importance
    
Run with: py -m streamlit run app.py
===============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Sri Lanka Rain Predictor",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------
# LOAD MODEL AND DATA
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load('models/xgboost_rain_predictor.pkl')
    scaler = joblib.load('processed_data/scaler.pkl')
    feature_columns = joblib.load('processed_data/feature_columns.pkl')
    results = joblib.load('models/results_summary.pkl')
    return model, scaler, feature_columns, results

@st.cache_resource
def load_shap_explainer(_model, X_bg):
    return shap.TreeExplainer(_model)

try:
    model, scaler, feature_columns, results_summary = load_model()
    X_train = pd.read_csv('processed_data/X_train.csv')
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"❌ Model not found. Please run the training scripts first.\nError: {e}")

# ---------------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(120deg, #1e3a5f 0%, #2980b9 50%, #27ae60 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .prediction-rain {
        background: linear-gradient(135deg, #3498db 0%, #2c3e50 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
    }
    .prediction-sun {
        background: linear-gradient(135deg, #f39c12 0%, #e74c3c 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------
st.markdown('<p class="main-header">🌧️ Sri Lanka Rain Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict tomorrow\'s rainfall using XGBoost & Explainable AI</p>', unsafe_allow_html=True)

if not model_loaded:
    st.stop()

# ---------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------
st.sidebar.title("🔧 Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["🌦️ Make Prediction", "📊 Model Performance", "🔍 Explainability", "📈 Data Insights"]
)

# ---------------------------------------------------------------
# PAGE 1: MAKE PREDICTION
# ---------------------------------------------------------------
if page == "🌦️ Make Prediction":
    st.header("🌦️ Rain Prediction")
    st.write("Enter today's weather conditions to predict if it will rain tomorrow.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🌡️ Temperature")
        temp_max = st.slider("Max Temperature (°C)", 15.0, 40.0, 30.0, 0.1)
        temp_min = st.slider("Min Temperature (°C)", 10.0, 35.0, 24.0, 0.1)
        temp_mean = (temp_max + temp_min) / 2
        apparent_max = st.slider("Apparent Max Temp (°C)", 15.0, 45.0, 34.0, 0.1)
        apparent_min = st.slider("Apparent Min Temp (°C)", 10.0, 40.0, 27.0, 0.1)
        apparent_mean = (apparent_max + apparent_min) / 2
    
    with col2:
        st.subheader("🌧️ Precipitation")
        precipitation = st.slider("Precipitation (mm)", 0.0, 100.0, 5.0, 0.1)
        rain = st.slider("Rain (mm)", 0.0, 100.0, 5.0, 0.1)
        precip_hours = st.slider("Precipitation Hours", 0.0, 24.0, 3.0, 0.5)
        snowfall = 0.0  # Sri Lanka doesn't get snow
        weathercode = st.selectbox("Weather Code", [0, 1, 2, 3, 51, 53, 55, 61, 63, 65],
                                   index=5, help="0=Clear, 1-3=Partly cloudy, 51-55=Drizzle, 61-65=Rain")
    
    with col3:
        st.subheader("💨 Wind & Other")
        windspeed = st.slider("Wind Speed (km/h)", 0.0, 50.0, 15.0, 0.1)
        windgusts = st.slider("Wind Gusts (km/h)", 0.0, 80.0, 35.0, 0.1)
        winddirection = st.slider("Wind Direction (°)", 0, 360, 220, 1)
        radiation = st.slider("Solar Radiation (MJ/m²)", 0.0, 30.0, 15.0, 0.1)
        et0 = st.slider("ET₀ Evapotranspiration", 0.0, 8.0, 3.5, 0.1)
    
    # Additional features
    st.subheader("📍 Location & Time")
    col4, col5 = st.columns(2)
    with col4:
        cities = ['Colombo', 'Kandy', 'Galle', 'Jaffna', 'Trincomalee', 
                  'Anuradhapura', 'Batticaloa', 'Matara', 'Badulla', 'Ratnapura']
        city = st.selectbox("City", cities, index=0)
        month = st.selectbox("Month", list(range(1, 13)), index=0)
    with col5:
        elevation_map = {'Colombo': 16, 'Kandy': 477, 'Galle': 12, 'Jaffna': 5,
                        'Trincomalee': 3, 'Anuradhapura': 89, 'Batticaloa': 3,
                        'Matara': 3, 'Badulla': 680, 'Ratnapura': 34}
        elevation = elevation_map.get(city, 16)
        st.metric("Elevation (m)", elevation)
        
        day_of_year = month * 30  # Approximate
        day_of_week = 3  # Wednesday default
    
    if st.button("🔮 Predict Tomorrow's Weather", type="primary", use_container_width=True):
        # Build feature vector
        temp_range = temp_max - temp_min
        apparent_temp_range = apparent_max - apparent_min
        temp_diff_apparent = temp_mean - apparent_mean
        heat_index = apparent_max - temp_max
        wind_gust_ratio = windgusts / (windspeed + 0.001)
        is_rainy = 1 if precipitation > 0 else 0
        heavy_rain = 1 if precipitation > 20 else 0
        daylight_hours = 12.0  # Sri Lanka near equator
        quarter = (month - 1) // 3 + 1
        is_weekend = 0
        
        # Season encoding
        if month in [12, 1, 2]:
            season_encoded = 2  # NE_Monsoon
        elif month in [3, 4]:
            season_encoded = 0  # Inter_Monsoon_1
        elif month in [5, 6, 7, 8, 9]:
            season_encoded = 3  # SW_Monsoon
        else:
            season_encoded = 1  # Inter_Monsoon_2
        
        # City encoding (approximate)
        city_map = {c: i for i, c in enumerate(sorted(cities))}
        city_encoded = city_map.get(city, 0)
        
        # Create feature array matching exact order
        feature_values = {
            'weathercode': weathercode,
            'temperature_2m_max': temp_max,
            'temperature_2m_min': temp_min,
            'temperature_2m_mean': temp_mean,
            'apparent_temperature_max': apparent_max,
            'apparent_temperature_min': apparent_min,
            'apparent_temperature_mean': apparent_mean,
            'shortwave_radiation_sum': radiation,
            'precipitation_sum': precipitation,
            'rain_sum': rain,
            'snowfall_sum': snowfall,
            'precipitation_hours': precip_hours,
            'windspeed_10m_max': windspeed,
            'windgusts_10m_max': windgusts,
            'winddirection_10m_dominant': winddirection,
            'et0_fao_evapotranspiration': et0,
            'elevation': elevation,
            'daylight_hours': daylight_hours,
            'month': month,
            'day_of_year': day_of_year,
            'day_of_week': day_of_week,
            'quarter': quarter,
            'is_weekend': is_weekend,
            'temp_range': temp_range,
            'apparent_temp_range': apparent_temp_range,
            'temp_diff_apparent': temp_diff_apparent,
            'heat_index': heat_index,
            'wind_gust_ratio': wind_gust_ratio,
            'is_rainy': is_rainy,
            'heavy_rain': heavy_rain,
            'city_encoded': city_encoded,
            'season_encoded': season_encoded,
        }
        
        X_input = pd.DataFrame([feature_values])[feature_columns]
        X_input_scaled = pd.DataFrame(scaler.transform(X_input), columns=feature_columns)
        
        # Predict
        prediction = model.predict(X_input_scaled)[0]
        probability = model.predict_proba(X_input_scaled)[0]
        
        st.markdown("---")
        
        # Display result
        col_result1, col_result2 = st.columns(2)
        
        with col_result1:
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-rain">
                    <h2>🌧️ RAIN EXPECTED TOMORROW</h2>
                    <h3>Confidence: {probability[1]*100:.1f}%</h3>
                    <p>Take an umbrella! ☂️</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-sun">
                    <h2>☀️ NO RAIN EXPECTED TOMORROW</h2>
                    <h3>Confidence: {probability[0]*100:.1f}%</h3>
                    <p>Enjoy the sunshine! 🌤️</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col_result2:
            # SHAP explanation for this prediction
            st.subheader("🔍 Why this prediction?")
            explainer_local = shap.TreeExplainer(model)
            shap_vals = explainer_local.shap_values(X_input_scaled)
            
            shap_explanation = pd.DataFrame({
                'Feature': feature_columns,
                'Value': X_input.iloc[0].values,
                'SHAP Value': shap_vals[0]
            }).sort_values('SHAP Value', key=abs, ascending=False).head(10)
            
            for _, row in shap_explanation.iterrows():
                direction = "↑ Rain" if row['SHAP Value'] > 0 else "↓ No Rain"
                color = "🔴" if row['SHAP Value'] > 0 else "🟢"
                st.write(f"{color} **{row['Feature']}** = {row['Value']:.2f} → {direction} ({row['SHAP Value']:+.4f})")

# ---------------------------------------------------------------
# PAGE 2: MODEL PERFORMANCE
# ---------------------------------------------------------------
elif page == "📊 Model Performance":
    st.header("📊 Model Performance Dashboard")
    
    # Metrics cards
    col1, col2, col3, col4, col5 = st.columns(5)
    metrics = [
        ("Accuracy", results_summary['test_accuracy']),
        ("Precision", results_summary['test_precision']),
        ("Recall", results_summary['test_recall']),
        ("F1-Score", results_summary['test_f1']),
        ("AUC-ROC", results_summary['test_auc_roc'])
    ]
    
    for col, (name, value) in zip([col1, col2, col3, col4, col5], metrics):
        col.metric(name, f"{value:.4f}")
    
    st.markdown("---")
    
    # Show plots
    col1, col2 = st.columns(2)
    
    plot_files = {
        'Confusion Matrix': 'plots/09_confusion_matrix.png',
        'ROC Curve': 'plots/10_roc_curve.png',
        'Precision-Recall Curve': 'plots/11_precision_recall_curve.png',
        'Feature Importance': 'plots/12_feature_importance.png',
        'Performance Comparison': 'plots/13_performance_comparison.png'
    }
    
    for i, (title, path) in enumerate(plot_files.items()):
        if os.path.exists(path):
            if i % 2 == 0:
                with col1:
                    st.subheader(title)
                    st.image(path)
            else:
                with col2:
                    st.subheader(title)
                    st.image(path)
    
    # Hyperparameters
    st.subheader("⚙️ Best Hyperparameters")
    params_df = pd.DataFrame(
        list(results_summary['best_params'].items()),
        columns=['Parameter', 'Value']
    )
    st.table(params_df)

# ---------------------------------------------------------------
# PAGE 3: EXPLAINABILITY
# ---------------------------------------------------------------
elif page == "🔍 Explainability":
    st.header("🔍 Model Explainability (XAI)")
    
    st.subheader("SHAP (SHapley Additive exPlanations)")
    st.write("""
    SHAP values show how each feature contributes to the prediction. 
    Positive SHAP values push the prediction toward **Rain**, while 
    negative values push toward **No Rain**.
    """)
    
    shap_plots = {
        'SHAP Summary (Bar)': 'plots/14_shap_summary_bar.png',
        'SHAP Beeswarm': 'plots/15_shap_beeswarm.png',
        'SHAP Dependence': 'plots/16_shap_dependence.png',
        'SHAP Waterfall (Rain)': 'plots/17_shap_waterfall_rain.png',
        'SHAP Waterfall (No Rain)': 'plots/18_shap_waterfall_norain.png',
    }
    
    for title, path in shap_plots.items():
        if os.path.exists(path):
            st.subheader(title)
            st.image(path, use_container_width=True)

# ---------------------------------------------------------------
# PAGE 4: DATA INSIGHTS
# ---------------------------------------------------------------
elif page == "📈 Data Insights":
    st.header("📈 Data Exploration & Insights")
    
    eda_plots = {
        'Target Distribution': 'plots/02_target_distribution.png',
        'Temperature Distributions': 'plots/03_temperature_distributions.png',
        'Monthly Precipitation': 'plots/04_monthly_precipitation.png',
        'Rain by Season': 'plots/05_rain_by_season.png',
        'Rain by City': 'plots/06_rain_by_city.png',
        'Correlation Heatmap': 'plots/07_correlation_heatmap.png',
        'Feature Distributions by Target': 'plots/08_boxplots_by_target.png',
    }
    
    for title, path in eda_plots.items():
        if os.path.exists(path):
            st.subheader(title)
            st.image(path, use_container_width=True)
            st.markdown("---")

# ---------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p>🌧️ Sri Lanka Rain Predictor | Machine Learning Assignment</p>
    <p>Built with XGBoost + SHAP + LIME + Streamlit</p>
    <p>Dataset: Sri Lanka Weather Dataset (Kaggle)</p>
</div>
""", unsafe_allow_html=True)
