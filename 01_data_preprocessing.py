"""
===============================================================================
Sri Lanka Weather Prediction - Data Preprocessing & EDA
===============================================================================
Problem: Predict whether it will rain the next day in Sri Lankan cities
         based on historical weather data.

Dataset: Sri Lanka Weather Dataset from Kaggle
         - Source: Open-Meteo API historical weather data
         - Coverage: Multiple cities in Sri Lanka from 2010 onwards
         - ~147,000 daily weather observations across 27 cities
         
Target Variable: 'will_rain_tomorrow' (binary: 1 = rain, 0 = no rain)
         - Derived from next day's precipitation_sum > 0

Features: Temperature, apparent temperature, wind, radiation, 
          precipitation, evapotranspiration, and temporal features.

Preprocessing Steps:
    1. Data loading and initial exploration
    2. Handling missing values
    3. Feature engineering (temporal features, lag features)
    4. Encoding categorical variables
    5. Creating the target variable
    6. Normalization/Scaling
    7. Train/Validation/Test split
    8. Saving processed data
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')

# Create output directories
os.makedirs('plots', exist_ok=True)
os.makedirs('processed_data', exist_ok=True)

print("=" * 70)
print("STEP 1: DATA LOADING & INITIAL EXPLORATION")
print("=" * 70)

# Load the dataset
df = pd.read_csv('SriLanka_Weather_Dataset.csv')

print(f"\nDataset Shape: {df.shape}")
print(f"Number of Rows: {df.shape[0]:,}")
print(f"Number of Columns: {df.shape[1]}")
print(f"\nColumn Names:\n{list(df.columns)}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nFirst 5 Rows:")
print(df.head())
print(f"\nBasic Statistics:")
print(df.describe())

# Check unique cities
print(f"\nUnique Cities ({df['city'].nunique()}): {df['city'].unique()}")
print(f"\nDate Range: {df['time'].min()} to {df['time'].max()}")

# ---------------------------------------------------------------
# STEP 2: MISSING VALUES ANALYSIS & HANDLING
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: MISSING VALUES ANALYSIS & HANDLING")
print("=" * 70)

missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct
}).sort_values('Missing %', ascending=False)
print("\nMissing Values Summary:")
print(missing_df[missing_df['Missing Count'] > 0])

# Visualize missing values
fig, ax = plt.subplots(figsize=(12, 6))
cols_with_missing = missing_df[missing_df['Missing Count'] > 0].index.tolist()
if cols_with_missing:
    missing_df.loc[cols_with_missing, 'Missing %'].plot(kind='bar', ax=ax, color='#e74c3c')
    ax.set_title('Missing Values by Column (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage Missing')
    plt.tight_layout()
    plt.savefig('plots/01_missing_values.png', dpi=150, bbox_inches='tight')
    print("\n✓ Missing values plot saved to plots/01_missing_values.png")
else:
    print("\n✓ No missing values found in the dataset!")
plt.close()

# Drop rows with missing values (if any) - they are typically minimal
initial_rows = len(df)
df.dropna(inplace=True)
print(f"\nRows before cleaning: {initial_rows:,}")
print(f"Rows after cleaning: {len(df):,}")
print(f"Rows dropped: {initial_rows - len(df):,}")

# ---------------------------------------------------------------
# STEP 3: DATA TYPE CONVERSION
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: DATA TYPE CONVERSION")
print("=" * 70)

# Convert time to datetime
df['time'] = pd.to_datetime(df['time'])

# Convert sunrise/sunset to datetime
df['sunrise'] = pd.to_datetime(df['sunrise'])
df['sunset'] = pd.to_datetime(df['sunset'])

# Calculate daylight hours
df['daylight_hours'] = (df['sunset'] - df['sunrise']).dt.total_seconds() / 3600
print(f"Daylight hours range: {df['daylight_hours'].min():.2f} - {df['daylight_hours'].max():.2f}")

print("✓ Datetime conversions completed")

# ---------------------------------------------------------------
# STEP 4: FEATURE ENGINEERING
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: FEATURE ENGINEERING")
print("=" * 70)

# 4.1 Temporal Features
df['year'] = df['time'].dt.year
df['month'] = df['time'].dt.month
df['day_of_year'] = df['time'].dt.dayofyear
df['day_of_week'] = df['time'].dt.dayofweek
df['quarter'] = df['time'].dt.quarter
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# 4.2 Monsoon season encoding (important for Sri Lanka)
# Northeast Monsoon: Dec-Feb, Southwest Monsoon: May-Sep
# Inter-monsoon periods: Mar-Apr, Oct-Nov
def get_season(month):
    if month in [12, 1, 2]:
        return 'NE_Monsoon'
    elif month in [3, 4]:
        return 'Inter_Monsoon_1'
    elif month in [5, 6, 7, 8, 9]:
        return 'SW_Monsoon'
    else:
        return 'Inter_Monsoon_2'

df['season'] = df['month'].apply(get_season)
print("✓ Temporal features created: year, month, day_of_year, day_of_week, quarter, is_weekend, season")

# 4.3 Temperature-based features
df['temp_range'] = df['temperature_2m_max'] - df['temperature_2m_min']
df['apparent_temp_range'] = df['apparent_temperature_max'] - df['apparent_temperature_min']
df['temp_diff_apparent'] = df['temperature_2m_mean'] - df['apparent_temperature_mean']
df['heat_index'] = df['apparent_temperature_max'] - df['temperature_2m_max']
print("✓ Temperature-derived features created")

# 4.4 Wind features
df['wind_gust_ratio'] = df['windgusts_10m_max'] / (df['windspeed_10m_max'] + 0.001)
print("✓ Wind-derived features created")

# 4.5 Precipitation features
df['is_rainy'] = (df['precipitation_sum'] > 0).astype(int)
df['heavy_rain'] = (df['precipitation_sum'] > 20).astype(int)
print("✓ Precipitation-derived features created")

# 4.6 Create TARGET VARIABLE: Will it rain tomorrow?
df = df.sort_values(['city', 'time']).reset_index(drop=True)
df['will_rain_tomorrow'] = df.groupby('city')['is_rainy'].shift(-1)
# Drop last row per city (no next-day data)
df.dropna(subset=['will_rain_tomorrow'], inplace=True)
df['will_rain_tomorrow'] = df['will_rain_tomorrow'].astype(int)

print(f"\n✓ Target variable 'will_rain_tomorrow' created")
print(f"  Rain tomorrow (1): {df['will_rain_tomorrow'].sum():,} ({df['will_rain_tomorrow'].mean()*100:.1f}%)")
print(f"  No rain tomorrow (0): {(df['will_rain_tomorrow']==0).sum():,} ({(1-df['will_rain_tomorrow'].mean())*100:.1f}%)")

# ---------------------------------------------------------------
# STEP 5: EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 70)

# 5.1 Target distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Pie chart
labels = ['No Rain Tomorrow', 'Rain Tomorrow']
sizes = [df['will_rain_tomorrow'].value_counts()[0], df['will_rain_tomorrow'].value_counts()[1]]
colors = ['#3498db', '#e74c3c']
axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, 
            textprops={'fontsize': 12})
axes[0].set_title('Target Distribution', fontsize=14, fontweight='bold')

# Bar chart
df['will_rain_tomorrow'].value_counts().plot(kind='bar', color=colors, ax=axes[1])
axes[1].set_title('Target Class Counts', fontsize=14, fontweight='bold')
axes[1].set_xticklabels(['No Rain', 'Rain'], rotation=0)
axes[1].set_ylabel('Count')
plt.tight_layout()
plt.savefig('plots/02_target_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Target distribution plot saved")

# 5.2 Temperature distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, col in enumerate(['temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean', 'temp_range']):
    ax = axes[idx // 2, idx % 2]
    df[col].hist(bins=50, ax=ax, color='#2ecc71', edgecolor='white', alpha=0.8)
    ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
plt.suptitle('Temperature Feature Distributions', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plots/03_temperature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Temperature distribution plot saved")

# 5.3 Monthly precipitation pattern
fig, ax = plt.subplots(figsize=(12, 6))
monthly_precip = df.groupby('month')['precipitation_sum'].mean()
monthly_precip.plot(kind='bar', color='#3498db', ax=ax, edgecolor='white')
ax.set_title('Average Monthly Precipitation in Sri Lanka', fontsize=14, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Average Precipitation (mm)')
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], rotation=45)
plt.tight_layout()
plt.savefig('plots/04_monthly_precipitation.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Monthly precipitation plot saved")

# 5.4 Rain probability by season
fig, ax = plt.subplots(figsize=(10, 6))
season_rain = df.groupby('season')['will_rain_tomorrow'].mean() * 100
season_order = ['NE_Monsoon', 'Inter_Monsoon_1', 'SW_Monsoon', 'Inter_Monsoon_2']
season_rain = season_rain.reindex(season_order)
bars = ax.bar(season_rain.index, season_rain.values, 
              color=['#e74c3c', '#f39c12', '#3498db', '#2ecc71'], edgecolor='white')
ax.set_title('Rain Tomorrow Probability by Season', fontsize=14, fontweight='bold')
ax.set_ylabel('Probability of Rain Tomorrow (%)')
ax.set_xlabel('Season')
for bar, val in zip(bars, season_rain.values):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/05_rain_by_season.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Rain by season plot saved")

# 5.5 City-wise rain probability
fig, ax = plt.subplots(figsize=(14, 6))
city_rain = df.groupby('city')['will_rain_tomorrow'].mean().sort_values(ascending=False) * 100
city_rain.plot(kind='bar', ax=ax, color='#9b59b6', edgecolor='white')
ax.set_title('Rain Tomorrow Probability by City', fontsize=14, fontweight='bold')
ax.set_ylabel('Probability of Rain Tomorrow (%)')
ax.set_xlabel('City')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plots/06_rain_by_city.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Rain by city plot saved")

# 5.6 Correlation Heatmap
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Remove identifiers and keep key features for correlation
key_features = ['temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
                'apparent_temperature_max', 'apparent_temperature_min',
                'shortwave_radiation_sum', 'precipitation_sum', 'rain_sum',
                'precipitation_hours', 'windspeed_10m_max', 'windgusts_10m_max',
                'et0_fao_evapotranspiration', 'daylight_hours', 'temp_range',
                'wind_gust_ratio', 'will_rain_tomorrow']

fig, ax = plt.subplots(figsize=(16, 12))
corr_matrix = df[key_features].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, ax=ax,
            annot_kws={'size': 8})
ax.set_title('Correlation Heatmap of Key Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/07_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Correlation heatmap saved")

# 5.7 Box plots for key features by rain/no rain
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
box_features = ['temperature_2m_mean', 'precipitation_sum', 'windspeed_10m_max',
                'shortwave_radiation_sum', 'precipitation_hours', 'et0_fao_evapotranspiration']
for idx, col in enumerate(box_features):
    ax = axes[idx // 3, idx % 3]
    df.boxplot(column=col, by='will_rain_tomorrow', ax=ax,
               patch_artist=True,
               boxprops=dict(facecolor='#3498db', alpha=0.7),
               medianprops=dict(color='red', linewidth=2))
    ax.set_title(col, fontsize=11, fontweight='bold')
    ax.set_xlabel('Will Rain Tomorrow')
    ax.set_ylabel(col)
fig.suptitle('Feature Distributions by Rain/No Rain Tomorrow', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plots/08_boxplots_by_target.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Box plots saved")

# ---------------------------------------------------------------
# STEP 6: ENCODING CATEGORICAL VARIABLES
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6: ENCODING CATEGORICAL VARIABLES")
print("=" * 70)

# Encode city using Label Encoding
le_city = LabelEncoder()
df['city_encoded'] = le_city.fit_transform(df['city'])
print(f"✓ City encoded: {dict(zip(le_city.classes_, le_city.transform(le_city.classes_)))}")

# Encode season using Label Encoding  
le_season = LabelEncoder()
df['season_encoded'] = le_season.fit_transform(df['season'])
print(f"✓ Season encoded: {dict(zip(le_season.classes_, le_season.transform(le_season.classes_)))}")

# Encode country (all Sri Lanka, but kept for completeness)
le_country = LabelEncoder()
df['country_encoded'] = le_country.fit_transform(df['country'])

# ---------------------------------------------------------------
# STEP 7: FEATURE SELECTION
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 7: FEATURE SELECTION")
print("=" * 70)

# Select features for modeling
feature_columns = [
    # Original weather features
    'weathercode',
    'temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
    'apparent_temperature_max', 'apparent_temperature_min', 'apparent_temperature_mean',
    'shortwave_radiation_sum',
    'precipitation_sum', 'rain_sum', 'snowfall_sum',
    'precipitation_hours',
    'windspeed_10m_max', 'windgusts_10m_max', 'winddirection_10m_dominant',
    'et0_fao_evapotranspiration',
    'elevation',
    
    # Engineered features
    'daylight_hours',
    'month', 'day_of_year', 'day_of_week', 'quarter', 'is_weekend',
    'temp_range', 'apparent_temp_range', 'temp_diff_apparent', 'heat_index',
    'wind_gust_ratio',
    'is_rainy', 'heavy_rain',
    'city_encoded', 'season_encoded',
]

target_column = 'will_rain_tomorrow'

# Verify all columns exist
missing_cols = [c for c in feature_columns if c not in df.columns]
if missing_cols:
    print(f"⚠ Missing columns: {missing_cols}")
    feature_columns = [c for c in feature_columns if c in df.columns]

print(f"Total features selected: {len(feature_columns)}")
print(f"Features: {feature_columns}")

X = df[feature_columns].copy()
y = df[target_column].copy()

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# ---------------------------------------------------------------
# STEP 8: NORMALIZATION / SCALING
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 8: NORMALIZATION / SCALING")
print("=" * 70)

# Apply StandardScaler for normalization
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=feature_columns,
    index=X.index
)

print("✓ StandardScaler applied (mean=0, std=1)")
print(f"\nScaled data statistics:\n{X_scaled.describe().round(3)}")

# ---------------------------------------------------------------
# STEP 9: TRAIN / VALIDATION / TEST SPLIT
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 9: TRAIN / VALIDATION / TEST SPLIT")
print("=" * 70)

# 70% Train, 15% Validation, 15% Test (stratified)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Training set:   {X_train.shape[0]:>7,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Validation set: {X_val.shape[0]:>7,} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"Test set:       {X_test.shape[0]:>7,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"\nTarget distribution in each set:")
for name, target in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
    rain_pct = target.mean() * 100
    print(f"  {name}: Rain={rain_pct:.1f}%, No Rain={100-rain_pct:.1f}%")

# ---------------------------------------------------------------
# STEP 10: SAVE PROCESSED DATA
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 10: SAVING PROCESSED DATA")
print("=" * 70)

# Save the splits
X_train.to_csv('processed_data/X_train.csv', index=False)
X_val.to_csv('processed_data/X_val.csv', index=False)
X_test.to_csv('processed_data/X_test.csv', index=False)
y_train.to_csv('processed_data/y_train.csv', index=False)
y_val.to_csv('processed_data/y_val.csv', index=False)
y_test.to_csv('processed_data/y_test.csv', index=False)

# Save also unscaled data for explainability
X_unscaled = df[feature_columns].copy()
X_train_unscaled = X_unscaled.loc[X_train.index]
X_test_unscaled = X_unscaled.loc[X_test.index]
X_train_unscaled.to_csv('processed_data/X_train_unscaled.csv', index=False)
X_test_unscaled.to_csv('processed_data/X_test_unscaled.csv', index=False)

# Save feature columns list and scaler
import joblib
joblib.dump(scaler, 'processed_data/scaler.pkl')
joblib.dump(le_city, 'processed_data/label_encoder_city.pkl')
joblib.dump(le_season, 'processed_data/label_encoder_season.pkl')
joblib.dump(feature_columns, 'processed_data/feature_columns.pkl')

print("✓ All processed data saved to 'processed_data/' directory")
print("  - X_train.csv, X_val.csv, X_test.csv")
print("  - y_train.csv, y_val.csv, y_test.csv")
print("  - X_train_unscaled.csv, X_test_unscaled.csv")
print("  - scaler.pkl, label_encoder_city.pkl, label_encoder_season.pkl")
print("  - feature_columns.pkl")

# ---------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("PREPROCESSING SUMMARY")
print("=" * 70)
print(f"""
Dataset: Sri Lanka Weather Dataset (Kaggle)
Source: Open-Meteo API historical weather data
Coverage: {df['city'].nunique()} cities in Sri Lanka
Date Range: {df['time'].min().strftime('%Y-%m-%d')} to {df['time'].max().strftime('%Y-%m-%d')}
Total Records: {len(df):,}

Preprocessing Steps Completed:
  1. ✓ Data loaded and explored
  2. ✓ Missing values handled (dropped rows with NaN)
  3. ✓ Datetime conversions (time, sunrise, sunset → daylight_hours)
  4. ✓ Feature engineering:
     - Temporal: year, month, day_of_year, day_of_week, quarter, is_weekend
     - Seasonal: Sri Lanka monsoon seasons (NE, SW, Inter-monsoon)
     - Temperature: ranges, differences, heat index
     - Wind: gust ratio
     - Precipitation: is_rainy, heavy_rain
  5. ✓ Target variable created: will_rain_tomorrow
  6. ✓ Categorical encoding (LabelEncoder for city, season)
  7. ✓ Feature selection ({len(feature_columns)} features)
  8. ✓ Normalization (StandardScaler)
  9. ✓ Train/Val/Test split (70/15/15, stratified)
 10. ✓ Data saved to processed_data/

EDA Plots Generated (saved to plots/):
  - 01_missing_values.png
  - 02_target_distribution.png
  - 03_temperature_distributions.png
  - 04_monthly_precipitation.png
  - 05_rain_by_season.png
  - 06_rain_by_city.png
  - 07_correlation_heatmap.png
  - 08_boxplots_by_target.png
""")

print("✓ Preprocessing complete! Run 02_model_training.py next.")
