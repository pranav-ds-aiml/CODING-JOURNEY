import pandas as pd
import numpy as np

print("="*70)
print("FEATURE ENGINEERING - DATE/TIME FEATURES")
print("="*70)

# Sample data
date_range = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
np.random.seed(42)

df = pd.DataFrame({
    'date': np.random.choice(date_range, 100),
    'sales': np.random.randint(1000, 10000, 100)
})

df = df.sort_values('date').reset_index(drop=True)

print("\nOriginal Data:")
print(df.head(10))
print()

# ==================================================================
# EXTRACT TIME COMPONENTS
# ==================================================================

print("="*70)
print("EXTRACTING TIME COMPONENTS")
print("="*70)

# Basic components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
df['day_name'] = df['date'].dt.day_name()
df['week_of_year'] = df['date'].dt.isocalendar().week
df['quarter'] = df['date'].dt.quarter

# Time-based features
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)

print("\nExtracted Date Features:")
print(df[['date', 'year', 'month', 'day', 'day_name', 'is_weekend']].head(10))
print()

# ==================================================================
# CYCLICAL FEATURES
# ==================================================================

print("="*70)
print("CYCLICAL FEATURES (Sine/Cosine Encoding)")
print("="*70)

# Month as cyclical feature (1-12 repeats)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Day of week as cyclical (0-6 repeats)
df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

print("\nCyclical Features:")
print(df[['month', 'month_sin', 'month_cos', 'day_of_week', 'day_of_week_sin']].head(10))
print()

# ==================================================================
# TIME-BASED AGGREGATIONS
# ==================================================================

print("="*70)
print("TIME-BASED AGGREGATIONS")
print("="*70)

# Rolling statistics
df['sales_rolling_7d_mean'] = df['sales'].rolling(window=7).mean()
df['sales_rolling_30d_mean'] = df['sales'].rolling(window=30).mean()
df['sales_rolling_7d_std'] = df['sales'].rolling(window=7).std()

# Lag features (previous values)
df['sales_lag_1'] = df['sales'].shift(1)
df['sales_lag_7'] = df['sales'].shift(7)
df['sales_lag_30'] = df['sales'].shift(30)

# Difference features
df['sales_diff_1d'] = df['sales'].diff(1)
df['sales_pct_change'] = df['sales'].pct_change()

print("\nTime-Based Features:")
print(df[['date', 'sales', 'sales_lag_1', 'sales_rolling_7d_mean', 'sales_diff_1d']].head(10))
print()

# ==================================================================
# ELAPSED TIME
# ==================================================================

print("="*70)
print("ELAPSED TIME FEATURES")
print("="*70)

# Days since first date
df['days_since_start'] = (df['date'] - df['date'].min()).dt.days

# Days until end
df['days_until_end'] = (df['date'].max() - df['date']).dt.days

print("\nElapsed Time Features:")
print(df[['date', 'days_since_start', 'days_until_end']].head(10))
print()

print("="*70)
print(" DATE/TIME FEATURE ENGINEERING COMPLETE!")
print("="*70)