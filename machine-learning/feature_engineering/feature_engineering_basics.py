import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

print("="*70)
print("FEATURE ENGINEERING - NUMERICAL FEATURES")
print("="*70)

# Sample data
np.random.seed(42)
df = pd.DataFrame({
    'age': np.random.randint(18, 70, 100),
    'income': np.random.randint(30000, 150000, 100),
    'experience': np.random.randint(0, 30, 100),
    'hours_worked': np.random.randint(20, 60, 100),
    'projects_completed': np.random.randint(5, 50, 100)
})

print("\nOriginal Data:")
print(df.head(10))
print()

# ==================================================================
# TECHNIQUE 1: ARITHMETIC FEATURES
# ==================================================================

print("="*70)
print("TECHNIQUE 1: ARITHMETIC FEATURES")
print("="*70)

# Addition
df['total_productivity'] = df['hours_worked'] + df['projects_completed']

# Subtraction
df['income_experience_gap'] = df['income'] - (df['experience'] * 1000)

# Multiplication
df['productivity_score'] = df['hours_worked'] * df['projects_completed']

# Division (ratio)
df['projects_per_hour'] = df['projects_completed'] / df['hours_worked']
df['income_per_experience'] = df['income'] / (df['experience'] + 1)  # +1 to avoid division by zero

print("\nNew Features Created:")
print(df[['hours_worked', 'projects_completed', 'projects_per_hour']].head())
print()

# ==================================================================
# TECHNIQUE 2: POLYNOMIAL FEATURES
# ==================================================================

print("="*70)
print("TECHNIQUE 2: POLYNOMIAL FEATURES")
print("="*70)

# Square
df['age_squared'] = df['age'] ** 2

# Cube
df['experience_cubed'] = df['experience'] ** 3

# Square root
df['income_sqrt'] = np.sqrt(df['income'])

# Interaction (product of two features)
df['age_experience_interaction'] = df['age'] * df['experience']

print("\nPolynomial Features:")
print(df[['age', 'age_squared', 'experience', 'experience_cubed']].head())
print()

# ==================================================================
# TECHNIQUE 3: LOGARITHMIC TRANSFORMATIONS
# ==================================================================

print("="*70)
print("TECHNIQUE 3: LOGARITHMIC TRANSFORMATIONS")
print("="*70)

# Log transformation (for skewed data)
df['log_income'] = np.log1p(df['income'])  # log1p = log(1 + x) to handle zeros

# Compare distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(df['income'], bins=30, color='skyblue', edgecolor='black')
axes[0].set_title('Original Income Distribution', fontweight='bold')
axes[0].set_xlabel('Income')
axes[0].set_ylabel('Frequency')

axes[1].hist(df['log_income'], bins=30, color='lightgreen', edgecolor='black')
axes[1].set_title('Log-Transformed Income', fontweight='bold')
axes[1].set_xlabel('Log(Income)')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('feature_eng_01_log_transform.png', dpi=300)
plt.close()
print("✓ Saved feature_eng_01_log_transform.png")
print()

# ==================================================================
# TECHNIQUE 4: BINNING (DISCRETIZATION)
# ==================================================================

print("="*70)
print("TECHNIQUE 4: BINNING/DISCRETIZATION")
print("="*70)

# Equal-width binning
df['age_bin'] = pd.cut(df['age'], bins=5, labels=['Very Young', 'Young', 'Middle', 'Senior', 'Veteran'])

# Custom bins
income_bins = [0, 50000, 75000, 100000, 200000]
income_labels = ['Low', 'Medium', 'High', 'Very High']
df['income_category'] = pd.cut(df['income'], bins=income_bins, labels=income_labels)

# Quantile-based binning
df['experience_quartile'] = pd.qcut(df['experience'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

print("\nBinning Examples:")
print(df[['age', 'age_bin', 'income', 'income_category']].head(10))
print()

# ==================================================================
# TECHNIQUE 5: SCALING/NORMALIZATION
# ==================================================================

print("="*70)
print("TECHNIQUE 5: SCALING & NORMALIZATION")
print("="*70)

# MinMax Scaling (0 to 1)
scaler_minmax = MinMaxScaler()
df['income_minmax'] = scaler_minmax.fit_transform(df[['income']])

# Standard Scaling (mean=0, std=1)
scaler_standard = StandardScaler()
df['income_standard'] = scaler_standard.fit_transform(df[['income']])

# Robust Scaling (resistant to outliers)
scaler_robust = RobustScaler()
df['income_robust'] = scaler_robust.fit_transform(df[['income']])

print("\nScaling Comparison:")
print(df[['income', 'income_minmax', 'income_standard', 'income_robust']].describe())
print()

# ==================================================================
# TECHNIQUE 6: AGGREGATION FEATURES
# ==================================================================

print("="*70)
print("TECHNIQUE 6: AGGREGATION FEATURES")
print("="*70)

# Rolling statistics
df_sorted = df.sort_values('age')
df_sorted['projects_rolling_mean'] = df_sorted['projects_completed'].rolling(window=5).mean()
df_sorted['projects_rolling_std'] = df_sorted['projects_completed'].rolling(window=5).std()

# Cumulative features
df_sorted['cumulative_projects'] = df_sorted['projects_completed'].cumsum()

print("\nAggregation Features:")
print(df_sorted[['age', 'projects_completed', 'projects_rolling_mean', 'cumulative_projects']].head(10))
print()

print("="*70)
print("NUMERICAL FEATURE ENGINEERING COMPLETE!")
print("="*70)