import pandas as pd
import numpy as np

print("="*70)
print("FEATURE ENGINEERING - CATEGORICAL FEATURES")
print("="*70)

# Sample data
np.random.seed(42)
df = pd.DataFrame({
    'employee_id': range(1, 101),
    'department': np.random.choice(['IT', 'HR', 'Sales', 'Finance', 'Marketing'], 100),
    'city': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Pune'], 100),
    'education': np.random.choice(['Bachelor', 'Master', 'PhD'], 100),
    'performance': np.random.choice(['Poor', 'Average', 'Good', 'Excellent'], 100),
    'salary': np.random.randint(40000, 120000, 100)
})

print("\nOriginal Data:")
print(df.head(10))
print()

# ==================================================================
# TECHNIQUE 1: LABEL ENCODING (Ordinal)
# ==================================================================

print("="*70)
print("TECHNIQUE 1: LABEL ENCODING")
print("="*70)

# Manual mapping for ordinal data
education_map = {'Bachelor': 1, 'Master': 2, 'PhD': 3}
df['education_encoded'] = df['education'].map(education_map)

performance_map = {'Poor': 1, 'Average': 2, 'Good': 3, 'Excellent': 4}
df['performance_encoded'] = df['performance'].map(performance_map)

print("\nLabel Encoding (Ordinal):")
print(df[['education', 'education_encoded', 'performance', 'performance_encoded']].head())
print()

# ==================================================================
# TECHNIQUE 2: ONE-HOT ENCODING
# ==================================================================

print("="*70)
print("TECHNIQUE 2: ONE-HOT ENCODING")
print("="*70)

# One-hot encoding for nominal categories
dept_dummies = pd.get_dummies(df['department'], prefix='dept')
city_dummies = pd.get_dummies(df['city'], prefix='city')

df_encoded = pd.concat([df, dept_dummies, city_dummies], axis=1)

print("\nOne-Hot Encoded Columns:")
print(dept_dummies.head())
print()

# ==================================================================
# TECHNIQUE 3: FREQUENCY ENCODING
# ==================================================================

print("="*70)
print("TECHNIQUE 3: FREQUENCY ENCODING")
print("="*70)

# Count how many times each category appears
dept_counts = df['department'].value_counts()
df['dept_frequency'] = df['department'].map(dept_counts)

city_counts = df['city'].value_counts()
df['city_frequency'] = df['city'].map(city_counts)

print("\nFrequency Encoding:")
print(df[['department', 'dept_frequency', 'city', 'city_frequency']].head(10))
print()

# ==================================================================
# TECHNIQUE 4: TARGET ENCODING (Mean Encoding)
# ==================================================================

print("="*70)
print("TECHNIQUE 4: TARGET ENCODING")
print("="*70)

# Average target value per category
dept_mean_salary = df.groupby('department')['salary'].mean()
df['dept_mean_salary'] = df['department'].map(dept_mean_salary)

city_mean_salary = df.groupby('city')['salary'].mean()
df['city_mean_salary'] = df['city'].map(city_mean_salary)

print("\nTarget Encoding (Mean Salary):")
print(df[['department', 'dept_mean_salary', 'city', 'city_mean_salary']].head(10))
print()

# ==================================================================
# TECHNIQUE 5: BINARY ENCODING
# ==================================================================

print("="*70)
print("TECHNIQUE 5: BINARY ENCODING")
print("="*70)

# Create binary flags
df['is_it'] = (df['department'] == 'IT').astype(int)
df['is_mumbai'] = (df['city'] == 'Mumbai').astype(int)
df['has_advanced_degree'] = df['education'].isin(['Master', 'PhD']).astype(int)
df['is_high_performer'] = df['performance'].isin(['Good', 'Excellent']).astype(int)

print("\nBinary Features:")
print(df[['department', 'is_it', 'city', 'is_mumbai', 'has_advanced_degree']].head(10))
print()

# ==================================================================
# TECHNIQUE 6: COMBINING CATEGORIES
# ==================================================================

print("="*70)
print("TECHNIQUE 6: COMBINING CATEGORIES")
print("="*70)

# Group rare categories together
df['city_grouped'] = df['city'].replace({
    'Chennai': 'Other',
    'Pune': 'Other'
})

# Combine categories
df['dept_city'] = df['department'] + '_' + df['city']

print("\nCombined Categories:")
print(df[['department', 'city', 'dept_city']].head(10))
print()

print("="*70)
print("✅ CATEGORICAL FEATURE ENGINEERING COMPLETE!")
print("="*70)