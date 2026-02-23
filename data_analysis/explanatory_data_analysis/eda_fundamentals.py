import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("EDA FRAMEWORK - THE 7-STEP PROCESS")
print("="*70)

np.random.seed(42)
data = {
    'ID': range(1, 101),
    'Age': np.random.randint(18, 70, 100),
    'Salary': np.random.normal(60000, 20000, 100),
    'Experience': np.random.randint(0, 20, 100),
    'Department': np.random.choice(['IT', 'HR', 'Sales', 'Finance'], 100),
    'Performance': np.random.choice(['Excellent', 'Good', 'Average', 'Poor'], 100),
    'City': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Chennai'], 100)
}

# Add some missing values
data['Salary'][np.random.choice(100, 5, replace=False)] = np.nan

df = pd.DataFrame(data)

# STEP 1: FIRST LOOK AT DATA

print("\n" + "="*70)
print("STEP 1: FIRST LOOK AT DATA")
print("="*70)

print("\n📊 Dataset Shape:")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n📋 First 5 Rows:")
print(df.head())

print("\n📋 Last 5 Rows:")
print(df.tail())

print("\n🔍 Random Sample (5 rows):")
print(df.sample(5))

# STEP 2: DATA TYPES AND INFO
print("\n" + "="*70)
print("STEP 2: DATA TYPES AND INFORMATION")
print("="*70)

print("\n📝 Dataset Info:")
df.info()

print("\n📊 Column Data Types:")
print(df.dtypes)

print("\n🔢 Memory Usage:")
print(f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")

# STEP 3: MISSING VALUES ANALYSIS
print("\n" + "="*70)
print("STEP 3: MISSING VALUES ANALYSIS")
print("="*70)

missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing_Count': missing,
    'Percentage': missing_pct
})

print("\n❌ Missing Values:")
print(missing_df[missing_df['Missing_Count'] > 0])

# STEP 4: STATISTICAL SUMMARY
print("\n" + "="*70)
print("STEP 4: STATISTICAL SUMMARY")
print("="*70)

print("\n📈 Numerical Columns Summary:")
print(df.describe())

print("\n📊 Categorical Columns Summary:")
print(df.describe(include='object'))

# Custom statistics
print("\n📊 Custom Statistics:")
for col in df.select_dtypes(include=[np.number]).columns:
    print(f"\n{col}:")
    print(f"  Mean: {df[col].mean():.2f}")
    print(f"  Median: {df[col].median():.2f}")
    print(f"  Std Dev: {df[col].std():.2f}")
    print(f"  Min: {df[col].min():.2f}")
    print(f"  Max: {df[col].max():.2f}")
    print(f"  Range: {df[col].max() - df[col].min():.2f}")

# STEP 5: DISTRIBUTION ANALYSIS
print("\n" + "="*70)
print("STEP 5: DISTRIBUTION ANALYSIS")
print("="*70)

# Value counts for categorical
print("\n📊 Department Distribution:")
print(df['Department'].value_counts())
print("\nPercentages:")
print(df['Department'].value_counts(normalize=True) * 100)

# Binning numerical data
age_bins = [0, 25, 35, 50, 100]
age_labels = ['Young', 'Mid', 'Senior', 'Veteran']
df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)

print("\n📊 Age Group Distribution:")
print(df['Age_Group'].value_counts())

# STEP 6: OUTLIER DETECTION
print("\n" + "="*70)
print("STEP 6: OUTLIER DETECTION")
print("="*70)

# IQR method
for col in ['Age', 'Salary', 'Experience']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    print(f"\n{col}:")
    print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"  Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")

# STEP 7: CORRELATION ANALYSIS
print("\n" + "="*70)
print("STEP 7: CORRELATION ANALYSIS")
print("="*70)

# Correlation matrix
numeric_df = df.select_dtypes(include=[np.number])
correlation = numeric_df.corr()

print("\n🔗 Correlation Matrix:")
print(correlation)

# Find strong correlations
print("\n Strong Correlations (|r| > 0.5):")
for i in range(len(correlation.columns)):
    for j in range(i+1, len(correlation.columns)):
        if abs(correlation.iloc[i, j]) > 0.5:
            print(f"{correlation.columns[i]} <-> {correlation.columns[j]}: {correlation.iloc[i, j]:.3f}")

# ==================================================================
# VISUALIZATION
# ==================================================================

print("\n" + "="*70)
print("CREATING EDA VISUALIZATIONS...")
print("="*70)

fig = plt.figure(figsize=(16, 12))

# 1. Age Distribution
plt.subplot(3, 3, 1)
plt.hist(df['Age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Age Distribution', fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# 2. Salary Distribution
plt.subplot(3, 3, 2)
plt.hist(df['Salary'].dropna(), bins=20, color='lightgreen', edgecolor='black')
plt.title('Salary Distribution', fontweight='bold')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# 3. Department Count
plt.subplot(3, 3, 3)
dept_counts = df['Department'].value_counts()
plt.bar(dept_counts.index, dept_counts.values, color='coral')
plt.title('Employees by Department', fontweight='bold')
plt.xlabel('Department')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')

# 4. Performance Distribution
plt.subplot(3, 3, 4)
perf_counts = df['Performance'].value_counts()
plt.pie(perf_counts.values, labels=perf_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Performance Distribution', fontweight='bold')

# 5. Age vs Salary Scatter
plt.subplot(3, 3, 5)
plt.scatter(df['Age'], df['Salary'], alpha=0.5, c=df['Experience'], cmap='viridis')
plt.colorbar(label='Experience')
plt.title('Age vs Salary', fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.grid(True, alpha=0.3)

# 6. Experience Distribution
plt.subplot(3, 3, 6)
plt.boxplot(df['Experience'])
plt.title('Experience Distribution (Box Plot)', fontweight='bold')
plt.ylabel('Years')
plt.grid(True, alpha=0.3)

# 7. Correlation Heatmap
plt.subplot(3, 3, 7)
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap', fontweight='bold')

# 8. Salary by Department
plt.subplot(3, 3, 8)
df.boxplot(column='Salary', by='Department', ax=plt.gca())
plt.title('Salary by Department', fontweight='bold')
plt.suptitle('')  # Remove automatic title
plt.xlabel('Department')
plt.ylabel('Salary')

# 9. Age Group Distribution
plt.subplot(3, 3, 9)
age_group_counts = df['Age_Group'].value_counts().sort_index()
plt.bar(age_group_counts.index, age_group_counts.values, color='steelblue')
plt.title('Age Group Distribution', fontweight='bold')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('eda_fundamentals_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved eda_fundamentals_analysis.png")

print("\n" + "="*70)
print("✅ EDA FUNDAMENTALS COMPLETE!")
print("="*70)