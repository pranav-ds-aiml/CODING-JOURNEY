import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

print("="*70)
print("COMPLETE FEATURE ENGINEERING PIPELINE")
print("Building ML-Ready Dataset from Raw Data")
print("="*70)

# ==================================================================
# LOAD TITANIC DATA (Our Example Dataset)
# ==================================================================

print("\n📊 Loading Titanic dataset...")
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
print(f"\nOriginal columns: {df.columns.tolist()}")
print()

# ==================================================================
# STEP 1: HANDLE MISSING VALUES
# ==================================================================

print("="*70)
print("STEP 1: HANDLING MISSING VALUES")
print("="*70)

print("\nMissing values before:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Age: Fill with median by class (logical imputation)
df['Age'] = df.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))

# Embarked: Fill with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Cabin: Too many missing, create binary flag instead
df['Has_Cabin'] = (~df['Cabin'].isnull()).astype(int)

# Fare: Fill with median
df['Fare'].fillna(df['Fare'].median(), inplace=True)

print("\nMissing values after:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print()

# ==================================================================
# STEP 2: CREATE NEW FEATURES
# ==================================================================

print("="*70)
print("STEP 2: CREATING NEW FEATURES")
print("="*70)

# Family size
df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
print(" Created Family_Size")

# Is alone
df['Is_Alone'] = (df['Family_Size'] == 1).astype(int)
print(" Created Is_Alone")

# Title from name
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
print(" Extracted Title from Name")

# Simplify titles
title_mapping = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
    'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
    'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
    'Capt': 'Rare', 'Sir': 'Rare'
}
df['Title'] = df['Title'].map(title_mapping)
df['Title'].fillna('Rare', inplace=True)
print(" Simplified Title categories")

# Age groups
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                         labels=['Child', 'Teen', 'Young_Adult', 'Adult', 'Senior'])
print(" Created Age_Group")

# Fare categories
df['Fare_Category'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'Very_High'])
print(" Created Fare_Category")

# Deck from cabin
df['Deck'] = df['Cabin'].str[0] if 'Cabin' in df.columns else 'Unknown'
df['Deck'].fillna('Unknown', inplace=True)
print(" Extracted Deck from Cabin")

# Ticket frequency (how many passengers with same ticket)
ticket_counts = df['Ticket'].value_counts()
df['Ticket_Frequency'] = df['Ticket'].map(ticket_counts)
print(" Created Ticket_Frequency")

print(f"\nNew features created! Now have {len(df.columns)} columns")
print()

# ==================================================================
# STEP 3: ENCODE CATEGORICAL FEATURES
# ==================================================================

print("="*70)
print("STEP 3: ENCODING CATEGORICAL FEATURES")
print("="*70)

# Binary encoding
df['Sex_Binary'] = (df['Sex'] == 'male').astype(int)
print(" Binary encoded Sex")

# Label encoding for ordinal
pclass_map = {1: 3, 2: 2, 3: 1}  # Higher class = higher value
df['Pclass_Encoded'] = df['Pclass'].map(pclass_map)
print(" Label encoded Pclass")

# One-hot encoding for nominal
embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
title_dummies = pd.get_dummies(df['Title'], prefix='Title')
age_group_dummies = pd.get_dummies(df['Age_Group'], prefix='AgeGroup')

df = pd.concat([df, embarked_dummies, title_dummies, age_group_dummies], axis=1)
print(" One-hot encoded Embarked, Title, Age_Group")

# Frequency encoding
embarked_freq = df['Embarked'].value_counts()
df['Embarked_Frequency'] = df['Embarked'].map(embarked_freq)
print(" Frequency encoded Embarked")

print()

# ==================================================================
# STEP 4: SCALE NUMERICAL FEATURES
# ==================================================================

print("="*70)
print("STEP 4: SCALING NUMERICAL FEATURES")
print("="*70)

numerical_features = ['Age', 'Fare', 'Family_Size', 'Ticket_Frequency']

scaler = StandardScaler()
for feature in numerical_features:
    df[f'{feature}_Scaled'] = scaler.fit_transform(df[[feature]])
    print(f" Scaled {feature}")

print()

# ==================================================================
# STEP 5: CREATE INTERACTION FEATURES
# ==================================================================

print("="*70)
print("STEP 5: INTERACTION FEATURES")
print("="*70)

df['Age_Pclass'] = df['Age'] * df['Pclass']
print(" Created Age × Pclass interaction")

df['Fare_Family'] = df['Fare'] * df['Family_Size']
print(" Created Fare × Family_Size interaction")

df['Sex_Pclass'] = df['Sex_Binary'] * df['Pclass']
print(" Created Sex × Pclass interaction")

print()

# ==================================================================
# STEP 6: SELECT FINAL FEATURES FOR ML
# ==================================================================

print("="*70)
print("STEP 6: SELECTING FEATURES FOR ML")
print("="*70)

# Features to use in model
feature_columns = [
    # Original engineered
    'Pclass', 'Sex_Binary', 'Age', 'Fare', 'Family_Size', 'Is_Alone',
    'Has_Cabin', 'Ticket_Frequency',
    
    # Scaled features
    'Age_Scaled', 'Fare_Scaled',
    
    # One-hot encoded
    'Embarked_C', 'Embarked_Q', 'Embarked_S',
    'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare',
    
    # Interactions
    'Age_Pclass', 'Fare_Family', 'Sex_Pclass'
]

# Create final datasets
X = df[feature_columns].copy()
y = df['Survived']

print(f"\n✓ Final feature set: {len(feature_columns)} features")
print(f"✓ Dataset shape: {X.shape}")
print(f"\nFeatures used:")
for i, col in enumerate(feature_columns, 1):
    print(f"  {i}. {col}")
print()

# ==================================================================
# STEP 7: TRAIN/TEST SPLIT
# ==================================================================

print("="*70)
print("STEP 7: TRAIN/TEST SPLIT")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")
print()

# ==================================================================
# STEP 8: SAVE PROCESSED DATA
# ==================================================================

print("="*70)
print("STEP 8: SAVING PROCESSED DATA")
print("="*70)

# Save train/test sets
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv('titanic_train_engineered.csv', index=False)
test_df.to_csv('titanic_test_engineered.csv', index=False)

print("✓ Saved titanic_train_engineered.csv")
print("✓ Saved titanic_test_engineered.csv")

# Feature importance analysis (correlation with target)
correlation_with_target = X_train.corrwith(y_train).abs().sort_values(ascending=False)

plt.figure(figsize=(10, 8))
correlation_with_target.head(15).plot(kind='barh', color='steelblue')
plt.title('Top 15 Features by Correlation with Survival', fontsize=14, fontweight='bold')
plt.xlabel('Absolute Correlation')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('feature_eng_02_feature_importance.png', dpi=300)
plt.close()
print(" Saved feature_eng_02_feature_importance.png")

# Feature summary
feature_summary = pd.DataFrame({
    'Feature': feature_columns,
    'Type': ['Original/Engineered'] * len(feature_columns),
    'Correlation': correlation_with_target[feature_columns].values
}).sort_values('Correlation', ascending=False, key=lambda x: abs(x))

feature_summary.to_csv('feature_summary.csv', index=False)
print(" Saved feature_summary.csv")

print("\n" + "="*70)
print(" COMPLETE FEATURE ENGINEERING PIPELINE FINISHED!")
print("="*70)

print("\n SUMMARY:")
print(f"• Started with {df.shape[1] - len(feature_columns)} original features")
print(f"• Created {len(feature_columns)} engineered features")
print(f"• Dataset ready for machine learning!")
print(f"• Top 3 features: {correlation_with_target.head(3).index.tolist()}")

print("\n Files created:")
print("1. titanic_train_engineered.csv - Training data (ML-ready)")
print("2. titanic_test_engineered.csv - Test data (ML-ready)")
print("3. feature_summary.csv - Feature analysis")
print("4. feature_eng_02_feature_importance.png - Feature importance chart")

print("\n Next Steps:")
print("✓ This data is now ready for machine learning models!")
print("✓ Week 3 we'll build models using these engineered features")
print("✓ Feature engineering can improve accuracy by 20-30%!")