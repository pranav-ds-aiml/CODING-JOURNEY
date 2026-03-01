import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import pickle

print("="*70)
print(" TITANIC SURVIVAL PREDICTION PROJECT ")
print("="*70)

# ==================================================================
# STEP 1: LOAD ENGINEERED FEATURES (from Day 10!)
# ==================================================================

print("\n" + "="*70)
print("STEP 1: LOAD DATA")
print("="*70)

# Load the engineered features we created on Day 10
try:
    train_df = pd.read_csv('../../week 2/day 10/titanic_train_engineered.csv')
    test_df = pd.read_csv('../../week 2/day 10/titanic_test_engineered.csv')
    print("✓ Loaded engineered Titanic data from Day 10!")
except:
    # If not available, load raw and do basic feature selection
    print("  Engineered data not found, loading raw Titanic data...")
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    
    # Basic feature engineering
    print(df.isnull().sum())
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    df['Sex_Binary'] = (df['Sex'] == 'male').astype(int)
    df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
    
    # Select features
    features = ['Pclass', 'Sex_Binary', 'Age', 'Fare', 'Family_Size']
    X = df[features]
    y = df['Survived']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df = train_df.fillna(train_df.median(numeric_only=True))
test_df = test_df.fillna(train_df.median(numeric_only=True))

print("\nNaN check after cleaning:")
print(train_df.isnull().sum())
print(test_df.isnull().sum())

print(f"\n✓ Training set: {len(train_df)} passengers")
print(f"✓ Test set: {len(test_df)} passengers")

# Separate features and target
X_train = train_df.drop('Survived', axis=1)
y_train = train_df['Survived']
X_test = test_df.drop('Survived', axis=1)
y_test = test_df['Survived']

print(f"\n Features: {X_train.shape[1]}")
print(f"   Samples: {X_train.shape[0]} train, {X_test.shape[0]} test")

# Class distribution
print(f"\n Survival Distribution:")
print(f"   Survived: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
print(f"   Died: {len(y_train)-y_train.sum()} ({(1-y_train.mean())*100:.1f}%)")

# ==================================================================
# STEP 2: TRAIN MODELS
# ==================================================================

print("\n" + "="*70)
print("STEP 2: TRAINING MODELS")
print("="*70)

# Model 1: Logistic Regression
print("\n1️⃣  Training Logistic Regression...")
log_model = LogisticRegression(random_state=42, max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
y_pred_proba_log = log_model.predict_proba(X_test)[:, 1]
print("   ✓ Complete!")

# Model 2: K-Nearest Neighbors
print("\n2️⃣  Training K-Nearest Neighbors (K=5)...")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
y_pred_proba_knn = knn_model.predict_proba(X_test)[:, 1]
print("   ✓ Complete!")

# ==================================================================
# STEP 3: EVALUATE MODELS
# ==================================================================

print("\n" + "="*70)
print("STEP 3: MODEL EVALUATION")
print("="*70)

def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    """Calculate all metrics"""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    print(f"\n📊 {model_name} RESULTS:")
    print(f"   Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   AUC:       {auc:.4f}")
    
    return {'Model': model_name, 'Accuracy': acc, 'Precision': prec, 
            'Recall': rec, 'F1': f1, 'AUC': auc}

# Evaluate both models
results_log = evaluate_model(y_test, y_pred_log, y_pred_proba_log, "Logistic Regression")
results_knn = evaluate_model(y_test, y_pred_knn, y_pred_proba_knn, "KNN (K=5)")

# Comparison table
comparison_df = pd.DataFrame([results_log, results_knn])
print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)
print("\n" + comparison_df.to_string(index=False))

# Best model
best_model_name = comparison_df.loc[comparison_df['AUC'].idxmax(), 'Model']
print(f"\n🏆 BEST MODEL: {best_model_name}")

# ==================================================================
# STEP 4: CONFUSION MATRICES
# ==================================================================

print("\n" + "="*70)
print("STEP 4: CONFUSION MATRICES")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Logistic Regression
cm_log = confusion_matrix(y_test, y_pred_log)
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Died', 'Survived'], yticklabels=['Died', 'Survived'],
            cbar=False, annot_kws={'size': 14})
axes[0].set_title('Logistic Regression', fontweight='bold', fontsize=13)
axes[0].set_xlabel('Predicted', fontweight='bold')
axes[0].set_ylabel('Actual', fontweight='bold')

# KNN
cm_knn = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['Died', 'Survived'], yticklabels=['Died', 'Survived'],
            cbar=False, annot_kws={'size': 14})
axes[1].set_title('K-Nearest Neighbors', fontweight='bold', fontsize=13)
axes[1].set_xlabel('Predicted', fontweight='bold')
axes[1].set_ylabel('Actual', fontweight='bold')

plt.suptitle('Titanic Survival - Confusion Matrices', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('titanic_01_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✓ Saved titanic_01_confusion_matrices.png")

# ==================================================================
# STEP 5: ROC CURVES
# ==================================================================

print("\n" + "="*70)
print("STEP 5: ROC CURVES")
print("="*70)

# Calculate ROC curves
fpr_log, tpr_log, _ = roc_curve(y_test, y_pred_proba_log)
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_pred_proba_knn)

auc_log = roc_auc_score(y_test, y_pred_proba_log)
auc_knn = roc_auc_score(y_test, y_pred_proba_knn)

plt.figure(figsize=(10, 8))
plt.plot(fpr_log, tpr_log, linewidth=3, label=f'Logistic Regression (AUC={auc_log:.3f})')
plt.plot(fpr_knn, tpr_knn, linewidth=3, label=f'KNN (AUC={auc_knn:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC=0.5)')
plt.xlabel('False Positive Rate', fontweight='bold', fontsize=12)
plt.ylabel('True Positive Rate', fontweight='bold', fontsize=12)
plt.title('Titanic Survival - ROC Curves', fontweight='bold', fontsize=14)
plt.legend(fontsize=12, loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('titanic_02_roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✓ Saved titanic_02_roc_curves.png")

# ==================================================================
# STEP 6: MAKE PREDICTIONS
# ==================================================================

print("\n" + "="*70)
print("STEP 6: PREDICT NEW PASSENGERS")
print("="*70)

# Example new passengers
new_passengers = pd.DataFrame({
    'Pclass': [1, 3, 2],
    'Sex_Binary': [0, 1, 0],  # 0=female, 1=male
    'Age': [25, 30, 35],
    'Fare': [100, 10, 30],
    'Family_Size': [2, 1, 3]
})

# Add other required features with zeros (simplified)
for col in X_train.columns:
    if col not in new_passengers.columns:
        new_passengers[col] = 0

new_passengers = new_passengers[X_train.columns]  # Same order

# Use best model (Logistic Regression typically)
predictions = log_model.predict(new_passengers[:3])
probabilities = log_model.predict_proba(new_passengers[:3])[:, 1]

print("\n New Passengers:")
result_df = pd.DataFrame({
    'Class': [1, 3, 2],
    'Sex': ['Female', 'Male', 'Female'],
    'Age': [25, 30, 35],
    'Prediction': ['Survived' if p==1 else 'Died' for p in predictions],
    'Probability': [f'{prob:.1%}' for prob in probabilities]
})
print(result_df.to_string(index=False))

# ==================================================================
# STEP 7: SAVE MODEL
# ==================================================================

print("\n" + "="*70)
print("STEP 7: SAVE BEST MODEL")
print("="*70)

with open('titanic_model.pkl', 'wb') as f:
    pickle.dump(log_model, f)

print("\n✓ Saved titanic_model.pkl")

# ==================================================================
# SUMMARY
# ==================================================================

print("\n" + "="*70)
print("✅ TITANIC SURVIVAL PREDICTION COMPLETE!")
print("="*70)

