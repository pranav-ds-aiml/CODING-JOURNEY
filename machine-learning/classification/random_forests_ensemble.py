import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

print("="*70)
print("RANDOM FORESTS & ENSEMBLE METHODS")
print("="*70)

# ==================================================================
# WHAT ARE ENSEMBLE METHODS?
# ==================================================================

print("\n📚 ENSEMBLE METHODS - THE WISDOM OF CROWDS")
print("-" * 70)

# ==================================================================
# GENERATE DATA
# ==================================================================

print("\n" + "="*70)
print("CREATING DATASET")
print("="*70)

np.random.seed(42)
n_samples = 500

# Create non-linear data
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=n_samples, noise=0.3, random_state=42)

print(f"✓ Created {n_samples} samples (non-linear pattern)")
print(f"  Class 0: {(y==0).sum()}")
print(f"  Class 1: {(y==1).sum()}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==================================================================
# RANDOM FOREST
# ==================================================================

print("\n" + "="*70)
print("ALGORITHM 1: RANDOM FOREST")
print("="*70)

rf_model = RandomForestClassifier(
    n_estimators=100,        # 100 trees
    max_depth=10,            # Limit depth
    min_samples_split=5,     # Min samples to split
    max_features='sqrt',     # sqrt(n_features) per split
    random_state=42,
    n_jobs=-1                # Use all CPU cores
)

rf_model.fit(X_train, y_train)

print("✅ Training complete!")
print(f"   Trained {rf_model.n_estimators} trees")

# Predictions
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# Evaluate
acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)
rec_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_pred_proba_rf)

print(f"\n RANDOM FOREST PERFORMANCE:")
print(f"   Accuracy:  {acc_rf:.4f}")
print(f"   Precision: {prec_rf:.4f}")
print(f"   Recall:    {rec_rf:.4f}")
print(f"   F1-Score:  {f1_rf:.4f}")
print(f"   AUC:       {auc_rf:.4f}")

# ==================================================================
# GRADIENT BOOSTING
# ==================================================================

print("\n" + "="*70)
print("ALGORITHM 2: GRADIENT BOOSTING")
print("="*70)

print("\n Training Gradient Boosting...")

gb_model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,       # How much each tree contributes
    random_state=42
)

gb_model.fit(X_train, y_train)

print(" Training complete!")

# Predictions
y_pred_gb = gb_model.predict(X_test)
y_pred_proba_gb = gb_model.predict_proba(X_test)[:, 1]

# Evaluate
acc_gb = accuracy_score(y_test, y_pred_gb)
prec_gb = precision_score(y_test, y_pred_gb)
rec_gb = recall_score(y_test, y_pred_gb)
f1_gb = f1_score(y_test, y_pred_gb)
auc_gb = roc_auc_score(y_test, y_pred_proba_gb)

print(f"\n GRADIENT BOOSTING PERFORMANCE:")
print(f"   Accuracy:  {acc_gb:.4f}")
print(f"   Precision: {prec_gb:.4f}")
print(f"   Recall:    {rec_gb:.4f}")
print(f"   F1-Score:  {f1_gb:.4f}")
print(f"   AUC:       {auc_gb:.4f}")

# ==================================================================
# COMPARE ALL MODELS
# ==================================================================

print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)

#Training single decision tree for comaprision
dt_model=DecisionTreeClassifier(max_depth=10,random_state=42)
dt_model.fit(X_train,y_train)
y_pred_dt=dt_model.predict(X_test)
y_pred_proba_dt=dt_model.predict_proba(X_test)[:,1]

# Training logistic regression for comparison
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]

comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_dt),
        acc_rf,
        acc_gb
    ],
    'Precision': [
        precision_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_dt),
        prec_rf,
        prec_gb
    ],
    'Recall': [
        recall_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_dt),
        rec_rf,
        rec_gb
    ],
    'F1-Score': [
        f1_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_dt),
        f1_rf,
        f1_gb
    ],
    'AUC': [
        roc_auc_score(y_test, y_pred_proba_lr),
        roc_auc_score(y_test, y_pred_proba_dt),
        auc_rf,
        auc_gb
    ]
}).sort_values('AUC', ascending=False)

print("\nMODEL COMPARISON TABLE:")
print(comparison_df.to_string(index=False))

best_model=comparison_df.iloc[0]['Model']
best_auc=comparison_df.iloc[0]['AUC']

print(f"\n BEST MODEL: {best_model} (AUC = {best_auc:.4f})")


# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: All metrics comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
x = np.arange(len(metrics))
width = 0.2

for i, model_name in enumerate(comparison_df['Model']):
    values = comparison_df[comparison_df['Model']==model_name][metrics].values[0]
    axes[0].bar(x + i*width, values, width, label=model_name, alpha=0.8)

axes[0].set_xlabel('Metrics', fontweight='bold')
axes[0].set_ylabel('Score', fontweight='bold')
axes[0].set_title('Model Comparison - All Metrics', fontweight='bold')
axes[0].set_xticks(x + width * 1.5)
axes[0].set_xticklabels(metrics)
axes[0].legend(fontsize=9)
axes[0].set_ylim(0, 1.1)
axes[0].grid(True, alpha=0.3, axis='y')

# Plot 2: AUC comparison
axes[1].barh(comparison_df['Model'], comparison_df['AUC'], 
             color=['gold', 'silver', '#CD7F32', 'gray'][:len(comparison_df)])
axes[1].set_xlabel('AUC Score', fontweight='bold')
axes[1].set_title('Model Ranking by AUC', fontweight='bold')
axes[1].set_xlim(0, 1.1)
axes[1].grid(True, alpha=0.3, axis='x')

for i, (model, auc_val) in enumerate(zip(comparison_df['Model'], comparison_df['AUC'])):
    axes[1].text(auc_val + 0.02, i, f'{auc_val:.3f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('ensemble_01_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✓ Saved ensemble_01_model_comparison.png")

# ==================================================================
# ROC CURVES
# ==================================================================

print("\n" + "="*70)
print("ROC CURVE COMPARISON")
print("="*70)

fpr_lr,tpr_lr=roc_curve(y_test,y_pred_proba_lr)
fpr_dt,tpr_dt=roc_curve(y_test,y_pred_proba_dt)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_pred_proba_gb)

plt.figure(figsize=(10, 8))
plt.plot(fpr_lr, tpr_lr, linewidth=2, label=f'Logistic Regression (AUC={roc_auc_score(y_test, y_pred_proba_lr):.3f})')
plt.plot(fpr_dt, tpr_dt, linewidth=2, label=f'Decision Tree (AUC={roc_auc_score(y_test, y_pred_proba_dt):.3f})')
plt.plot(fpr_rf, tpr_rf, linewidth=2, label=f'Random Forest (AUC={auc_rf:.3f})')
plt.plot(fpr_gb, tpr_gb, linewidth=2, label=f'Gradient Boosting (AUC={auc_gb:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC=0.5)')
plt.xlabel('False Positive Rate', fontweight='bold', fontsize=12)
plt.ylabel('True Positive Rate', fontweight='bold', fontsize=12)
plt.title('ROC Curves - All Models', fontweight='bold', fontsize=14)
plt.legend(fontsize=11, loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ensemble_02_roc_curves.png', dpi=300)
plt.close()

print("\n✓ Saved ensemble_02_roc_curves.png")

# ==================================================================
# FEATURE IMPORTANCE (Random Forest)
# ==================================================================

print("\n" + "="*70)
print("FEATURE IMPORTANCE - RANDOM FOREST")
print("="*70)
importances_rf = rf_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': [f'Feature_{i}' for i in range(X.shape[1])],
    'Importance': importances_rf
}).sort_values('Importance', ascending=False)

print("\n📊 Feature Importance:")
print(importance_df.to_string(index=False))

print(f"\n💡 Most important: {importance_df.iloc[0]['Feature']}")

# ==================================================================
# CROSS-VALIDATION
# ==================================================================

print("\n" + "="*70)
print("CROSS-VALIDATION (Robust Evaluation)")
print("="*70)

cv_scores_rf = cross_val_score(rf_model, X, y, cv=5, scoring='roc_auc')
cv_scores_gb = cross_val_score(gb_model, X, y, cv=5, scoring='roc_auc')

print(f"\nRandom Forest CV Scores: {cv_scores_rf}")
print(f"   Mean: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})")

print(f"\nGradient Boosting CV Scores: {cv_scores_gb}")
print(f"   Mean: {cv_scores_gb.mean():.4f} (+/- {cv_scores_gb.std():.4f})")


