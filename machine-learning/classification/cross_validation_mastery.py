import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, cross_val_score, cross_validate,
    KFold, StratifiedKFold, LeaveOneOut, ShuffleSplit
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print("="*70)
print("CROSS-VALIDATION - THE GOLD STANDARD FOR MODEL EVALUATION")
print("="*70)

# ==================================================================
# GENERATE SAMPLE DATA
# ==================================================================

print("\n" + "="*70)
print("CREATING SAMPLE DATASET")
print("="*70)

from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=200,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

print(f"✓ Created dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"  Class 0: {(y==0).sum()}")
print(f"  Class 1: {(y==1).sum()}")

# ==================================================================
# METHOD 1: SINGLE TRAIN/TEST SPLIT (Baseline)
# ==================================================================

print("\n" + "="*70)
print("METHOD 1: SINGLE TRAIN/TEST SPLIT (What We've Been Doing)")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
single_score = rf.score(X_test, y_test)

print(f"\n Single Split Result:")
print(f"   Test Accuracy: {single_score:.4f}")
print(f"\  But is this reliable? What if we got a lucky split?")

# Try multiple random splits to see variance
print("\n Let's try 10 different random splits...")

scores_multiple_splits = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=i
    )
    rf.fit(X_train, y_train)
    score = rf.score(X_test, y_test)
    scores_multiple_splits.append(score)

print(f"\n Scores from 10 different splits:")
print(f"   {[f'{s:.3f}' for s in scores_multiple_splits]}")
print(f"\n   Mean: {np.mean(scores_multiple_splits):.4f}")
print(f"   Std:  {np.std(scores_multiple_splits):.4f}")
print(f"   Min:  {np.min(scores_multiple_splits):.4f}")
print(f"   Max:  {np.max(scores_multiple_splits):.4f}")

print(f"\n Range: {np.max(scores_multiple_splits) - np.min(scores_multiple_splits):.4f}")
print(f"   → Results vary by {(np.max(scores_multiple_splits) - np.min(scores_multiple_splits))*100:.1f}%!")
print(f"   → Which score do we report? ")

# ==================================================================
# METHOD 2: K-FOLD CROSS-VALIDATION
# ==================================================================

print("\n" + "="*70)
print("METHOD 2: K-FOLD CROSS-VALIDATION (The Standard)")
print("="*70)

# Perform 5-fold cross-validation
print("\n🔄 Performing 5-Fold Cross-Validation...")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')

print(f"\n📊 5-Fold CV Scores: {[f'{s:.3f}' for s in cv_scores]}")
print(f"\n   Mean:  {cv_scores.mean():.4f} ✅ (Report this!)")
print(f"   Std:   {cv_scores.std():.4f}")
print(f"   95% CI: [{cv_scores.mean() - 1.96*cv_scores.std():.4f}, "
      f"{cv_scores.mean() + 1.96*cv_scores.std():.4f}]")

print(f"\n💡 We can confidently say:")
print(f"   Model accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ==================================================================
# METHOD 3: STRATIFIED K-FOLD (For Classification)
# ==================================================================

print("\n" + "="*70)
print("METHOD 3: STRATIFIED K-FOLD (Better for Classification)")
print("="*70)


# Compare regular vs stratified
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stratified_scores = cross_val_score(rf, X, y, cv=skf, scoring='accuracy')

print(f"\n📊 Regular K-Fold:     {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"📊 Stratified K-Fold:  {stratified_scores.mean():.4f} ± {stratified_scores.std():.4f}")

# Check class distributions in folds
print("\n🔍 Class distribution in each fold:")
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    y_fold = y[test_idx]
    print(f"   Fold {fold}: Class 0: {(y_fold==0).sum()}/{len(y_fold)} "
          f"({(y_fold==0).sum()/len(y_fold)*100:.1f}%)")

# ==================================================================
# CROSS-VALIDATE WITH MULTIPLE METRICS
# ==================================================================

print("\n" + "="*70)
print("CROSS-VALIDATION WITH MULTIPLE METRICS")
print("="*70)

print("\n🔄 Evaluating multiple metrics simultaneously...")

scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

cv_results = cross_validate(
    rf, X, y, 
    cv=5, 
    scoring=scoring,
    return_train_score=True
)

# Display results
print("\n📊 CROSS-VALIDATION RESULTS:")
print("-" * 70)

metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
    'Test_Mean': [
        cv_results['test_accuracy'].mean(),
        cv_results['test_precision'].mean(),
        cv_results['test_recall'].mean(),
        cv_results['test_f1'].mean(),
        cv_results['test_roc_auc'].mean()
    ],
    'Test_Std': [
        cv_results['test_accuracy'].std(),
        cv_results['test_precision'].std(),
        cv_results['test_recall'].std(),
        cv_results['test_f1'].std(),
        cv_results['test_roc_auc'].std()
    ],
    'Train_Mean': [
        cv_results['train_accuracy'].mean(),
        cv_results['train_precision'].mean(),
        cv_results['train_recall'].mean(),
        cv_results['train_f1'].mean(),
        cv_results['train_roc_auc'].mean()
    ]
})

print(metrics_df.to_string(index=False))

# Check for overfitting
print("\n🔍 OVERFITTING CHECK:")
print("-" * 70)
for metric in ['accuracy', 'f1', 'roc_auc']:
    train_mean = cv_results[f'train_{metric}'].mean()
    test_mean = cv_results[f'test_{metric}'].mean()
    gap = train_mean - test_mean
    
    print(f"\n{metric.upper()}:")
    print(f"   Train: {train_mean:.4f}")
    print(f"   Test:  {test_mean:.4f}")
    print(f"   Gap:   {gap:.4f}", end="")
    
    if gap > 0.1:
        print("  Possible overfitting!")
    elif gap > 0.05:
        print("   Slight overfitting")
    else:
        print(" Good generalization")

# ==================================================================
# VISUALIZE CV RESULTS
# ==================================================================

print("\n" + "="*70)
print("VISUALIZING CROSS-VALIDATION RESULTS")
print("="*70)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Scores across folds
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
fold_numbers = list(range(1, 6))

for metric in metrics_to_plot:
    axes[0, 0].plot(fold_numbers, cv_results[f'test_{metric}'], 
                   marker='o', label=metric.capitalize(), linewidth=2)

axes[0, 0].set_xlabel('Fold', fontweight='bold')
axes[0, 0].set_ylabel('Score', fontweight='bold')
axes[0, 0].set_title('Scores Across Folds', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim([0.5, 1.0])

# Plot 2: Mean scores with error bars
test_means = [cv_results[f'test_{m}'].mean() for m in metrics_to_plot]
test_stds = [cv_results[f'test_{m}'].std() for m in metrics_to_plot]

axes[0, 1].bar(range(len(metrics_to_plot)), test_means, 
              yerr=test_stds, capsize=5, alpha=0.7, color='steelblue')
axes[0, 1].set_xticks(range(len(metrics_to_plot)))
axes[0, 1].set_xticklabels([m.capitalize() for m in metrics_to_plot], rotation=45)
axes[0, 1].set_ylabel('Score', fontweight='bold')
axes[0, 1].set_title('Mean CV Scores with Std Dev', fontweight='bold')
axes[0, 1].set_ylim([0.5, 1.0])
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Plot 3: Train vs Test comparison
train_means = [cv_results[f'train_{m}'].mean() for m in metrics_to_plot]
x = np.arange(len(metrics_to_plot))
width = 0.35

axes[1, 0].bar(x - width/2, train_means, width, label='Train', alpha=0.8)
axes[1, 0].bar(x + width/2, test_means, width, label='Test', alpha=0.8)
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels([m.capitalize() for m in metrics_to_plot], rotation=45)
axes[1, 0].set_ylabel('Score', fontweight='bold')
axes[1, 0].set_title('Train vs Test Performance', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].set_ylim([0.5, 1.0])
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Score distribution (boxplot)
score_data = [cv_results[f'test_{m}'] for m in metrics_to_plot]
axes[1, 1].boxplot(score_data, labels=[m.capitalize() for m in metrics_to_plot])
axes[1, 1].set_ylabel('Score', fontweight='bold')
axes[1, 1].set_title('Score Distribution Across Folds', fontweight='bold')
axes[1, 1].set_ylim([0.5, 1.0])
axes[1, 1].grid(True, alpha=0.3, axis='y')
plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)

plt.suptitle('Cross-Validation Analysis', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('cv_01_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✓ Saved cv_01_analysis.png")

# ==================================================================
# COMPARE MODELS WITH CV
# ==================================================================

print("\n" + "="*70)
print("COMPARING MULTIPLE MODELS WITH CROSS-VALIDATION")
print("="*70)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

print("\n Cross-validating all models (5-fold)...")

comparison_results = []

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    comparison_results.append({
        'Model': name,
        'Mean_AUC': scores.mean(),
        'Std_AUC': scores.std(),
        'Min_AUC': scores.min(),
        'Max_AUC': scores.max()
    })
    print(f"   {name}: {scores.mean():.4f} ± {scores.std():.4f}")

comparison_df = pd.DataFrame(comparison_results).sort_values('Mean_AUC', ascending=False)

print("\n MODEL COMPARISON (5-Fold CV):")
print(comparison_df.to_string(index=False))

best_model = comparison_df.iloc[0]['Model']
print(f"\n🏆 Best Model: {best_model}")

# Visualize comparison
plt.figure(figsize=(10, 6))
plt.barh(comparison_df['Model'], comparison_df['Mean_AUC'], 
         xerr=comparison_df['Std_AUC'], capsize=5, alpha=0.7, color='steelblue')
plt.xlabel('AUC Score', fontweight='bold', fontsize=12)
plt.title('Model Comparison (5-Fold Cross-Validation)', fontweight='bold', fontsize=14)
plt.xlim([0.5, 1.0])
plt.grid(True, alpha=0.3, axis='x')

for i, row in comparison_df.iterrows():
    plt.text(row['Mean_AUC'] + 0.01, i, f"{row['Mean_AUC']:.3f}", 
            va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('cv_02_model_comparison.png', dpi=300)
plt.close()

print("\n✓ Saved cv_02_model_comparison.png")

# ==================================================================
# DIFFERENT CV STRATEGIES
# ==================================================================

print("\n" + "="*70)
print("DIFFERENT CROSS-VALIDATION STRATEGIES")
print("="*70)

# Demonstrate different strategies
print("\n Comparing different CV strategies...")

cv_strategies = {
    '5-Fold': KFold(n_splits=5, shuffle=True, random_state=42),
    '10-Fold': KFold(n_splits=10, shuffle=True, random_state=42),
    'Stratified 5-Fold': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    'Shuffle Split (5)': ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
}

rf = RandomForestClassifier(n_estimators=100, random_state=42)

cv_strategy_results = []
for name, cv_strategy in cv_strategies.items():
    scores = cross_val_score(rf, X, y, cv=cv_strategy, scoring='accuracy')
    cv_strategy_results.append({
        'Strategy': name,
        'Mean': scores.mean(),
        'Std': scores.std()
    })

strategy_df = pd.DataFrame(cv_strategy_results)
print("\n Different CV Strategies (Random Forest):")
print(strategy_df.to_string(index=False))