import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV,
    cross_val_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score

print("="*70)
print("HYPERPARAMETER TUNING - OPTIMIZE MODEL PERFORMANCE")
print("="*70)

# ==================================================================
# GENERATE DATA
# ==================================================================

print("\n" + "="*70)
print("CREATING DATASET")
print("="*70)

X, y = make_classification(
    n_samples=500,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")

# ==================================================================
# BASELINE: DEFAULT PARAMETERS
# ==================================================================

print("\n" + "="*70)
print("BASELINE: DEFAULT HYPERPARAMETERS")
print("="*70)

rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train, y_train)

baseline_score = rf_default.score(X_test, y_test)
baseline_cv_scores = cross_val_score(rf_default, X_train, y_train, cv=5)

print(f"\n BASELINE (Default Parameters):")
print(f"   Test Accuracy: {baseline_score:.4f}")
print(f"   CV Accuracy: {baseline_cv_scores.mean():.4f} ± {baseline_cv_scores.std():.4f}")

print(f"\n Default Parameters Used:")
print(f"   n_estimators: {rf_default.n_estimators}")
print(f"   max_depth: {rf_default.max_depth}")
print(f"   min_samples_split: {rf_default.min_samples_split}")
print(f"   min_samples_leaf: {rf_default.min_samples_leaf}")
print(f"   max_features: {rf_default.max_features}")

# ==================================================================
# METHOD 1: MANUAL TUNING
# ==================================================================

print("\n" + "="*70)
print("METHOD 1: MANUAL TUNING (Trial and Error)")
print("="*70)

print("\nTesting different n_estimators values...")

n_estimators_values=[50, 100, 200, 300, 500]
manual_results=[]

for n_est in n_estimators_values:
    rf = RandomForestClassifier(n_estimators=n_est, random_state=42)
    scores = cross_val_score(rf, X_train, y_train, cv=5)
    manual_results.append({
        'n_estimators': n_est,
        'CV_Score': scores.mean(),
        'CV_Std': scores.std()
    })
    print(f"   n_estimators={n_est}: {scores.mean():.4f} ± {scores.std():.4f}")

manual_df=pd.DataFrame(manual_results)
best_n_est=manual_df.loc[manual_df['CV_Score'].idxmax(), 'n_estimators']
print(f"\n Best n_estimators: {best_n_est}")

# Visualize
plt.figure(figsize=(10, 6))
plt.plot(manual_df['n_estimators'], manual_df['CV_Score'], 
        marker='o', linewidth=2, markersize=10)
plt.fill_between(manual_df['n_estimators'],
                manual_df['CV_Score'] - manual_df['CV_Std'],
                manual_df['CV_Score'] + manual_df['CV_Std'],
                alpha=0.2)
plt.xlabel('n_estimators', fontweight='bold', fontsize=12)
plt.ylabel('CV Accuracy', fontweight='bold', fontsize=12)
plt.title('Manual Tuning: n_estimators', fontweight='bold', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('tuning_01_manual.png', dpi=300)
plt.close()

print("\n✓ Saved tuning_01_manual.png")

# ==================================================================
# METHOD 2: GRID SEARCH
# ==================================================================

print("\n" + "="*70)
print("METHOD 2: GRID SEARCH (Exhaustive Search)")
print("="*70)

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

total_combinations = (len(param_grid['n_estimators']) *
                     len(param_grid['max_depth']) *
                     len(param_grid['min_samples_split']) *
                     len(param_grid['min_samples_leaf']))

print(f"\nGrid Search will try {total_combinations} combinations...")
print(f"   With 5-fold CV: {total_combinations * 5} models to train")
print(f"   This might take a few minutes...")

start_time = time.time()

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,              # Use all CPU cores
    verbose=1,              # Show progress
    return_train_score=True
)

grid_search.fit(X_train, y_train)

grid_time = time.time() - start_time

print(f"\n Grid Search complete! ({grid_time:.1f} seconds)")

# Best parameters
print(f"\n BEST PARAMETERS:")
for param, value in grid_search.best_params_.items():
    print(f"   {param}: {value}")

print(f"\nBEST CV SCORE: {grid_search.best_score_:.4f}")

# Test on held-out set
grid_test_score = grid_search.score(X_test, y_test)
print(f" TEST SCORE: {grid_test_score:.4f}")

print(f"\n IMPROVEMENT:")
print(f"   Baseline CV: {baseline_cv_scores.mean():.4f}")
print(f"   Tuned CV:    {grid_search.best_score_:.4f}")
print(f"   Gain:        {(grid_search.best_score_ - baseline_cv_scores.mean()):.4f} "
      f"({(grid_search.best_score_ - baseline_cv_scores.mean())*100:.1f}%)")

# Analyze results
results_df = pd.DataFrame(grid_search.cv_results_)

# Top 10 parameter combinations
print("\n TOP 10 PARAMETER COMBINATIONS:")
top_10 = results_df.nsmallest(10, 'rank_test_score')[
    ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
]
for idx, row in top_10.iterrows():
    print(f"\n   Rank {int(row['rank_test_score'])}: "
          f"Score={row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}")
    print(f"   {row['params']}")

# Visualize top parameters
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: n_estimators effect
axes[0, 0].boxplot([results_df[results_df['param_n_estimators']==n]['mean_test_score'] 
                    for n in param_grid['n_estimators']],
                   labels=param_grid['n_estimators'])
axes[0, 0].set_xlabel('n_estimators', fontweight='bold')
axes[0, 0].set_ylabel('CV AUC', fontweight='bold')
axes[0, 0].set_title('Effect of n_estimators', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Plot 2: max_depth effect  
depth_labels = [str(d) if d is not None else 'None' for d in param_grid['max_depth']]
axes[0, 1].boxplot([results_df[results_df['param_max_depth']==d]['mean_test_score'] 
                    for d in param_grid['max_depth']],
                   labels=depth_labels)
axes[0, 1].set_xlabel('max_depth', fontweight='bold')
axes[0, 1].set_ylabel('CV AUC', fontweight='bold')
axes[0, 1].set_title('Effect of max_depth', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Plot 3: min_samples_split effect
axes[1, 0].boxplot([results_df[results_df['param_min_samples_split']==m]['mean_test_score'] 
                    for m in param_grid['min_samples_split']],
                   labels=param_grid['min_samples_split'])
axes[1, 0].set_xlabel('min_samples_split', fontweight='bold')
axes[1, 0].set_ylabel('CV AUC', fontweight='bold')
axes[1, 0].set_title('Effect of min_samples_split', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Score distribution
axes[1, 1].hist(results_df['mean_test_score'], bins=30, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(grid_search.best_score_, color='red', linestyle='--', 
                  linewidth=2, label=f'Best: {grid_search.best_score_:.3f}')
axes[1, 1].axvline(baseline_cv_scores.mean(), color='green', linestyle='--', 
                  linewidth=2, label=f'Baseline: {baseline_cv_scores.mean():.3f}')
axes[1, 1].set_xlabel('CV AUC Score', fontweight='bold')
axes[1, 1].set_ylabel('Frequency', fontweight='bold')
axes[1, 1].set_title('Score Distribution', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.suptitle('Grid Search Analysis', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('tuning_02_grid_search.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✓ Saved tuning_02_grid_search.png")

# ==================================================================
# METHOD 3: RANDOM SEARCH
# ==================================================================

print("\n" + "="*70)
print("METHOD 3: RANDOM SEARCH (Faster Alternative)")
print("="*70)

from scipy.stats import randint, uniform

# Define parameter distributions
param_distributions = {
    'n_estimators': randint(50, 500),              # Random int between 50-500
    'max_depth': [5, 10, 15, 20, 25, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None]
}

print(f"\n Random Search (trying 50 random combinations)...")

start_time = time.time()

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    n_iter=50,              # Number of random combinations
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1,
    random_state=42,
    return_train_score=True
)

random_search.fit(X_train, y_train)

random_time=time.time() - start_time

print(f"\nRandom Search complete! ({random_time:.1f} seconds)")

# Best parameters
print(f"\n BEST PARAMETERS:")
for param, value in random_search.best_params_.items():
    print(f"   {param}: {value}")

print(f"\n BEST CV SCORE: {random_search.best_score_:.4f}")

random_test_score = random_search.score(X_test, y_test)
print(f" TEST SCORE: {random_test_score:.4f}")

# Compare methods
print("\n" + "="*70)
print("COMPARISON: GRID SEARCH vs RANDOM SEARCH")
print("="*70)

comparison = pd.DataFrame({
    'Method': ['Grid Search', 'Random Search'],
    'Best_CV_Score': [grid_search.best_score_, random_search.best_score_],
    'Test_Score': [grid_test_score, random_test_score],
    'Time_seconds': [grid_time, random_time],
    'Models_Trained': [total_combinations * 5, 50 * 5]
})

print("\n" + comparison.to_string(index=False))

print(f"\n Random Search was {grid_time/random_time:.1f}x faster!")
print(f"   Score difference: {abs(grid_search.best_score_ - random_search.best_score_):.4f}")

# ==================================================================
# LEARNING CURVES
# ==================================================================

print("\n" + "="*70)
print("LEARNING CURVES - DIAGNOSE OVERFITTING/UNDERFITTING")
print("="*70)

from sklearn.model_selection import learning_curve

print("\n Generating learning curves...")

train_sizes, train_scores, val_scores = learning_curve(
    grid_search.best_estimator_,
    X_train, y_train,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='roc_auc',
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training Score', linewidth=2, marker='o')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
plt.plot(train_sizes, val_mean, label='Validation Score', linewidth=2, marker='s')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
plt.xlabel('Training Set Size', fontweight='bold', fontsize=12)
plt.ylabel('AUC Score', fontweight='bold', fontsize=12)
plt.title('Learning Curves', fontweight='bold', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('tuning_03_learning_curves.png', dpi=300)
plt.close()

print("\n✓ Saved tuning_03_learning_curves.png")

# Diagnose
final_gap = train_mean[-1] - val_mean[-1]
print(f"\n DIAGNOSIS:")
print(f"   Final Training Score: {train_mean[-1]:.4f}")
print(f"   Final Validation Score: {val_mean[-1]:.4f}")
print(f"   Gap: {final_gap:.4f}")

if final_gap > 0.1:
    print(f"     Overfitting detected! (gap > 0.1)")
    print(f"   → Consider: More data, simpler model, regularization")
elif final_gap > 0.05:
    print(f"     Slight overfitting (gap = {final_gap:.3f})")
    print(f"   → Model okay but could be improved")
else:
    print(f"    Good generalization! (gap < 0.05)")
    print(f"   → Model is well-tuned")

# ==================================================================
# FINAL MODEL
# ==================================================================

print("\n" + "="*70)
print("FINAL TUNED MODEL")
print("="*70)

# Use best model from grid search
final_model = grid_search.best_estimator_

# Final evaluation
y_pred = final_model.predict(X_test)
y_pred_proba = final_model.predict_proba(X_test)[:, 1]

print("\nFINAL MODEL PERFORMANCE:")
print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))

final_auc=roc_auc_score(y_test, y_pred_proba)
print(f"AUC Score: {final_auc:.4f}")

# Save model
import pickle

with open('tuned_random_forest.pkl', 'wb') as f:
    pickle.dump(final_model, f)

print("\n✓ Saved tuned_random_forest.pkl")
