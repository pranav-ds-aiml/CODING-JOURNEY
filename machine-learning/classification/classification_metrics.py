import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

print("="*70)
print("CLASSIFICATION METRICS - BEYOND ACCURACY")
print("="*70)

# ==================================================================
# GENERATE IMBALANCED DATA (REALISTIC SCENARIO)
# ==================================================================

print("\n Creating dataset with class imbalance...")
print("   (Like real world: 95% legitimate emails, 5% spam)")

np.random.seed(42)

# Class 0 (Legitimate): 95%
n_class0 = 190
X_class0 = np.random.randn(n_class0, 2) * 0.5 + np.array([1, 1])
y_class0 = np.zeros(n_class0)

# Class 1 (Spam): 5%
n_class1 = 10
X_class1 = np.random.randn(n_class1, 2) * 0.5 + np.array([3, 3])
y_class1 = np.ones(n_class1)

X = np.vstack([X_class0, X_class1])
y = np.concatenate([y_class0, y_class1])

print(f"\n✓ Total samples: {len(y)}")
print(f"   Class 0 (Legitimate): {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
print(f"   Class 1 (Spam): {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# ==================================================================
# CONFUSION MATRIX
# ==================================================================

print("\n" + "="*70)
print("METRIC 1: CONFUSION MATRIX")
print("="*70)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n CONFUSION MATRIX:")
print(cm)
print(f"\nBreakdown:")
print(f"  True Negatives (TN):  {tn}")
print(f"  False Positives (FP): {fp}")
print(f"  False Negatives (FN): {fn}")
print(f"  True Positives (TP):  {tp}")

# Visualize
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'],
            cbar=False, annot_kws={'size': 16})
plt.xlabel('Predicted Label', fontweight='bold', fontsize=12)
plt.ylabel('True Label', fontweight='bold', fontsize=12)
plt.title('Confusion Matrix', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('classification_03_confusion_matrix.png', dpi=300)
plt.close()

print("\n✓ Saved classification_03_confusion_matrix.png")

# ==================================================================
# ACCURACY
# ==================================================================

print("\n" + "="*70)
print("METRIC 2: ACCURACY")
print("="*70)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Calculation: ({tp} + {tn}) / {len(y_test)} = {accuracy:.4f}")

# Dummy classifier (always predict majority class)
dummy_pred = np.zeros(len(y_test))  # Always predict 0
dummy_acc = accuracy_score(y_test, dummy_pred)
print(f"\n Dummy Classifier (always predict 0): {dummy_acc:.4f}")
print(f"   → This shows why accuracy alone is misleading!")

# ==================================================================
# PRECISION
# ==================================================================

precision = precision_score(y_test, y_pred)
print(f"\n🎯 Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"   Calculation: {tp} / ({tp} + {fp}) = {precision:.4f}")
print(f"\n💡 Of all predictions of Class 1, {precision*100:.1f}% were correct")

# ==================================================================
# RECALL (SENSITIVITY)
# ==================================================================

print("\n" + "="*70)
print("METRIC 4: RECALL (also called SENSITIVITY)")
print("="*70)

recall = recall_score(y_test, y_pred)
print(f"\n Recall: {recall:.4f} ({recall*100:.2f}%)")
print(f"   Calculation: {tp} / ({tp} + {fn}) = {recall:.4f}")
print(f"\n We caught {recall*100:.1f}% of all Class 1 instances")

# ==================================================================
# F1-SCORE
# ==================================================================

f1 = f1_score(y_test, y_pred)
print(f"\n🎯 F1-Score: {f1:.4f}")
print(f"   Calculation: 2 × ({precision:.4f} × {recall:.4f}) / ({precision:.4f} + {recall:.4f})")
print(f"   = {f1:.4f}")

# ==================================================================
# CLASSIFICATION REPORT
# ==================================================================

print("\n" + "="*70)
print("COMPLETE CLASSIFICATION REPORT")
print("="*70)

report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])
print("\n" + report)

# ==================================================================
# ROC CURVE & AUC
# ==================================================================

print("\n" + "="*70)
print("METRIC 6: ROC CURVE & AUC")
print("="*70)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n🎯 AUC Score: {auc:.4f}")

if auc >= 0.9:
    rating = "Excellent!"
elif auc >= 0.8:
    rating = "Very Good!"
elif auc >= 0.7:
    rating = "Good"
else:
    rating = "Needs Improvement"

print(f"   Rating: {rating}")

# Plot ROC Curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, linewidth=3, label=f'ROC Curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.5)')
plt.xlabel('False Positive Rate', fontweight='bold', fontsize=12)
plt.ylabel('True Positive Rate (Recall)', fontweight='bold', fontsize=12)
plt.title('ROC Curve', fontweight='bold', fontsize=14)
plt.legend(fontsize=12, loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('classification_04_roc_curve.png', dpi=300)
plt.close()

print("\n✓ Saved classification_04_roc_curve.png")

# ==================================================================
# SUMMARY COMPARISON
# ==================================================================

print("\n" + "="*70)
print("METRICS SUMMARY")
print("="*70)

summary_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
    'Score': [accuracy, precision, recall, f1, auc],
    'Interpretation': [
        f'{accuracy*100:.1f}% correct overall',
        f'{precision*100:.1f}% of predicted positives were correct',
        f'{recall*100:.1f}% of actual positives were caught',
        f'Balance: {f1:.3f}',
        f'Overall performance: {auc:.3f}'
    ]
})

print("\n" + summary_df.to_string(index=False))

# ==================================================================
# DECISION: WHICH METRIC TO USE?
# ==================================================================

print("\n" + "="*70)
print("WHICH METRIC SHOULD YOU USE?")
print("="*70)

print("\n" + "="*70)
print(" CLASSIFICATION METRICS COMPLETE!")
print("="*70)