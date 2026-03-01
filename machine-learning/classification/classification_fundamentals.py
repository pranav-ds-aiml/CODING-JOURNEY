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

print("="*70)
print("CLASSIFICATION - PREDICTING CATEGORIES")
print("="*70)

# ==================================================================
# GENERATE SAMPLE DATA
# ==================================================================

print("\n" + "="*70)
print("CREATING SAMPLE DATASET")
print("="*70)

np.random.seed(42)
n_samples = 200

# Create two classes with some overlap
# Class 0: Lower scores
X_class0 = np.random.randn(n_samples//2, 2) * 0.5 + np.array([1, 1])
y_class0 = np.zeros(n_samples//2)

# Class 1: Higher scores
X_class1 = np.random.randn(n_samples//2, 2) * 0.5 + np.array([3, 3])
y_class1 = np.ones(n_samples//2)

# Combine
X = np.vstack([X_class0, X_class1])
y = np.concatenate([y_class0, y_class1])

# Create DataFrame
df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
df['Class'] = y.astype(int)

print(f"\n✓ Created {n_samples} samples")
print(f"   Class 0: {(y==0).sum()} samples")
print(f"   Class 1: {(y==1).sum()} samples")

print("\nFirst 10 rows:")
print(df.head(10))

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='Class 0', 
            alpha=0.6, s=100, edgecolors='black')
plt.scatter(X[y==1, 0], X[y==1, 1], c='red', label='Class 1', 
            alpha=0.6, s=100, edgecolors='black')
plt.xlabel('Feature 1', fontweight='bold', fontsize=12)
plt.ylabel('Feature 2', fontweight='bold', fontsize=12)
plt.title('Binary Classification Data', fontweight='bold', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('classification_01_data.png', dpi=300)
plt.close()

print("\n✓ Saved classification_01_data.png")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ==================================================================
# ALGORITHM 1: LOGISTIC REGRESSION
# ==================================================================

print("\n" + "="*70)
print("ALGORITHM 1: LOGISTIC REGRESSION")
print("="*70)

# Train Logistic Regression
print("\n Training Logistic Regression...")

log_model = LogisticRegression(random_state=42)
log_model.fit(X_train, y_train)

print(" Training complete!")

# Predictions
y_pred_log = log_model.predict(X_test)
y_pred_proba_log = log_model.predict_proba(X_test)[:, 1]  # Probability of class 1

print("\n Sample predictions:")
sample_df = pd.DataFrame({
    'Actual': y_test[:10].astype(int),
    'Predicted': y_pred_log[:10].astype(int),
    'Probability': y_pred_proba_log[:10]
})
print(sample_df.to_string(index=False))

# Accuracy
accuracy_log = accuracy_score(y_test, y_pred_log)
print(f"\n Accuracy: {accuracy_log:.4f} ({accuracy_log*100:.2f}%)")

# ==================================================================
# ALGORITHM 2: K-NEAREST NEIGHBORS (KNN)
# ==================================================================

print("\n" + "="*70)
print("ALGORITHM 2: K-NEAREST NEIGHBORS (KNN)")
print("="*70)


# Train KNN
print("\n Training K-Nearest Neighbors (K=5)...")

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

print(" Training complete!")

# Predictions
y_pred_knn = knn_model.predict(X_test)
y_pred_proba_knn = knn_model.predict_proba(X_test)[:, 1]

# Accuracy
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"\n Accuracy: {accuracy_knn:.4f} ({accuracy_knn*100:.2f}%)")

# Try different K values
print("\n Testing different K values:")
k_values = [1, 3, 5, 7, 9, 11]
k_results = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    k_results.append({'K': k, 'Accuracy': acc})

k_df = pd.DataFrame(k_results)
print(k_df.to_string(index=False))

best_k = k_df.loc[k_df['Accuracy'].idxmax(), 'K']
print(f"\n Best K: {best_k}")

# ==================================================================
# VISUALIZE DECISION BOUNDARIES
# ==================================================================

print("\n" + "="*70)
print("VISUALIZING DECISION BOUNDARIES")
print("="*70)

def plot_decision_boundary(model, X, y, title):
    """Plot decision boundary for 2D data"""
    h = 0.02  # step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='Class 0', 
                edgecolors='black', s=50)
    plt.scatter(X[y==1, 0], X[y==1, 1], c='red', label='Class 1', 
                edgecolors='black', s=50)
    plt.xlabel('Feature 1', fontweight='bold')
    plt.ylabel('Feature 2', fontweight='bold')
    plt.title(title, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

plt.subplot(1, 2, 1)
plot_decision_boundary(log_model, X_test, y_test, 
                       f'Logistic Regression (Acc={accuracy_log:.3f})')

plt.subplot(1, 2, 2)
plot_decision_boundary(knn_model, X_test, y_test, 
                       f'KNN (K=5, Acc={accuracy_knn:.3f})')

plt.tight_layout()
plt.savefig('classification_02_decision_boundaries.png', dpi=300)
plt.close()

print("\n✓ Saved classification_02_decision_boundaries.png")

print("\n INTERPRETATION:")
print("   • Colored regions = predicted class")
print("   • Boundary = where model switches prediction")
print("   • Logistic Regression = Linear boundary (straight line)")
print("   • KNN = Non-linear boundary (flexible curves)")

print("\n" + "="*70)
print(" CLASSIFICATION FUNDAMENTALS COMPLETE!")
print("="*70)