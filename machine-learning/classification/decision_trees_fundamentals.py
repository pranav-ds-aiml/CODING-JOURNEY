import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

print("="*70)
print("DECISION TREES - VISUAL & INTUITIVE ML")
print("="*70)

# ==================================================================
# GENERATE SAMPLE DATA
# ==================================================================

print("\n" + "="*70)
print("CREATING SAMPLE DATASET")
print("="*70)

np.random.seed(42)

n_samples = 300

X_class0_1 = np.random.randn(100, 2) * np.array([5, 10000]) + np.array([25, 30000])
X_class1_1 = np.random.randn(100, 2) * np.array([5, 10000]) + np.array([25, 70000])
X_class1_2 = np.random.randn(100, 2) * np.array([8, 15000]) + np.array([50, 50000])

X = np.vstack([X_class0_1, X_class1_1, X_class1_2])
y = np.array([0]*100 + [1]*100 + [1]*100)

df=pd.DataFrame(X,columns=['Age','Income'])
df['Will_Buy']=y

print(f"\n✓ Created {n_samples} samples")
print(f"   Class 0 (Won't Buy): {(y==0).sum()}")
print(f"   Class 1 (Will Buy): {(y==1).sum()}")

print("\nSample data:")
print(df.head(10))

X_train,y_train,X_test,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# ==================================================================
# BUILD DECISION TREE
# ==================================================================
print("\n" + "="*70)
print("BUILDING DECISION TREE")
print("="*70)

print("\nTraining Decision Tree...")

tree_model=DecisionTreeClassifier(max_depth=3,min_samples_split=10,min_samples_leaf=5,random_state=42)

tree_model.fit(X_train,y_train)
print(" Training complete!")

y_pred_train=tree_model.predict(X_train)
y_pred_test=tree_model.predict(X_test)
y_pred_proba=tree_model.predict_proba(X_test)[:,1]

train_acc=accuracy_score(y_train,y_pred_train)
test_acc=accuracy_score(y_test,y_pred_test)
precision=precision_score(y_test,y_pred_test)
recall=recall_score(y_test,y_pred_test)
f1=f1_score(y_test,y_pred_test)
auc=roc_auc_score(y_test,y_pred_test)

print(f"\n📊 PERFORMANCE:")
print(f"   Training Accuracy:  {train_acc:.4f}")
print(f"   Test Accuracy:      {test_acc:.4f}")
print(f"   Precision:          {precision:.4f}")
print(f"   Recall:             {recall:.4f}")
print(f"   F1-Score:           {f1:.4f}")
print(f"   AUC:                {auc:.4f}")

if train_acc - test_acc > 0.1:
    print("\n Warning: Gap between train and test accuracy!")
    print("   Tree might be overfitting. Try reducing max_depth.")
else:
    print("\nGood! Train and test scores are close.")

# ==================================================================
# VISUALIZE THE TREE
# ==================================================================

print("\n" + "="*70)
print("VISUALIZING THE TREE")
print("="*70)

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(
    tree_model,
    feature_names=['Age', 'Income'],
    class_names=['Won\'t Buy', 'Will Buy'],
    filled=True,
    rounded=True,
    fontsize=10,
    ax=ax
)
plt.title('Decision Tree Visualization', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('tree_01_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✓ Saved tree_01_visualization.png")

# ==================================================================
# FEATURE IMPORTANCE
# ==================================================================

print("\n" + "="*70)
print("FEATURE IMPORTANCE")
print("="*70)

importances=tree_model.feature_importances_
feature_importance_df=pd.DataFrame({
    'Feature': ['Age', 'Income'],
    'Importance': importances,
    'Percentage': importances * 100
}).sort_values('Importance', ascending=False)

print("\nFEATURE IMPORTANCE:")
print(feature_importance_df.to_string(index=False))

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], 
         color='steelblue', alpha=0.8)
plt.xlabel('Importance', fontweight='bold', fontsize=12)
plt.title('Feature Importance', fontweight='bold', fontsize=14)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('tree_02_feature_importance.png', dpi=300)
plt.close()

print("\n✓ Saved tree_02_feature_importance.png")

most_important = feature_importance_df.iloc[0]['Feature']
print(f"\n Most important feature: {most_important}")

# ==================================================================
# DECISION BOUNDARY
# ==================================================================

print("\n" + "="*70)
print("VISUALIZING DECISION BOUNDARY")
print("="*70)

def plot_decision_boundary(model, X, y, title):
    """Plot decision boundary for 2D data"""
    h = 2000  # step size
    x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
    y_min, y_max = X[:, 1].min() - 5000, X[:, 1].max() + 5000
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 2),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu', levels=1)
    scatter = plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', 
                         label='Won\'t Buy', edgecolors='black', s=50, alpha=0.7)
    scatter = plt.scatter(X[y==1, 0], X[y==1, 1], c='red', 
                         label='Will Buy', edgecolors='black', s=50, alpha=0.7)
    plt.xlabel('Age', fontweight='bold', fontsize=12)
    plt.ylabel('Income ($)', fontweight='bold', fontsize=12)
    plt.title(title, fontweight='bold', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

plt.figure(figsize=(12, 8))
plot_decision_boundary(tree_model, X_test, y_test, 
                      f'Decision Tree Boundary (Acc={test_acc:.3f})')
plt.tight_layout()
plt.savefig('tree_03_decision_boundary.png', dpi=300)
plt.close()

print("\n✓ Saved tree_03_decision_boundary.png")

print("\n INTERPRETATION:")
print("   • Rectangular regions = Tree creates boxes!")
print("   • Each box = One leaf node in tree")
print("   • Non-linear = Can model complex patterns")

# ==================================================================
# IMPACT OF MAX_DEPTH
# ==================================================================

print("\n" + "="*70)
print("IMPACT OF MAX_DEPTH PARAMETER")
print("="*70)

depths=[1,2,3,5,10,None]
depth_results=[]

for depth in depths:
    tree=DecisionTreeClassifier(max_depth=depth,random_state=42)
    tree.fit(X_train,y_train)

    train_score=tree.score(X_train,y_train)
    test_score=tree.score(X_test,y_test)
    gap=train_score-test_score

    depth_results.append({
        'max_depth':str(depth),
        'Train_Accuracy': train_score,
        'Test_Accuracy': test_score,
        'Gap': gap
    })

depth_df = pd.DataFrame(depth_results)
print("\n DEPTH COMPARISON:")
print(depth_df.to_string(index=False))

# Find best depth
best_idx = depth_df['Test_Accuracy'].idxmax()
best_depth = depth_df.iloc[best_idx]['max_depth']
print(f"\n Best max_depth: {best_depth}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Accuracy vs Depth
axes[0].plot(range(len(depths)), depth_df['Train_Accuracy'], 
            marker='o', linewidth=2, markersize=8, label='Train')
axes[0].plot(range(len(depths)), depth_df['Test_Accuracy'], 
            marker='s', linewidth=2, markersize=8, label='Test')
axes[0].set_xticks(range(len(depths)))
axes[0].set_xticklabels(depth_df['max_depth'])
axes[0].set_xlabel('max_depth', fontweight='bold')
axes[0].set_ylabel('Accuracy', fontweight='bold')
axes[0].set_title('Accuracy vs Tree Depth', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Overfitting Gap
axes[1].bar(range(len(depths)), depth_df['Gap'], color='coral', alpha=0.7)
axes[1].set_xticks(range(len(depths)))
axes[1].set_xticklabels(depth_df['max_depth'])
axes[1].set_xlabel('max_depth', fontweight='bold')
axes[1].set_ylabel('Train - Test Gap', fontweight='bold')
axes[1].set_title('Overfitting Gap (Lower is Better)', fontweight='bold')
axes[1].axhline(y=0.1, color='red', linestyle='--', linewidth=2, label='Acceptable Gap')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.suptitle('Impact of max_depth Parameter', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('tree_04_depth_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✓ Saved tree_04_depth_analysis.png")

print("\n OBSERVATIONS:")
print("   • Depth 1-2: Underfitting (too simple)")
print("   • Depth 3-5: Good balance")
print("   • Depth 10+: Overfitting (memorizing training data)")