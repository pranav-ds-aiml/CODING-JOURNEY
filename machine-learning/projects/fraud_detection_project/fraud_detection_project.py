import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import pickle

print("="*70)
print("CREDIT CARD FRAUD DETECTION PROJECT")
print("="*70)

# ==================================================================
# CREATING SIMULATED FRAUD DATA
# ==================================================================

print("\n" + "="*70)
print("STEP 1: CREATE DATASET (Simulated Credit Card Transactions)")
print("="*70)

np.random.seed(42)

n_legitimate=9900
n_fraud=100

legitimate = pd.DataFrame({
    'Amount': np.random.lognormal(4, 1.5, n_legitimate),
    'Time': np.random.randint(0, 172800, n_legitimate),  # 2 days in seconds
    'V1': np.random.randn(n_legitimate),
    'V2': np.random.randn(n_legitimate),
    'V3': np.random.randn(n_legitimate),
    'Class': 0
})

fraud = pd.DataFrame({
    'Amount': np.random.lognormal(5.5, 1, n_fraud),
    'Time': np.random.choice(
    list(range(0, 28800)) + list(range(68400, 86400)),n_fraud),
  # Night time
    'V1': np.random.randn(n_fraud) + 2,  
    'V2': np.random.randn(n_fraud) - 1.5,
    'V3': np.random.randn(n_fraud) + 1,
    'Class': 1
})

# Combine
df = pd.concat([legitimate, fraud], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

print(f"\n✓ Created {len(df)} transactions")
print(f"   Legitimate: {n_legitimate} ({n_legitimate/len(df)*100:.2f}%)")
print(f"   Fraud:      {n_fraud} ({n_fraud/len(df)*100:.2f}%)")

print("\nEXTREME IMBALANCE: 99:1 ratio!")

print("\nSample data:")
print(df.head(10))

# Save dataset
df.to_csv('credit_card_data.csv', index=False)
print("\n✓ Saved credit_card_data.csv")

# ==================================================================
# EXPLORATORY ANALYSIS
# ==================================================================

print("\n" + "="*70)
print("STEP 2: EXPLORATORY ANALYSIS")
print("="*70)

# Statistics by class
print("\nAmount Statistics by Class:")
print(df.groupby('Class')['Amount'].describe())

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Class distribution
axes[0, 0].bar(['Legitimate', 'Fraud'], [n_legitimate, n_fraud], 
               color=['green', 'red'], alpha=0.7)
axes[0, 0].set_ylabel('Count', fontweight='bold')
axes[0, 0].set_title('Class Distribution (IMBALANCED!)', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Plot 2: Amount distribution
axes[0, 1].hist(df[df['Class']==0]['Amount'], bins=50, alpha=0.7, 
                label='Legitimate', color='green')
axes[0, 1].hist(df[df['Class']==1]['Amount'], bins=50, alpha=0.7, 
                label='Fraud', color='red')
axes[0, 1].set_xlabel('Amount ($)', fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontweight='bold')
axes[0, 1].set_title('Transaction Amount Distribution', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].set_xlim(0, 1000)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Plot 3: Time of day
axes[1, 0].scatter(df[df['Class']==0]['Time'], df[df['Class']==0]['Amount'], 
                  alpha=0.1, s=1, label='Legitimate', color='green')
axes[1, 0].scatter(df[df['Class']==1]['Time'], df[df['Class']==1]['Amount'], 
                  alpha=0.8, s=20, label='Fraud', color='red')
axes[1, 0].set_xlabel('Time (seconds)', fontweight='bold')
axes[1, 0].set_ylabel('Amount ($)', fontweight='bold')
axes[1, 0].set_title('Transaction Patterns', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Feature V1 distribution
axes[1, 1].hist(df[df['Class']==0]['V1'], bins=50, alpha=0.7, 
                label='Legitimate', color='green')
axes[1, 1].hist(df[df['Class']==1]['V1'], bins=50, alpha=0.7, 
                label='Fraud', color='red')
axes[1, 1].set_xlabel('Feature V1', fontweight='bold')
axes[1, 1].set_ylabel('Frequency', fontweight='bold')
axes[1, 1].set_title('Feature V1 Distribution', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.suptitle('Credit Card Fraud - Exploratory Analysis', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('fraud_01_eda.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✓ Saved fraud_01_eda.png")

# ==================================================================
# PREPARE DATA
# ==================================================================

print("\n" + "="*70)
print("STEP 3: PREPARE DATA")
print("="*70)

X = df.drop('Class', axis=1)
y = df['Class']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split with stratification (IMPORTANT for imbalanced data!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"  Fraud: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
print(f"\nTest set: {X_test.shape[0]} samples")
print(f"  Fraud: {y_test.sum()} ({y_test.mean()*100:.2f}%)")

# ==================================================================
# BASELINE MODEL
# ==================================================================

print("\n" + "="*70)
print("STEP 4: BASELINE (Always Predict Legitimate)")
print("="*70)

# Dummy classifier - always predict majority class
y_pred_dummy = np.zeros(len(y_test))
acc_dummy = accuracy_score(y_test, y_pred_dummy)

print(f"\nBaseline (Always Predict Legitimate):")
print(f"   Accuracy: {acc_dummy:.4f} ({acc_dummy*100:.2f}%)")
print(f"\n99% accuracy but USELESS!")
print(f"   → Missed ALL fraud!")
print(f"   → This is why accuracy is misleading!")

# ==================================================================
# TRAIN MULTIPLE MODELS
# ==================================================================

print("\n" + "="*70)
print("STEP 5: TRAIN MULTIPLE MODELS")
print("="*70)

models={}
predictions={}
probabilities={}

# Model 1: Logistic Regression
print("\nLogistic Regression...")

lr_model=LogisticRegression(class_weight='balanced',random_state=42,max_iter=10)
lr_model.fit(X_train,y_train)
models['Logistic Regression']=lr_model
predictions['Logistic Regression']=lr_model.predict(X_test)
probabilities['Logistic Regression']=lr_model.predict_proba(X_test)[:, 1]
print("   ✓ Complete")

#Model 2:Decision Tree
print("\n Decision Tree...")
dt_model=DecisionTreeClassifier(max_depth=10,random_state=42,class_weight='balanced')
dt_model.fit(X_train,y_train)
models['Decision Tree']=dt_model
predictions['Decision Tree']=dt_model.predict(X_test)
probabilities['Decision Tree']=dt_model.predict_proba(X_test)[:, 1]
print("   ✓ Complete")

#Model 3:Random Forest
print("\nRandom Forest...")
rf_model=RandomForestClassifier(max_depth=15,n_estimators=100,class_weight='balanced',random_state=42,n_jobs=-1)
rf_model.fit(X_train, y_train)
models['Random Forest'] = rf_model
predictions['Random Forest'] = rf_model.predict(X_test)
probabilities['Random Forest'] = rf_model.predict_proba(X_test)[:,1]
print("   ✓ Complete")


#Model 4:Gradient Boosting
print("\n Gradient Boosting")
gb_model=GradientBoostingClassifier(n_estimators=100,max_depth=5,learning_rate=0.1,random_state=42)

gb_model.fit(X_train,y_train)
models['Gradient Boosting'] = gb_model
predictions['Gradient Boosting'] = gb_model.predict(X_test)
probabilities['Gradient Boosting'] = gb_model.predict_proba(X_test)[:, 1]
print("   ✓ Complete")

print("\n All models trained!")

# ==================================================================
# EVALUATING ALL MODELS
# ==================================================================

print("\n" + "="*70)
print("STEP 6: MODEL EVALUATION")
print("="*70)

results=[]

for name in models.keys():
    y_pred=predictions[name]
    y_proba=probabilities[name]

    acc=accuracy_score(y_test,y_pred)
    prec=precision_score(y_test,y_pred)
    rec=recall_score(y_test,y_pred)
    f1=f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'AUC': auc
    })
    print(f"\n{name}:")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {prec:.4f} (of flagged fraud, {prec*100:.1f}% actually fraud)")
    print(f"   Recall:    {rec:.4f} (caught {rec*100:.1f}% of all fraud)")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   AUC:       {auc:.4f}")

# Comparison table
results_df = pd.DataFrame(results).sort_values('Recall', ascending=False)

print("\n" + "="*70)
print("MODEL COMPARISON (Sorted by Recall - Most Important!)")
print("="*70)
print("\n" + results_df.to_string(index=False))

best_model_name = results_df.iloc[0]['Model']
best_recall = results_df.iloc[0]['Recall']

print(f"\nBEST MODEL (by Recall): {best_model_name}")
print(f"   Catches {best_recall*100:.1f}% of fraud!")

# ==================================================================
# CONFUSION MATRICES
# ==================================================================

print("\n" + "="*70)
print("STEP 7: CONFUSION MATRICES")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for idx, name in enumerate(models.keys()):
    row = idx // 2
    col = idx % 2
    
    cm = confusion_matrix(y_test, predictions[name])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[row, col],
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'],
                cbar=False, annot_kws={'size': 14})
    
    axes[row, col].set_title(f'{name}', fontweight='bold', fontsize=12)
    axes[row, col].set_xlabel('Predicted', fontweight='bold')
    axes[row, col].set_ylabel('Actual', fontweight='bold')
    
    # Add text showing key metrics
    tn, fp, fn, tp = cm.ravel()
    text = f'Caught {tp}/{tp+fn} frauds\nFalse alarms: {fp}'
    axes[row, col].text(0.98, 0.02, text, transform=axes[row, col].transAxes,
                       fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Fraud Detection - Confusion Matrices', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('fraud_02_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✓ Saved fraud_02_confusion_matrices.png")

# ==================================================================
# ROC CURVES
# ==================================================================

print("\n" + "="*70)
print("STEP 8: ROC CURVES")
print("="*70)

plt.figure(figsize=(10, 8))

for name in models.keys():
    y_proba = probabilities[name]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={auc_score:.3f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC=0.5)')
plt.xlabel('False Positive Rate', fontweight='bold', fontsize=12)
plt.ylabel('True Positive Rate (Recall)', fontweight='bold', fontsize=12)
plt.title('ROC Curves - Fraud Detection Models', fontweight='bold', fontsize=14)
plt.legend(fontsize=11, loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fraud_03_roc_curves.png', dpi=300)
plt.close()

print("\n✓ Saved fraud_03_roc_curves.png")

# ==================================================================
# FEATURE IMPORTANCE (Random Forest)
# ==================================================================

print("\n" + "="*70)
print("STEP 9: FEATURE IMPORTANCE")
print("="*70)

importances=rf_model.feature_importances_
feature_names=X.columns

importance_df=pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\nFeature Importance (Random Forest):")
print(importance_df.to_string(index=False))

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue', alpha=0.8)
plt.xlabel('Importance', fontweight='bold', fontsize=12)
plt.title('Feature Importance - Fraud Detection', fontweight='bold', fontsize=14)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('fraud_04_feature_importance.png', dpi=300)
plt.close()

print("\n✓ Saved fraud_04_feature_importance.png")

most_important = importance_df.iloc[0]['Feature']
print(f"\nMost important feature: {most_important}")

# ==================================================================
# THRESHOLD ANALYSIS
# ==================================================================

print("\n" + "="*70)
print("STEP 10: THRESHOLD OPTIMIZATION")
print("="*70)

# Trying different thresholds on best model
best_model = models[best_model_name]
y_proba_best = probabilities[best_model_name]

thresholds_to_try = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
threshold_results = []

for thresh in thresholds_to_try:
    y_pred_thresh = (y_proba_best >= thresh).astype(int)
    
    prec = precision_score(y_test, y_pred_thresh, zero_division=0)
    rec = recall_score(y_test, y_pred_thresh)
    f1 = f1_score(y_test, y_pred_thresh)
    
    threshold_results.append({
        'Threshold': thresh,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    })

thresh_df = pd.DataFrame(threshold_results)

print("\nThreshold Analysis:")
print(thresh_df.to_string(index=False))

# Find best threshold for F1
best_thresh_idx = thresh_df['F1-Score'].idxmax()
best_threshold = thresh_df.iloc[best_thresh_idx]['Threshold']
best_f1_thresh = thresh_df.iloc[best_thresh_idx]['F1-Score']

print(f"\nOptimal Threshold (by F1): {best_threshold}")
print(f"   F1-Score: {best_f1_thresh:.4f}")

# Visualize threshold impact
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Precision & Recall vs Threshold
axes[0].plot(thresh_df['Threshold'], thresh_df['Precision'], 
            marker='o', linewidth=2, label='Precision', color='blue')
axes[0].plot(thresh_df['Threshold'], thresh_df['Recall'], 
            marker='s', linewidth=2, label='Recall', color='red')
axes[0].plot(thresh_df['Threshold'], thresh_df['F1-Score'], 
            marker='^', linewidth=2, label='F1-Score', color='green')
axes[0].axvline(x=best_threshold, color='orange', linestyle='--', 
               linewidth=2, label=f'Optimal ({best_threshold})')
axes[0].set_xlabel('Threshold', fontweight='bold')
axes[0].set_ylabel('Score', fontweight='bold')
axes[0].set_title('Metrics vs Threshold', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Precision-Recall Tradeoff
axes[1].plot(thresh_df['Recall'], thresh_df['Precision'], 
            marker='o', linewidth=2, color='purple')
for i, thresh in enumerate(thresholds_to_try):
    axes[1].annotate(f'{thresh:.1f}', 
                    (thresh_df.iloc[i]['Recall'], thresh_df.iloc[i]['Precision']),
                    fontsize=9)
axes[1].set_xlabel('Recall', fontweight='bold')
axes[1].set_ylabel('Precision', fontweight='bold')
axes[1].set_title('Precision-Recall Tradeoff', fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fraud_05_threshold_analysis.png', dpi=300)
plt.close()

print("\n✓ Saved fraud_05_threshold_analysis.png")

# ==================================================================
# FINAL PREDICTIONS WITH OPTIMAL THRESHOLD
# ==================================================================

print("\n" + "="*70)
print("STEP 11: FINAL MODEL WITH OPTIMAL THRESHOLD")
print("="*70)

y_pred_final = (y_proba_best >= best_threshold).astype(int)

cm_final = confusion_matrix(y_test, y_pred_final)
tn, fp, fn, tp = cm_final.ravel()

print(f"\nFINAL MODEL: {best_model_name}")
print(f"   Threshold: {best_threshold}")
print(f"\nConfusion Matrix:")
print(cm_final)
print(f"\nBreakdown:")
print(f"   True Negatives:  {tn} (legitimate correctly identified)")
print(f"   False Positives: {fp} (legitimate flagged as fraud)")
print(f"   False Negatives: {fn} (fraud missed)")
print(f"   True Positives:  {tp} (fraud caught)")

final_prec = precision_score(y_test, y_pred_final)
final_rec = recall_score(y_test, y_pred_final)
final_f1 = f1_score(y_test, y_pred_final)

print(f"\nFINAL METRICS:")
print(f"   Precision: {final_prec:.4f} ({final_prec*100:.1f}% of flags are real fraud)")
print(f"   Recall:    {final_rec:.4f} (caught {final_rec*100:.1f}% of all fraud)")
print(f"   F1-Score:  {final_f1:.4f}")

print(f"\nBUSINESS IMPACT:")
total_fraud = y_test.sum()
fraud_caught = tp
fraud_missed = fn
legitimate_blocked = fp

print(f"   Total fraud transactions: {total_fraud}")
print(f"   Fraud caught: {fraud_caught} ({fraud_caught/total_fraud*100:.1f}%)")
print(f"   Fraud missed: {fraud_missed} ({fraud_missed/total_fraud*100:.1f}%)")
print(f"   Legitimate blocked: {legitimate_blocked} (inconvenience)")

# ==================================================================
# SAVE BEST MODEL
# ==================================================================

print("\n" + "="*70)
print("STEP 12: SAVE MODEL")
print("="*70)

# Save model and threshold
model_data = {
    'model': best_model,
    'threshold': best_threshold,
    'feature_names': list(X.columns)
}

with open('fraud_detection_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("\n✓ Saved fraud_detection_model.pkl")
print("   (Model + optimal threshold)")