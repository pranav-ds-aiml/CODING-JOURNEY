import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, StratifiedKFold)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,roc_auc_score, confusion_matrix, classification_report, roc_curve)
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print(" COMPLETE ML PIPELINE - PROFESSIONAL WORKFLOW ")
print("="*70)

# ==================================================================
# STEP 1: LOAD & EXPLORE DATA
# ==================================================================

print("\n" + "="*70)
print("STEP 1: LOAD & EXPLORE DATA")
print("="*70)

np.random.seed(42)
n_samples = 1000

data = {
    'Age': np.random.randint(22, 60, n_samples),
    'Years_At_Company': np.random.randint(0, 20, n_samples),
    'Salary': np.random.randint(40000, 120000, n_samples),
    'Projects_Completed': np.random.randint(0, 30, n_samples),
    'Satisfaction_Score': np.random.uniform(1, 5, n_samples),
    'Performance_Rating': np.random.uniform(1, 5, n_samples),
    'Work_Hours_Per_Week': np.random.randint(35, 70, n_samples),
    'Promotions': np.random.randint(0, 5, n_samples),
    'Department': np.random.choice(['IT', 'Sales', 'HR', 'Finance', 'Operations'], n_samples),
    'Remote_Work': np.random.choice([0, 1], n_samples)
}

df = pd.DataFrame(data)

# Creating target variable (Attrition) based on logical patterns
attrition_prob = (
    0.1 +  
    (df['Satisfaction_Score'] < 2.5) * 0.3 +  
    (df['Salary'] < 60000) * 0.2 +  
    (df['Work_Hours_Per_Week'] > 55) * 0.15 +  
    (df['Years_At_Company'] < 2) * 0.15 +  
    (df['Promotions'] == 0) * 0.1  
)

df['Attrition'] = (np.random.random(n_samples) < attrition_prob).astype(int)

print(f"\n✓ Dataset created: {df.shape[0]} employees, {df.shape[1]} features")
print(f"\nTarget distribution:")
print(df['Attrition'].value_counts())
print(f"\nAttrition rate: {df['Attrition'].mean()*100:.1f}%")

print("\n First 10 rows:")
print(df.head(10))

print("\n Statistical Summary:")
print(df.describe())

# Save dataset
df.to_csv('employee_attrition.csv', index=False)
print("\n✓ Saved employee_attrition.csv")

# ==================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ==================================================================

print("\n" + "="*70)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("="*70)

# Checking missing values
print("\n Missing values:")
print(df.isnull().sum())
print("✓ No missing values!")

# Analyze attrition by department
print("\n Attrition by Department:")
dept_attrition = df.groupby('Department')['Attrition'].agg(['sum', 'count', 'mean'])
dept_attrition.columns = ['Left', 'Total', 'Attrition_Rate']
dept_attrition['Attrition_Rate'] = dept_attrition['Attrition_Rate'] * 100
print(dept_attrition)

# Correlation analysis
print("\n Correlation with Attrition:")
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation = df[numeric_cols].corr()['Attrition'].sort_values(ascending=False)
print(correlation)

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Attrition distribution
axes[0, 0].bar(['Stayed', 'Left'], df['Attrition'].value_counts().values, 
               color=['green', 'red'], alpha=0.7)
axes[0, 0].set_ylabel('Count', fontweight='bold')
axes[0, 0].set_title('Attrition Distribution', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Plot 2: Satisfaction vs Attrition
axes[0, 1].boxplot([df[df['Attrition']==0]['Satisfaction_Score'],
                     df[df['Attrition']==1]['Satisfaction_Score']],
                    labels=['Stayed', 'Left'])
axes[0, 1].set_ylabel('Satisfaction Score', fontweight='bold')
axes[0, 1].set_title('Satisfaction vs Attrition', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Plot 3: Salary vs Attrition
axes[0, 2].boxplot([df[df['Attrition']==0]['Salary'],
                     df[df['Attrition']==1]['Salary']],
                    labels=['Stayed', 'Left'])
axes[0, 2].set_ylabel('Salary ($)', fontweight='bold')
axes[0, 2].set_title('Salary vs Attrition', fontweight='bold')
axes[0, 2].grid(True, alpha=0.3, axis='y')

# Plot 4: Department attrition
dept_counts = df.groupby(['Department', 'Attrition']).size().unstack(fill_value=0)
dept_counts.plot(kind='bar', stacked=True, ax=axes[1, 0], 
                color=['green', 'red'], alpha=0.7)
axes[1, 0].set_xlabel('Department', fontweight='bold')
axes[1, 0].set_ylabel('Count', fontweight='bold')
axes[1, 0].set_title('Attrition by Department', fontweight='bold')
axes[1, 0].legend(['Stayed', 'Left'])
axes[1, 0].grid(True, alpha=0.3, axis='y')
plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)

# Plot 5: Work hours vs Attrition
axes[1, 1].hist([df[df['Attrition']==0]['Work_Hours_Per_Week'],
                 df[df['Attrition']==1]['Work_Hours_Per_Week']],
                bins=20, label=['Stayed', 'Left'], color=['green', 'red'], alpha=0.6)
axes[1, 1].set_xlabel('Work Hours/Week', fontweight='bold')
axes[1, 1].set_ylabel('Frequency', fontweight='bold')
axes[1, 1].set_title('Work Hours vs Attrition', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Plot 6: Correlation heatmap (top features)
top_features = correlation.head(6).index
corr_matrix = df[top_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            ax=axes[1, 2], square=True, linewidths=1, fmt='.2f')
axes[1, 2].set_title('Feature Correlations', fontweight='bold')

plt.suptitle('Employee Attrition - Exploratory Analysis', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('pipeline_01_eda.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✓ Saved pipeline_01_eda.png")

# ==================================================================
# STEP 3: FEATURE ENGINEERING
# ==================================================================

print("\n" + "="*70)
print("STEP 3: FEATURE ENGINEERING")
print("="*70)

# Creating new features
df['Salary_Per_Year'] = df['Salary'] / (df['Years_At_Company'] + 1)
df['Projects_Per_Year'] = df['Projects_Completed'] / (df['Years_At_Company'] + 1)
df['Satisfaction_x_Performance'] = df['Satisfaction_Score'] * df['Performance_Rating']
df['Overworked'] = (df['Work_Hours_Per_Week'] > 50).astype(int)
df['Underpaid'] = (df['Salary'] < 60000).astype(int)
df['Low_Satisfaction'] = (df['Satisfaction_Score'] < 2.5).astype(int)

print("✓ Created 6 new features:")
print("  • Salary_Per_Year")
print("  • Projects_Per_Year")
print("  • Satisfaction_x_Performance")
print("  • Overworked")
print("  • Underpaid")
print("  • Low_Satisfaction")

# One-hot encoding
dept_dummies = pd.get_dummies(df['Department'], prefix='Dept')
df = pd.concat([df, dept_dummies], axis=1)

print(f"\n✓ One-hot encoded Department")
print(f"  Total features now: {df.shape[1]}")

# ==================================================================
# STEP 4: PREPARE DATA
# ==================================================================

print("\n" + "="*70)
print("STEP 4: PREPARE DATA FOR MODELING")
print("="*70)

# Selecting features for modeling
feature_cols = [
    'Age', 'Years_At_Company', 'Salary', 'Projects_Completed',
    'Satisfaction_Score', 'Performance_Rating', 'Work_Hours_Per_Week',
    'Promotions', 'Remote_Work', 'Salary_Per_Year', 'Projects_Per_Year',
    'Satisfaction_x_Performance', 'Overworked', 'Underpaid', 'Low_Satisfaction',
    'Dept_Finance', 'Dept_HR', 'Dept_IT', 'Dept_Operations', 'Dept_Sales'
]

X = df[feature_cols]
y = df['Attrition']

print(f"\n✓ Selected {len(feature_cols)} features")
print(f"✓ Target: Attrition")

# Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n Data split:")
print(f"   Training: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"   Testing:  {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.0f}%)")

# ==================================================================
# STEP 5: TRY MULTIPLE MODELS
# ==================================================================

print("\n" + "="*70)
print("STEP 5: TRAIN & COMPARE MULTIPLE MODELS")
print("="*70)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

print("\n Training and evaluating 5 models with 5-fold CV...\n")

cv_results = []
trained_models = {}

for name, model in models.items():
    print(f"   {name}...", end=" ")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    
    model.fit(X_train, y_train)
    trained_models[name] = model
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    cv_results.append({
        'Model': name,
        'CV_AUC': cv_scores.mean(),
        'CV_Std': cv_scores.std(),
        'Test_Accuracy': accuracy_score(y_test, y_pred),
        'Test_Precision': precision_score(y_test, y_pred),
        'Test_Recall': recall_score(y_test, y_pred),
        'Test_F1': f1_score(y_test, y_pred),
        'Test_AUC': roc_auc_score(y_test, y_pred_proba)
    })
    
    print(f"CV AUC: {cv_scores.mean():.4f}")

results_df = pd.DataFrame(cv_results).sort_values('CV_AUC', ascending=False)

print("\n" + "="*70)
print("MODEL COMPARISON RESULTS")
print("="*70)
print("\n" + results_df.to_string(index=False))

best_model_name = results_df.iloc[0]['Model']
print(f"\n BEST MODEL (by CV AUC): {best_model_name}")

# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: All metrics comparison
metrics_cols = ['Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1', 'Test_AUC']
results_df.set_index('Model')[metrics_cols].plot(kind='bar', ax=axes[0], rot=45)
axes[0].set_ylabel('Score', fontweight='bold')
axes[0].set_title('Model Comparison - All Metrics', fontweight='bold')
axes[0].legend(['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'], loc='lower right')
axes[0].set_ylim([0, 1])
axes[0].grid(True, alpha=0.3, axis='y')

# Plot 2: CV AUC with error bars
axes[1].barh(results_df['Model'], results_df['CV_AUC'], 
            xerr=results_df['CV_Std'], capsize=5, alpha=0.7, color='steelblue')
axes[1].set_xlabel('CV AUC Score', fontweight='bold')
axes[1].set_title('Cross-Validation AUC (with std)', fontweight='bold')
axes[1].set_xlim([0.5, 1])
axes[1].grid(True, alpha=0.3, axis='x')

for i, (model, auc) in enumerate(zip(results_df['Model'], results_df['CV_AUC'])):
    axes[1].text(auc + 0.01, i, f'{auc:.3f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('pipeline_02_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✓ Saved pipeline_02_model_comparison.png")

# ==================================================================
# STEP 6: HYPERPARAMETER TUNING
# ==================================================================

print("\n" + "="*70)
print("STEP 6: HYPERPARAMETER TUNING (Best Model)")
print("="*70)

best_model = trained_models[best_model_name]

if best_model_name== 'Random Forest':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
elif best_model_name== 'Gradient Boosting':
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
else:
    # For other models, skiping detailed tuning
    param_grid = {}

if param_grid:
    print(f"\n Tuning {best_model_name} with Grid Search...")
    
    grid_search = GridSearchCV(
        trained_models[best_model_name].__class__(random_state=42),
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n Tuning complete!")
    print(f"\n BEST PARAMETERS:")
    for param, value in grid_search.best_params_.items():
        print(f"   {param}: {value}")
    
    print(f"\n IMPROVEMENT:")
    print(f"   Before tuning: {results_df.iloc[0]['CV_AUC']:.4f}")
    print(f"   After tuning:  {grid_search.best_score_:.4f}")
    print(f"   Gain: +{(grid_search.best_score_ - results_df.iloc[0]['CV_AUC'])*100:.2f}%")
    
    final_model = grid_search.best_estimator_
else:
    print(f"\n✓ Using {best_model_name} with default parameters")
    final_model = best_model

# ==================================================================
# STEP 7: FINAL EVALUATION
# ==================================================================

print("\n" + "="*70)
print("STEP 7: FINAL MODEL EVALUATION")
print("="*70)

y_pred_final = final_model.predict(X_test)
y_pred_proba_final = final_model.predict_proba(X_test)[:, 1]

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred_final, target_names=['Stayed', 'Left']))

# Additional metrics
final_accuracy = accuracy_score(y_test, y_pred_final)
final_precision = precision_score(y_test, y_pred_final)
final_recall = recall_score(y_test, y_pred_final)
final_f1 = f1_score(y_test, y_pred_final)
final_auc = roc_auc_score(y_test, y_pred_proba_final)

print(f"\n KEY METRICS:")
print(f"   Accuracy:  {final_accuracy:.4f}")
print(f"   Precision: {final_precision:.4f}")
print(f"   Recall:    {final_recall:.4f}")
print(f"   F1-Score:  {final_f1:.4f}")
print(f"   AUC:       {final_auc:.4f}")

# Check success criteria
print(f"\n SUCCESS CRITERIA CHECK:")
print(f"   Recall > 0.75? {final_recall > 0.75} ({final_recall:.4f})")
print(f"   Precision > 0.60? {final_precision > 0.60} ({final_precision:.4f})")

# Confusion Matrix
cm=confusion_matrix(y_test, y_pred_final)
tn, fp, fn, tp=cm.ravel()

print(f"\n CONFUSION MATRIX:")
print(cm)
print(f"\n   True Negatives:  {tn} (correctly predicted stayed)")
print(f"   False Positives: {fp} (predicted left but stayed)")
print(f"   False Negatives: {fn} (predicted stayed but left) ⚠️")
print(f"   True Positives:  {tp} (correctly predicted left)")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['Stayed', 'Left'], yticklabels=['Stayed', 'Left'],
            cbar=False, annot_kws={'size': 16})
axes[0, 0].set_title('Confusion Matrix', fontweight='bold', fontsize=13)
axes[0, 0].set_ylabel('Actual', fontweight='bold')
axes[0, 0].set_xlabel('Predicted', fontweight='bold')

# Plot 2: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_final)
axes[0, 1].plot(fpr, tpr, linewidth=3, label=f'ROC (AUC = {final_auc:.3f})')
axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
axes[0, 1].set_xlabel('False Positive Rate', fontweight='bold')
axes[0, 1].set_ylabel('True Positive Rate', fontweight='bold')
axes[0, 1].set_title('ROC Curve', fontweight='bold', fontsize=13)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Feature Importance
if hasattr(final_model, 'feature_importances_'):
    importances = final_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]  # Top 10
    
    axes[1, 0].barh(range(10), importances[indices], alpha=0.7, color='steelblue')
    axes[1, 0].set_yticks(range(10))
    axes[1, 0].set_yticklabels([feature_cols[i] for i in indices], fontsize=9)
    axes[1, 0].set_xlabel('Importance', fontweight='bold')
    axes[1, 0].set_title('Top 10 Feature Importances', fontweight='bold', fontsize=13)
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    axes[1, 0].invert_yaxis()

# Plot 4: Prediction Distribution
axes[1, 1].hist([y_pred_proba_final[y_test==0], y_pred_proba_final[y_test==1]],
                bins=20, label=['Stayed', 'Left'], color=['green', 'red'], alpha=0.6)
axes[1, 1].axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
axes[1, 1].set_xlabel('Predicted Probability', fontweight='bold')
axes[1, 1].set_ylabel('Frequency', fontweight='bold')
axes[1, 1].set_title('Prediction Distribution', fontweight='bold', fontsize=13)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.suptitle('Final Model Evaluation', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('pipeline_03_final_evaluation.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✓ Saved pipeline_03_final_evaluation.png")

# ==================================================================
# STEP 8: FEATURE IMPORTANCE & INSIGHTS
# ==================================================================

print("\n" + "="*70)
print("STEP 8: BUSINESS INSIGHTS")
print("="*70)

if hasattr(final_model, 'feature_importances_'):
    
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': final_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n TOP 10 FACTORS FOR ATTRITION:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"   {idx+1}. {row['Feature']}: {row['Importance']:.4f}")
    
    importance_df.to_csv('feature_importance.csv', index=False)
    print("\n✓ Saved feature_importance.csv")
    
    print("\n BUSINESS RECOMMENDATIONS:")
    print("━" * 70)
    
    top_features = importance_df.head(5)['Feature'].tolist()
    
    if 'Satisfaction_Score' in top_features:
        print("\n1. EMPLOYEE SATISFACTION IS CRITICAL")
        print("   → Conduct regular satisfaction surveys")
        print("   → Act on feedback quickly")
        print("   → Monitor satisfaction trends")
    
    if 'Salary' in top_features or 'Underpaid' in top_features:
        print("\n2. COMPENSATION MATTERS")
        print("   → Review salary competitiveness")
        print("   → Consider market adjustments")
        print("   → Transparent promotion criteria")
    
    if 'Work_Hours_Per_Week' in top_features or 'Overworked' in top_features:
        print("\n3. WORK-LIFE BALANCE")
        print("   → Monitor overtime hours")
        print("   → Encourage reasonable work hours")
        print("   → Consider flexible arrangements")
    
    if 'Promotions' in top_features:
        print("\n4. CAREER GROWTH OPPORTUNITIES")
        print("   → Clear career paths")
        print("   → Regular performance reviews")
        print("   → Internal promotion programs")

# ==================================================================
# STEP 9: SAVE FINAL MODEL
# ==================================================================

print("\n" + "="*70)
print("STEP 9: SAVE MODEL FOR DEPLOYMENT")
print("="*70)

# Save model and metadata
model_package = {
    'model': final_model,
    'feature_names': feature_cols,
    'model_name': best_model_name,
    'performance': {
        'accuracy': final_accuracy,
        'precision': final_precision,
        'recall': final_recall,
        'f1': final_f1,
        'auc': final_auc
    },
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
    'training_samples': len(X_train)
}

with open('attrition_model_final.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print("\n✓ Saved attrition_model_final.pkl")
print("\n Model package includes:")
print("   • Trained model")
print("   • Feature names")
print("   • Performance metrics")
print("   • Training metadata")

# ==================================================================
# STEP 10: EXAMPLE PREDICTIONS
# ==================================================================

print("\n" + "="*70)
print("STEP 10: EXAMPLE PREDICTIONS")
print("="*70)

# Create example employees
examples = pd.DataFrame({
    'Age': [30, 45, 28],
    'Years_At_Company': [5, 10, 1],
    'Salary': [55000, 90000, 48000],
    'Projects_Completed': [15, 30, 5],
    'Satisfaction_Score': [2.0, 4.5, 1.5],
    'Performance_Rating': [3.5, 4.5, 3.0],
    'Work_Hours_Per_Week': [60, 45, 55],
    'Promotions': [0, 3, 0],
    'Remote_Work': [0, 1, 0],
    'Description': [
        'Junior, overworked, low satisfaction',
        'Senior, happy, balanced',
        'New hire, underpaid, unhappy'
    ]
})

# Engineer features for examples
examples['Salary_Per_Year']=examples['Salary'] / (examples['Years_At_Company'] + 1)
examples['Projects_Per_Year']=examples['Projects_Completed'] / (examples['Years_At_Company'] + 1)
examples['Satisfaction_x_Performance']=examples['Satisfaction_Score'] * examples['Performance_Rating']
examples['Overworked']=(examples['Work_Hours_Per_Week'] > 50).astype(int)
examples['Underpaid']=(examples['Salary'] < 60000).astype(int)
examples['Low_Satisfaction']=(examples['Satisfaction_Score'] < 2.5).astype(int)

# Add department dummies (assume all IT for example)
for dept in ['Dept_Finance', 'Dept_HR', 'Dept_IT', 'Dept_Operations', 'Dept_Sales']:
    examples[dept] = 1 if dept == 'Dept_IT' else 0

# Prepare for prediction
X_examples=examples[feature_cols]

# Predict
predictions=final_model.predict(X_examples)
probabilities=final_model.predict_proba(X_examples)[:, 1]

print("\n ATTRITION PREDICTIONS:")
print("━" * 70)

for i, row in examples.iterrows():
    print(f"\nEmployee {i+1}: {row['Description']}")
    print(f"   Prediction: {'WILL LEAVE' if predictions[i] == 1 else 'Will Stay'}")
    print(f"   Probability: {probabilities[i]:.1%}")
    print(f"   Risk Level: {'HIGH ' if probabilities[i] > 0.7 else 'MEDIUM ' if probabilities[i] > 0.4 else 'LOW '}")
