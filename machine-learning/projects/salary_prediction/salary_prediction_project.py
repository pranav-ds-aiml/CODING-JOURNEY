import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score 
import pickle


print("="*70)
print("PROJECT: SALARY PREDICTION")
print("Predict salaries based on years of experience")
print("="*70)

# ==================================================================
# STEP 1: CREATE REALISTIC SALARY DATA
# ==================================================================

print("\n" + "="*70)
print("STEP 1: GENERATE SALARY DATASET")
print("="*70)

np.random.seed(42)
n_samples = 200

# Generate experience (0-20 years)
experience = np.random.uniform(0, 20, n_samples)

salary=(30000+4000*experience+150*experience**2+np.random.normal(0,5000,n_samples))

df = pd.DataFrame({
    'Experience_Years': experience,
    'Salary': salary
})

# Add categorical features
df['Education'] = np.random.choice(['Bachelor', 'Master', 'PhD'], n_samples, p=[0.5, 0.3, 0.2])
df['Department'] = np.random.choice(['IT', 'Sales', 'HR', 'Finance'], n_samples)

print(f"\n✓ Created {n_samples} employee records")
print("\nDataset preview:")
print(df.head(10))

print("\n Statistical Summary:")
print(df[['Experience_Years', 'Salary']].describe())

# Save dataset
df.to_csv('salary_data.csv', index=False)
print("\n✓ Saved salary_data.csv")

# ==================================================================
# STEP 2: E D A
# ==================================================================

corr=df['Experience_Years'].corr(df['Salary'])
print(f"\n Correlation (Experience vs Salary): {corr:.3f}")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Scatter plot
axes[0, 0].scatter(df['Experience_Years'], df['Salary'], alpha=0.6, edgecolors='black', s=50)
axes[0, 0].set_xlabel('Years of Experience', fontweight='bold')
axes[0, 0].set_ylabel('Salary ($)', fontweight='bold')
axes[0, 0].set_title('Experience vs Salary', fontweight='bold', fontsize=13)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Distribution of experience
axes[0, 1].hist(df['Experience_Years'], bins=20, color='skyblue', edgecolor='black')
axes[0, 1].set_xlabel('Years of Experience', fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontweight='bold')
axes[0, 1].set_title('Experience Distribution', fontweight='bold', fontsize=13)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Plot 3: Distribution of salary
axes[1, 0].hist(df['Salary'], bins=20, color='lightgreen', edgecolor='black')
axes[1, 0].set_xlabel('Salary ($)', fontweight='bold')
axes[1, 0].set_ylabel('Frequency', fontweight='bold')
axes[1, 0].set_title('Salary Distribution', fontweight='bold', fontsize=13)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Salary by Education
axes[1, 1].boxplot([df[df['Education']=='Bachelor']['Salary'],
                     df[df['Education']=='Master']['Salary'],
                     df[df['Education']=='PhD']['Salary']],
                    labels=['Bachelor', 'Master', 'PhD'])
axes[1, 1].set_ylabel('Salary ($)', fontweight='bold')
axes[1, 1].set_title('Salary by Education Level', fontweight='bold', fontsize=13)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.suptitle('Salary Data - Exploratory Analysis', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('salary_01_eda.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✓ Saved salary_01_eda.png")

# ==================================================================
# STEP 3: PREPARE DATA
# ==================================================================

print("\n" + "="*70)
print("STEP 3: DATA PREPARATION")
print("="*70)

X = df[['Experience_Years']].values
y = df['Salary'].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

print(f"\nTraining set:{X_train.shape[0]} samples")
print(f"\Test Set:{X_test.shape[0]} samples")

# ==================================================================
# STEP 4: BUILD MULTIPLE MODELS
# ==================================================================

print("\n" + "="*70)
print("STEP 4: TRAINING MULTIPLE MODELS")
print("="*70)

models={}
predictions={}
scores={}

#Model 1:Linear Regression
print("\n Training Linear Regression")
model_linear=LinearRegression()
model_linear.fit(X_train,y_train)
pred_linear=model_linear.predict(X_test)
r2_linear=r2_score(y_test,pred_linear)
models['Linear']=model_linear
predictions['Linear']=pred_linear
scores['Linear']=r2_linear
print(f"   ✓ R² = {r2_linear:.4f}")

# Model 2: Polynomial (degree 2)
print("\n Training Polynomial Regresssion (degree=2)...")
poly_features=PolynomialFeatures(degree=2,include_bias=False)
X_train_poly=poly_features.fit_transform(X_train)
X_test_poly=poly_features.transform(X_test)

model_poly=LinearRegression()
model_poly.fit(X_train_poly,y_train)
pred_poly=model_poly.predict(X_test_poly)
r2_poly=r2_score(y_test,pred_poly)
models['Polynomial']=(poly_features,model_poly)
predictions['Polynomial'] = pred_poly
scores['Polynomial'] = r2_poly
print(f"   ✓ R² = {r2_poly:.4f}")

# Model 3: Ridge Regression
print("\n3️⃣  Training Ridge Regression...")
model_ridge = Ridge(alpha=10.0)
model_ridge.fit(X_train, y_train)
pred_ridge = model_ridge.predict(X_test)
r2_ridge = r2_score(y_test, pred_ridge)
models['Ridge'] = model_ridge
predictions['Ridge'] = pred_ridge
scores['Ridge'] = r2_ridge
print(f"   ✓ R² = {r2_ridge:.4f}")

# Model 4: Lasso Regression
print("\n4️⃣  Training Lasso Regression...")
model_lasso = Lasso(alpha=100.0)
model_lasso.fit(X_train, y_train)
pred_lasso = model_lasso.predict(X_test)
r2_lasso = r2_score(y_test, pred_lasso)
models['Lasso'] = model_lasso
predictions['Lasso'] = pred_lasso
scores['Lasso'] = r2_lasso
print(f"   ✓ R² = {r2_lasso:.4f}")

# ==================================================================
# STEP 5: COMPARE MODELS
# ==================================================================

print("\n" + "="*70)
print("STEP 5: MODEL COMPARISON")
print("="*70)

# Creating comparison table
print("\n" + "="*70)
print("STEP 5: MODEL COMPARISON")
print("="*70)

comparison_df = pd.DataFrame({
    'Model': list(scores.keys()),
    'R²_Score': list(scores.values()),
    'MAE': [mean_absolute_error(y_test, predictions[model]) for model in scores.keys()],
    'RMSE': [np.sqrt(mean_squared_error(y_test, predictions[model])) for model in scores.keys()]
}).sort_values('R²_Score', ascending=False)

print("\n📊 MODEL PERFORMANCE:")
print(comparison_df.to_string(index=False))

# Best model
best_model_name = comparison_df.iloc[0]['Model']
best_r2 = comparison_df.iloc[0]['R²_Score']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   R² Score: {best_r2:.4f}")
print(f"   MAE: ${comparison_df.iloc[0]['MAE']:,.2f}")
print(f"   RMSE: ${comparison_df.iloc[0]['RMSE']:,.2f}")

# ==================================================================
# STEP 6: VISUALIZE ALL MODELS
# ==================================================================

print("\n" + "="*70)
print("STEP 6: VISUALIZATION")
print("="*70)

# Create smooth curve for visualization
X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
X_plot_poly = poly_features.transform(X_plot)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Linear
axes[0, 0].scatter(X_test, y_test, alpha=0.6, s=50, label='Actual', edgecolors='black')
axes[0, 0].scatter(X_test, pred_linear, alpha=0.6, s=50, label='Predicted', edgecolors='black')
axes[0, 0].plot(X_plot, model_linear.predict(X_plot), 'r-', linewidth=3, label='Model')
axes[0, 0].set_title(f'Linear Regression (R²={r2_linear:.3f})', fontweight='bold', fontsize=12)
axes[0, 0].set_xlabel('Experience (years)')
axes[0, 0].set_ylabel('Salary ($)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Polynomial
axes[0, 1].scatter(X_test, y_test, alpha=0.6, s=50, label='Actual', edgecolors='black')
axes[0, 1].scatter(X_test, pred_poly, alpha=0.6, s=50, label='Predicted', edgecolors='black')
axes[0, 1].plot(X_plot, model_poly.predict(X_plot_poly), 'purple', linewidth=3, label='Model')
axes[0, 1].set_title(f'Polynomial Regression (R²={r2_poly:.3f})', fontweight='bold', fontsize=12)
axes[0, 1].set_xlabel('Experience (years)')
axes[0, 1].set_ylabel('Salary ($)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Ridge
axes[1, 0].scatter(X_test, y_test, alpha=0.6, s=50, label='Actual', edgecolors='black')
axes[1, 0].scatter(X_test, pred_ridge, alpha=0.6, s=50, label='Predicted', edgecolors='black')
axes[1, 0].plot(X_plot, model_ridge.predict(X_plot), 'orange', linewidth=3, label='Model')
axes[1, 0].set_title(f'Ridge Regression (R²={r2_ridge:.3f})', fontweight='bold', fontsize=12)
axes[1, 0].set_xlabel('Experience (years)')
axes[1, 0].set_ylabel('Salary ($)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Comparison bar chart
model_names = list(scores.keys())
r2_values = list(scores.values())
colors = ['green', 'purple', 'orange', 'red']

bars = axes[1, 1].bar(model_names, r2_values, color=colors, alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Model Comparison (R² Scores)', fontweight='bold', fontsize=12)
axes[1, 1].set_ylabel('R² Score')
axes[1, 1].set_ylim(0, 1)
axes[1, 1].grid(True, alpha=0.3, axis='y')

for bar, r2 in zip(bars, r2_values):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{r2:.3f}', ha='center', fontweight='bold')

plt.suptitle('Salary Prediction Models - Comparison', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('salary_02_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved salary_02_model_comparison.png")

# ==================================================================
# STEP 7: MAKE PREDICTIONS FOR NEW DATA
# ==================================================================

print("\n" + "="*70)
print("STEP 7: PREDICT SALARY FOR NEW EMPLOYEES")
print("="*70)

#New Employees
new_employees=pd.DataFrame({
    'Experience_Years':[2,5,10,15],
    'Level': ['Junior', 'Mid-Level', 'Senior', 'Lead']
})

print("\n🆕 New Employees:")
print(new_employees.to_string(index=False))

# Use best model (Polynomial)
best_model = models['Polynomial']
poly_feat, regressor = best_model

new_X = new_employees[['Experience_Years']].values
new_X_poly = poly_feat.transform(new_X)
predicted_salaries = regressor.predict(new_X_poly)

new_employees['Predicted_Salary'] = predicted_salaries

print("\n💰 SALARY PREDICTIONS:")
print(new_employees.to_string(index=False))

# ==================================================================
# STEP 8: SAVE BEST MODEL
# ==================================================================

print("\n" + "="*70)
print("STEP 8: SAVE BEST MODEL")
print("="*70)

# Save the polynomial model (best performer)
with open('salary_model.pkl', 'wb') as f:
    pickle.dump((poly_feat, regressor), f)

print("\n✓ Saved salary_model.pkl")
print("\n Model can be loaded and used for predictions anytime!")
