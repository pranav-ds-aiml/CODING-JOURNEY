import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("="*70)
print("REGRESSION ALGORITHMS - COMPLETE GUIDE")
print("="*70)

# ==================================================================
# GENERATE SAMPLE DATA
# ==================================================================

print("\n📊 Generating sample dataset...")

np.random.seed(42)
n_samples = 100

# Non-linear relationship
X = np.linspace(0, 10, n_samples).reshape(-1, 1)
y = 2 + 3*X.flatten() + 0.5*X.flatten()**2 + np.random.normal(0, 3, n_samples)

# Create DataFrame
df = pd.DataFrame({'X': X.flatten(), 'y': y})

print(f"✓ Created {n_samples} samples")
print("\nFirst 5 rows:")
print(df.head())

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.6, s=50, edgecolors='black')
plt.xlabel('X', fontsize=12, fontweight='bold')
plt.ylabel('y', fontsize=12, fontweight='bold')
plt.title('Sample Data - Non-Linear Relationship', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('regression_01_data.png', dpi=300)
plt.close()
print("\n✓ Saved regression_01_data.png")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==================================================================
# ALGORITHM 1: SIMPLE LINEAR REGRESSION
# ==================================================================

print("\n" + "="*70)
print("ALGORITHM 1: SIMPLE LINEAR REGRESSION")
print("="*70)

# Train model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predictions
y_pred_linear = linear_model.predict(X_test)

# Metrics
r2_linear = r2_score(y_test, y_pred_linear)
mae_linear = mean_absolute_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))

print(f"\n PERFORMANCE:")
print(f"R² Score: {r2_linear:.4f}")
print(f"MAE: {mae_linear:.2f}")
print(f"RMSE: {rmse_linear:.2f}")

print(f"\n LEARNED EQUATION:")
print(f"y = {linear_model.coef_[0]:.2f}*X + {linear_model.intercept_:.2f}")

# ==================================================================
# ALGORITHM 2: POLYNOMIAL REGRESSION
# ==================================================================

print("\n" + "="*70)
print("ALGORITHM 2: POLYNOMIAL REGRESSION")
print("="*70)

# Create polynomial features (degree 2)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

print(f"\n Original features: {X_train.shape[1]}")
print(f"After polynomial (degree=2): {X_train_poly.shape[1]}")
print(f"   → Added X² term")

# Train model
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Predictions
y_pred_poly = poly_model.predict(X_test_poly)

# Metrics
r2_poly = r2_score(y_test, y_pred_poly)
mae_poly = mean_absolute_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))

print(f"\n PERFORMANCE:")
print(f"R² Score: {r2_poly:.4f} (vs {r2_linear:.4f} linear)")
print(f"MAE: {mae_poly:.2f} (vs {mae_linear:.2f} linear)")
print(f"RMSE: {rmse_poly:.2f} (vs {rmse_linear:.2f} linear)")

improvement = ((r2_poly - r2_linear) / r2_linear) * 100
print(f"\n Improvement: {improvement:.1f}% better R² score!")

# ==================================================================
# ALGORITHM 3: RIDGE REGRESSION (L2 Regularization)
# ==================================================================

print("\n" + "="*70)
print("ALGORITHM 3: RIDGE REGRESSION (L2 Regularization)")
print("="*70)

# Train with different alphas
alphas = [0.1, 1.0, 10.0, 100.0]
ridge_results = []

for alpha in alphas:
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train, y_train)
    y_pred_ridge = ridge_model.predict(X_test)
    r2 = r2_score(y_test, y_pred_ridge)
    ridge_results.append({'alpha': alpha, 'r2': r2, 'coef': ridge_model.coef_[0]})

ridge_df = pd.DataFrame(ridge_results)
print(f"\n📊 RIDGE PERFORMANCE (different alphas):")
print(ridge_df.to_string(index=False))

# Best alpha
best_alpha = ridge_df.loc[ridge_df['r2'].idxmax(), 'alpha']
print(f"\n✅ Best alpha: {best_alpha}")

# Train final model
ridge_model = Ridge(alpha=best_alpha)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)
r2_ridge = r2_score(y_test, y_pred_ridge)

print(f"Final R² Score: {r2_ridge:.4f}")

# ==================================================================
# ALGORITHM 4: LASSO REGRESSION (L1 Regularization)
# ==================================================================

print("\n" + "="*70)
print("ALGORITHM 4: LASSO REGRESSION (L1 Regularization)")
print("="*70)

# Train with different alphas
lasso_results = []

for alpha in alphas:
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(X_train, y_train)
    y_pred_lasso = lasso_model.predict(X_test)
    r2 = r2_score(y_test, y_pred_lasso)
    lasso_results.append({'alpha': alpha, 'r2': r2, 'coef': lasso_model.coef_[0]})

lasso_df = pd.DataFrame(lasso_results)
print(f"\n LASSO PERFORMANCE (different alphas):")
print(lasso_df.to_string(index=False))

# Best alpha
best_alpha_lasso = lasso_df.loc[lasso_df['r2'].idxmax(), 'alpha']
print(f"\n✅ Best alpha: {best_alpha_lasso}")

# Train final model
lasso_model = Lasso(alpha=best_alpha_lasso)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)
r2_lasso = r2_score(y_test, y_pred_lasso)

print(f"Final R² Score: {r2_lasso:.4f}")

# ==================================================================
# VISUALIZE ALL MODELS
# ==================================================================

print("\n" + "="*70)
print("CREATING COMPARISON VISUALIZATIONS...")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Prepare data for smooth curves
X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

# Plot 1: Linear Regression
axes[0, 0].scatter(X_train, y_train, alpha=0.5, s=30, label='Train', color='blue')
axes[0, 0].scatter(X_test, y_test, alpha=0.5, s=30, label='Test', color='red')
axes[0, 0].plot(X_plot, linear_model.predict(X_plot), 'g-', linewidth=3, label='Linear Model')
axes[0, 0].set_title(f'Linear Regression (R²={r2_linear:.3f})', fontweight='bold', fontsize=12)
axes[0, 0].set_xlabel('X')
axes[0, 0].set_ylabel('y')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Polynomial Regression
X_plot_poly = poly_features.transform(X_plot)
axes[0, 1].scatter(X_train, y_train, alpha=0.5, s=30, label='Train', color='blue')
axes[0, 1].scatter(X_test, y_test, alpha=0.5, s=30, label='Test', color='red')
axes[0, 1].plot(X_plot, poly_model.predict(X_plot_poly), 'purple', linewidth=3, label='Polynomial Model')
axes[0, 1].set_title(f'Polynomial Regression (R²={r2_poly:.3f})', fontweight='bold', fontsize=12)
axes[0, 1].set_xlabel('X')
axes[0, 1].set_ylabel('y')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Ridge Regression
axes[1, 0].scatter(X_train, y_train, alpha=0.5, s=30, label='Train', color='blue')
axes[1, 0].scatter(X_test, y_test, alpha=0.5, s=30, label='Test', color='red')
axes[1, 0].plot(X_plot, ridge_model.predict(X_plot), 'orange', linewidth=3, label='Ridge Model')
axes[1, 0].set_title(f'Ridge Regression (R²={r2_ridge:.3f})', fontweight='bold', fontsize=12)
axes[1, 0].set_xlabel('X')
axes[1, 0].set_ylabel('y')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Model Comparison
models = ['Linear', 'Polynomial', 'Ridge', 'Lasso']
r2_scores = [r2_linear, r2_poly, r2_ridge, r2_lasso]
colors_bar = ['green', 'purple', 'orange', 'red']

bars = axes[1, 1].bar(models, r2_scores, color=colors_bar, alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Model Comparison (R² Scores)', fontweight='bold', fontsize=12)
axes[1, 1].set_ylabel('R² Score')
axes[1, 1].set_ylim(0, 1)
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, score in zip(bars, r2_scores):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

plt.suptitle('Regression Algorithms Comparison', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('regression_02_all_models.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved regression_02_all_models.png")

# ==================================================================
# SUMMARY TABLE
# ==================================================================

print("\n" + "="*70)
print("FINAL COMPARISON TABLE")
print("="*70)

summary = pd.DataFrame({
    'Model': ['Linear', 'Polynomial', 'Ridge', 'Lasso'],
    'R²_Score': [r2_linear, r2_poly, r2_ridge, r2_lasso],
    'MAE': [mae_linear, mae_poly, 
            mean_absolute_error(y_test, y_pred_ridge),
            mean_absolute_error(y_test, y_pred_lasso)],
    'Best_For': [
        'Linear relationships',
        'Non-linear patterns',
        'Many correlated features',
        'Feature selection'
    ]
}).sort_values('R²_Score', ascending=False)

print("\n" + summary.to_string(index=False))

best_model = summary.iloc[0]['Model']
best_r2 = summary.iloc[0]['R²_Score']

print(f"\n WINNER: {best_model} with R² = {best_r2:.4f}")

print("\n" + "="*70)
print("REGRESSION ALGORITHMS COMPLETE!")
print("="*70)