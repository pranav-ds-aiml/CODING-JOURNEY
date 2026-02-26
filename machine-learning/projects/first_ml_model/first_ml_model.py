import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("="*70)
print("🎉 YOUR FIRST MACHINE LEARNING MODEL! 🎉")
print("="*70)

# ==================================================================
# PROBLEM: PREDICT HOUSE PRICES
# ==================================================================

print("\n PROBLEM DEFINITION")
print("-" * 70)
print("Task: Predict house prices based on features")
print("Type: Regression (predicting continuous values)")
print("Algorithm: Linear Regression")

# ==================================================================
# STEP 1: CREATE DATASET
# ==================================================================

print("\n" + "="*70)
print("STEP 1: CREATE/LOAD DATA")
print("="*70)

# Generate synthetic house data
np.random.seed(42)
n_samples = 100

data = {
    'Size_sqft': np.random.randint(500, 3000, n_samples),
    'Bedrooms': np.random.randint(1, 6, n_samples),
    'Age_years': np.random.randint(0, 50, n_samples),
    'Distance_to_city_km': np.random.uniform(1, 30, n_samples)
}

# Create target (Price) with realistic relationship
df = pd.DataFrame(data)
df['Price'] = (
    df['Size_sqft'] * 100 +  # Size is most important
    df['Bedrooms'] * 5000 +  # More bedrooms = higher price
    -df['Age_years'] * 500 +  # Older = cheaper
    -df['Distance_to_city_km'] * 1000 +  # Farther = cheaper
    np.random.normal(0, 10000, n_samples)  # Random noise
)

print("\n✓ Dataset created!")
print(f"Shape: {df.shape}")
print("\nFirst 10 rows:")
print(df.head(10))

print("\n Statistical Summary:")
print(df.describe())

# Save dataset
df.to_csv('house_prices.csv', index=False)
print("\n✓ Saved house_prices.csv")

# ==================================================================
# STEP 2: EXPLORE DATA (Quick EDA)
# ==================================================================

print("\n" + "="*70)
print("STEP 2: QUICK EXPLORATORY DATA ANALYSIS")
print("="*70)

# Check for missing values
print("\n Missing values:")
print(df.isnull().sum())

# Correlation with target
print("\n Correlation with Price:")
correlation = df.corr()['Price'].sort_values(ascending=False)
print(correlation)

# Visualize relationships
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].scatter(df['Size_sqft'], df['Price'], alpha=0.6, color='blue')
axes[0, 0].set_xlabel('Size (sqft)', fontweight='bold')
axes[0, 0].set_ylabel('Price ($)', fontweight='bold')
axes[0, 0].set_title('Size vs Price', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].scatter(df['Bedrooms'], df['Price'], alpha=0.6, color='green')
axes[0, 1].set_xlabel('Bedrooms', fontweight='bold')
axes[0, 1].set_ylabel('Price ($)', fontweight='bold')
axes[0, 1].set_title('Bedrooms vs Price', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].scatter(df['Age_years'], df['Price'], alpha=0.6, color='red')
axes[1, 0].set_xlabel('Age (years)', fontweight='bold')
axes[1, 0].set_ylabel('Price ($)', fontweight='bold')
axes[1, 0].set_title('Age vs Price', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].scatter(df['Distance_to_city_km'], df['Price'], alpha=0.6, color='orange')
axes[1, 1].set_xlabel('Distance to City (km)', fontweight='bold')
axes[1, 1].set_ylabel('Price ($)', fontweight='bold')
axes[1, 1].set_title('Distance vs Price', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('House Features vs Price', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('ml_03_feature_relationships.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✓ Saved ml_03_feature_relationships.png")

# ==================================================================
# STEP 3: PREPARE DATA (Features & Target)
# ==================================================================

print("\n" + "="*70)
print("STEP 3: PREPARE DATA")
print("="*70)

# Separate features (X) and target (y)
X = df[['Size_sqft', 'Bedrooms', 'Age_years', 'Distance_to_city_km']]
y = df['Price']

print(f"\n✓ Features (X) shape: {X.shape}")
print(f"✓ Target (y) shape: {y.shape}")

print("\n Features (X) - First 5 rows:")
print(X.head())

print("\n Target (y) - First 5 values:")
print(y.head())

# ==================================================================
# STEP 4: SPLIT DATA (Train/Test)
# ==================================================================

print("\n" + "="*70)
print("STEP 4: SPLIT DATA INTO TRAIN AND TEST SETS")
print("="*70)

# Split: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")

print(f"\n Split ratio:")
print(f"   Training: {X_train.shape[0]/len(X)*100:.0f}%")
print(f"   Testing: {X_test.shape[0]/len(X)*100:.0f}%")

# ==================================================================
# STEP 5: CREATE MODEL
# ==================================================================

print("\n" + "="*70)
print("STEP 5: CREATE THE MODEL")
print("="*70)

# Create Linear Regression model
model = LinearRegression()

print("\n✓ Model created: Linear Regression")
print(f"   Type: {type(model)}")
print("\n Model is currently untrained (blank slate)")

# ==================================================================
# STEP 6: TRAIN MODEL (THE MAGIC MOMENT!)
# ==================================================================

print("\n" + "="*70)
print("STEP 6: TRAIN THE MODEL 🎓")
print("="*70)

print("\n Training in progress...")

# TRAIN THE MODEL!
model.fit(X_train, y_train)

print(" TRAINING COMPLETE!")
print("\n Your model has learned the patterns!")

# Check what the model learned
print("\n Model learned these coefficients:")
print("   (How much each feature affects price)")
for feature, coef in zip(X.columns, model.coef_):
    print(f"   {feature}: ${coef:,.2f}")

print(f"\n   Intercept (base price): ${model.intercept_:,.2f}")

# ==================================================================
# STEP 7: MAKE PREDICTIONS
# ==================================================================

print("\n" + "="*70)
print("STEP 7: MAKE PREDICTIONS 🔮")
print("="*70)

# Predict on test set
y_pred = model.predict(X_test)

print(f"\n✓ Made {len(y_pred)} predictions!")

# Compare first 10 predictions vs actual
comparison = pd.DataFrame({
    'Actual_Price': y_test.values[:10],
    'Predicted_Price': y_pred[:10],
    'Difference': y_test.values[:10] - y_pred[:10]
})

print("\n First 10 predictions vs actual:")
print(comparison)

# ==================================================================
# STEP 8: EVALUATE MODEL
# ==================================================================

print("\n" + "="*70)
print("STEP 8: EVALUATE MODEL PERFORMANCE 📈")
print("="*70)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n PERFORMANCE METRICS:")
print("-" * 70)
print(f"Mean Absolute Error (MAE):  ${mae:,.2f}")
print(f"   → On average, predictions are off by ${mae:,.2f}")
print(f"\nRoot Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"   → Typical prediction error")
print(f"\nR² Score (R-squared): {r2:.4f}")
print(f"   → Model explains {r2*100:.2f}% of price variance")

if r2 > 0.9:
    print("   →  EXCELLENT MODEL!")
elif r2 > 0.7:
    print("   →  GOOD MODEL!")
elif r2 > 0.5:
    print("   →   DECENT MODEL")
else:
    print("   →  NEEDS IMPROVEMENT")

# ==================================================================
# STEP 9: VISUALIZE RESULTS
# ==================================================================

print("\n" + "="*70)
print("STEP 9: VISUALIZE PREDICTIONS")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Actual vs Predicted
axes[0].scatter(y_test, y_pred, alpha=0.6, color='blue', s=100, edgecolors='black')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=3, label='Perfect Prediction')
axes[0].set_xlabel('Actual Price ($)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Predicted Price ($)', fontsize=12, fontweight='bold')
axes[0].set_title('Actual vs Predicted Prices', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Plot 2: Residuals (errors)
residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.6, color='green', s=100, edgecolors='black')
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predicted Price ($)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
axes[1].set_title('Residual Plot', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.suptitle(f'Model Performance (R² = {r2:.3f})', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('ml_04_model_predictions.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✓ Saved ml_04_model_predictions.png")

# ==================================================================
# STEP 10: MAKE NEW PREDICTIONS
# ==================================================================

print("\n" + "="*70)
print("STEP 10: PREDICT PRICE FOR NEW HOUSE 🏠")
print("="*70)

# New house data
new_house = pd.DataFrame({
    'Size_sqft': [2000],
    'Bedrooms': [3],
    'Age_years': [5],
    'Distance_to_city_km': [10]
})

print("\n🏠 New House Features:")
print(new_house)

# Predict!
predicted_price = model.predict(new_house)[0]

print(f"\n🔮 PREDICTED PRICE: ${predicted_price:,.2f}")
print(f"\n💡 The model predicts this house is worth ${predicted_price:,.0f}")

# ==================================================================
# STEP 11: SAVE THE MODEL
# ==================================================================

print("\n" + "="*70)
print("STEP 11: SAVE THE MODEL")
print("="*70)

import pickle

# Save model to file
with open('house_price_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("\n✓ Model saved as 'house_price_model.pkl'")
print("\n📝 You can now load this model later and use it without retraining!")

# Demonstrate loading
with open('house_price_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Test loaded model
test_prediction = loaded_model.predict(new_house)[0]
print(f"\n✓ Loaded model works! Prediction: ${test_prediction:,.2f}")


