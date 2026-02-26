import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("MACHINE LEARNING FUNDAMENTALS")
print("="*70)

print("\n" + "="*70)
print("SIMPLE EXAMPLE: HEIGHT vs WEIGHT")
print("="*70)

# Generate sample data
np.random.seed(42)
heights = np.array([150, 160, 165, 170, 175, 180, 185, 190])  # cm
weights = heights * 0.6 + np.random.randn(8) * 2 + 10  # kg with some noise

print("\nSample Data:")
df_sample = pd.DataFrame({'Height (cm)': heights, 'Weight (kg)': weights})
print(df_sample)

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(heights, weights, color='blue', s=100, alpha=0.6, edgecolors='black')
plt.xlabel('Height (cm)', fontsize=12, fontweight='bold')
plt.ylabel('Weight (kg)', fontsize=12, fontweight='bold')
plt.title('Height vs Weight - The Pattern', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add trend line (what ML will learn!)
z = np.polyfit(heights, weights, 1)
p = np.poly1d(z)
plt.plot(heights, p(heights), "r--", linewidth=2, label='Pattern ML will learn')
plt.legend(fontsize=11)

plt.tight_layout()
plt.savefig('ml_01_concept.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✓ Saved ml_01_concept.png")

# Visualize split
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Training data
train_heights = heights[:6]
train_weights = weights[:6]
axes[0].scatter(train_heights, train_weights, color='green', s=150, 
                alpha=0.6, edgecolors='black', linewidths=2)
axes[0].set_xlabel('Height (cm)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Weight (kg)', fontsize=12, fontweight='bold')
axes[0].set_title('Training Set (75%) - Model Learns Here', 
                  fontsize=13, fontweight='bold', color='green')
axes[0].grid(True, alpha=0.3)
axes[0].text(165, 112, '6 points\nModel LEARNS', 
             fontsize=12, ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen'))

# Test data
test_heights = heights[6:]
test_weights = weights[6:]
axes[1].scatter(test_heights, test_weights, color='red', s=150, 
                alpha=0.6, edgecolors='black', linewidths=2)
axes[1].set_xlabel('Height (cm)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Weight (kg)', fontsize=12, fontweight='bold')
axes[1].set_title('Test Set (25%) - Model Tested Here', 
                  fontsize=13, fontweight='bold', color='red')
axes[1].grid(True, alpha=0.3)
axes[1].text(187.5, 118, '2 points\nModel TESTED', 
             fontsize=12, ha='center', bbox=dict(boxstyle='round', facecolor='lightcoral'))

plt.savefig('ml_02_train_test_split.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✓ Saved ml_02_train_test_split.png")

# ==================================================================
# MODEL EVALUATION METRICS
# ==================================================================

print("\n" + "="*70)
print("HOW TO EVALUATE MODELS")
print("="*70)

print("\n FOR REGRESSION (predicting numbers):")
print("-" * 70)
print("1. Mean Absolute Error (MAE)")
print("   → Average of |actual - predicted|")
print("   → Example: If predicting price, MAE=$5000 means avg error is $5K")
print("   → Lower is better!")

print("\n2. Mean Squared Error (MSE)")
print("   → Average of (actual - predicted)²")
print("   → Penalizes large errors more")
print("   → Lower is better!")

print("\n3. Root Mean Squared Error (RMSE)")
print("   → √MSE")
print("   → In same units as target")
print("   → Lower is better!")

print("\n4. R² Score (R-squared)")
print("   → How much variance is explained (0 to 1)")
print("   → 1.0 = Perfect fit")
print("   → 0.8 = 80% variance explained (Good!)")
print("   → Higher is better!")

print("\n FOR CLASSIFICATION (predicting categories):")
print("-" * 70)
print("1. Accuracy")
print("   → % of correct predictions")
print("   → 0.95 = 95% accurate")

print("\n2. Precision")
print("   → Of predicted positives, how many were correct?")

print("\n3. Recall")
print("   → Of actual positives, how many did we catch?")

print("\n4. F1-Score")
print("   → Balance between Precision and Recall")

print("\n(We'll explore classification metrics deeply on Day 13!)")

print("\n" + "="*70)
print(" ML FUNDAMENTALS COMPLETE!")
print("="*70)

