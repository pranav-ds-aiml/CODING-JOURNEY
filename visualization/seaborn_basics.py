import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sns.set_style("whitegrid")
sns.set_palette("husl")

tips = sns.load_dataset('tips')  # Built-in dataset

print("Tips dataset:")
print(tips.head(10))
print()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='sex', 
                size='size', sizes=(50, 200), alpha=0.6)
plt.title('Total Bill vs Tip', fontsize=14, fontweight='bold')
plt.show()

plt.figure(figsize=(10, 6))
sns.regplot(data=tips, x='total_bill', y='tip', scatter_kws={'alpha':0.5})
plt.title('Total Bill vs Tip (with regression)', fontsize=14, fontweight='bold')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=tips, x='total_bill', kde=True, bins=20)
plt.title('Distribution of Total Bill', fontsize=14, fontweight='bold')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=tips, x='day', y='total_bill', hue='sex')
plt.title('Total Bill by Day and Gender', fontsize=14, fontweight='bold')
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(data=tips, x='day', y='total_bill', hue='sex', split=True)
plt.title('Total Bill Distribution by Day', fontsize=14, fontweight='bold')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=tips, x='day', y='total_bill', hue='sex', ci=95)
plt.title('Average Total Bill by Day', fontsize=14, fontweight='bold')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=tips, x='day', hue='sex')
plt.title('Number of Customers by Day', fontsize=14, fontweight='bold')
plt.show()

numeric_data = tips[['total_bill', 'tip', 'size']]
correlation = numeric_data.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
            linewidths=1, linecolor='white', fmt='.2f')
plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
plt.show()

sns.pairplot(tips, hue='sex', height=2.5)
plt.suptitle('Pair Plot of Tips Dataset', y=1.02, fontsize=16, fontweight='bold')
plt.show()

sns.jointplot(data=tips, x='total_bill', y='tip', kind='hex', height=8)
plt.suptitle('Joint Distribution', y=1.02, fontsize=14, fontweight='bold')
plt.show()