import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(42)
data = pd.DataFrame({
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'] * 3,
    'Region': ['North']*6 + ['South']*6 + ['East']*6,
    'Sales': np.random.randint(40, 100, 18),
    'Profit': np.random.randint(10, 30, 18)
})

# 1. Categorical plot with points
plt.figure(figsize=(12, 6))
sns.catplot(data=data, x='Month', y='Sales', hue='Region', 
            kind='point', height=6, aspect=1.5)
plt.title('Monthly Sales by Region', fontsize=14, fontweight='bold')
plt.show()

# 2. FacetGrid 
g = sns.FacetGrid(data, col='Region', height=5)
g.map(sns.barplot, 'Month', 'Sales', order=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])
g.set_titles(col_template='{col_name} Region', fontweight='bold')
g.set_axis_labels('Month', 'Sales')
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Sales by Region and Month', fontsize=16, fontweight='bold')
plt.show()

#  Swarm plot 
plt.figure(figsize=(12, 6))
sns.swarmplot(data=data, x='Month', y='Sales', hue='Region', size=8)
plt.title('Sales Distribution by Month', fontsize=14, fontweight='bold')
plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#  Strip plot
plt.figure(figsize=(12, 6))
sns.stripplot(data=data, x='Region', y='Profit', size=10, alpha=0.6)
plt.title('Profit Distribution by Region', fontsize=14, fontweight='bold')
plt.show()