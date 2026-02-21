import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Line plot
axes[0, 0].plot(x, np.sin(x), 'b-', linewidth=2)
axes[0, 0].set_title('Sine Wave')
axes[0, 0].set_xlabel('X')
axes[0, 0].set_ylabel('sin(x)')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Scatter
axes[0, 1].scatter(np.random.randn(50), np.random.randn(50), 
                   c=np.random.rand(50), s=100, alpha=0.6)
axes[0, 1].set_title('Random Scatter')
axes[0, 1].set_xlabel('X')
axes[0, 1].set_ylabel('Y')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Bar chart
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]
axes[1, 0].bar(categories, values, color='orange', alpha=0.7)
axes[1, 0].set_title('Category Values')
axes[1, 0].set_xlabel('Category')
axes[1, 0].set_ylabel('Value')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Histogram
axes[1, 1].hist(np.random.randn(1000), bins=30, color='green', 
                alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Distribution')
axes[1, 1].set_xlabel('Value')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('subplots_demo.png', dpi=300, bbox_inches='tight')
plt.show()