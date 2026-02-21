import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Basic line plot
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(x, np.sin(x), label='sin(x)', linewidth=2)
plt.plot(x, np.cos(x), label='cos(x)', linewidth=2)
plt.plot(x, np.sin(x) + np.cos(x), label='sin(x) + cos(x)', linewidth=2)
plt.title('Trigonometric Functions', fontsize=14, fontweight='bold')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Scatter plot
x_scatter = np.random.randn(100)
y_scatter = np.random.randn(100)
colors = np.random.rand(100)
sizes = 1000 * np.random.rand(100)

plt.figure(figsize=(10, 6))
plt.scatter(x_scatter, y_scatter, c=colors, s=sizes, alpha=0.5, cmap='viridis')
plt.colorbar(label='Color value')
plt.title('Scatter Plot with Colors and Sizes', fontsize=14, fontweight='bold')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, alpha=0.3)
plt.show()

# Bar chart
categories = ['Python', 'Java', 'JavaScript', 'C++', 'Go']
values = [85, 70, 75, 60, 55]

plt.figure(figsize=(10, 6))
bars = plt.bar(categories, values, color='steelblue', alpha=0.7)
plt.title('Programming Language Popularity', fontsize=14, fontweight='bold')
plt.xlabel('Language')
plt.ylabel('Popularity Score')
plt.ylim(0, 100)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height}',
             ha='center', va='bottom')

plt.grid(True, alpha=0.3, axis='y')
plt.show()

# Horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(categories, values, color='coral', alpha=0.7)
plt.title('Programming Language Popularity', fontsize=14, fontweight='bold')
plt.xlabel('Popularity Score')
plt.ylabel('Language')
plt.xlim(0, 100)
plt.grid(True, alpha=0.3, axis='x')
plt.show()

# Histogram
data = np.random.randn(1000)

plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, color='green', alpha=0.7, edgecolor='black')
plt.title('Normal Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3, axis='y')
plt.show()

# Pie chart
sizes = [35, 25, 20, 15, 5]
labels = ['Python', 'JavaScript', 'Java', 'C++', 'Others']
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
explode = (0.1, 0, 0, 0, 0)  # explode 1st slice

plt.figure(figsize=(10, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('Language Usage Distribution', fontsize=14, fontweight='bold')
plt.axis('equal')
plt.show()