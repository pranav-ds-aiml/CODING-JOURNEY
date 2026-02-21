import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

print("="*70)
print("DATA VISUALIZATION PORTFOLIO")
print("Creating 10 Professional Visualizations")
print("="*70)

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

np.random.seed(42)

# ==================================================================
# CHART 1: LINE CHART - Stock Price Trends
# ==================================================================

print("\n1. Creating Stock Price Trends...")
dates = pd.date_range('2024-01-01', periods=100, freq='D')
stock_data = pd.DataFrame({
    'Date': dates,
    'AAPL': 150 + np.cumsum(np.random.randn(100) * 2),
    'GOOGL': 140 + np.cumsum(np.random.randn(100) * 2),
    'MSFT': 300 + np.cumsum(np.random.randn(100) * 3)
})

fig1, ax1 = plt.subplots(figsize=(14, 7))
ax1.plot(stock_data['Date'], stock_data['AAPL'], label='Apple', linewidth=2.5)
ax1.plot(stock_data['Date'], stock_data['GOOGL'], label='Google', linewidth=2.5)
ax1.plot(stock_data['Date'], stock_data['MSFT'], label='Microsoft', linewidth=2.5)
ax1.set_title('Tech Stock Price Trends (100 Days)', fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
ax1.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=12, loc='upper left', frameon=True, shadow=True)
ax1.grid(True, alpha=0.3, linestyle='--')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('chart_01_stock_trends.png', bbox_inches='tight')
plt.close()
print("✓ Saved chart_01_stock_trends.png")

# ==================================================================
# CHART 2: BAR CHART - Sales by Category
# ==================================================================

print("2. Creating Sales by Category...")
categories = ['Electronics', 'Clothing', 'Food', 'Books', 'Toys', 'Sports']
sales_2023 = [85000, 65000, 45000, 32000, 28000, 22000]
sales_2024 = [92000, 71000, 48000, 35000, 31000, 25000]

fig2, ax2 = plt.subplots(figsize=(12, 7))
x = np.arange(len(categories))
width = 0.35

bars1 = ax2.bar(x - width/2, sales_2023, width, label='2023', color='#3498db', alpha=0.8)
bars2 = ax2.bar(x + width/2, sales_2024, width, label='2024', color='#e74c3c', alpha=0.8)

ax2.set_title('Sales Comparison by Category (2023 vs 2024)', fontsize=16, fontweight='bold', pad=20)
ax2.set_xlabel('Category', fontsize=12, fontweight='bold')
ax2.set_ylabel('Sales (₹)', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(categories)
ax2.legend(fontsize=12, frameon=True, shadow=True)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'₹{height/1000:.0f}K',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('chart_02_sales_comparison.png', bbox_inches='tight')
plt.close()
print("✓ Saved chart_02_sales_comparison.png")

# ==================================================================
# CHART 3: PIE CHART - Market Share
# ==================================================================

print("3. Creating Market Share Distribution...")
companies = ['Apple', 'Samsung', 'Xiaomi', 'Oppo', 'Vivo', 'Others']
market_share = [28, 24, 15, 12, 10, 11]
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#c2c2f0']
explode = (0.1, 0.05, 0, 0, 0, 0)

fig3, ax3 = plt.subplots(figsize=(10, 8))
wedges, texts, autotexts = ax3.pie(market_share, labels=companies, colors=colors,
                                     autopct='%1.1f%%', startangle=90, explode=explode,
                                     shadow=True, textprops={'fontsize': 11, 'weight': 'bold'})

ax3.set_title('Smartphone Market Share 2024', fontsize=16, fontweight='bold', pad=20)

# Make percentage text more visible
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)
    autotext.set_weight('bold')

plt.savefig('chart_03_market_share.png', bbox_inches='tight')
plt.close()
print("✓ Saved chart_03_market_share.png")

# ==================================================================
# CHART 4: HISTOGRAM - Age Distribution
# ==================================================================

print("4. Creating Age Distribution...")
ages = np.concatenate([
    np.random.normal(25, 5, 200),
    np.random.normal(45, 8, 150),
    np.random.normal(65, 6, 100)
])

fig4, ax4 = plt.subplots(figsize=(12, 7))
n, bins, patches = ax4.hist(ages, bins=30, color='#2ecc71', alpha=0.7, edgecolor='black')

# Color bars by height
cm = plt.cm.get_cmap('RdYlGn_r')
for i, patch in enumerate(patches):
    patch.set_facecolor(cm(n[i]/max(n)))

ax4.set_title('Customer Age Distribution', fontsize=16, fontweight='bold', pad=20)
ax4.set_xlabel('Age', fontsize=12, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax4.axvline(ages.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {ages.mean():.1f}')
ax4.legend(fontsize=12, frameon=True, shadow=True)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('chart_04_age_distribution.png', bbox_inches='tight')
plt.close()
print("✓ Saved chart_04_age_distribution.png")

# ==================================================================
# CHART 5: SCATTER PLOT - Sales vs Advertising
# ==================================================================

print("5. Creating Sales vs Advertising Spend...")
advertising = np.random.uniform(10, 100, 50)
sales = advertising * 2.5 + np.random.normal(0, 20, 50)
profit_margin = np.random.uniform(10, 30, 50)

fig5, ax5 = plt.subplots(figsize=(12, 7))
scatter = ax5.scatter(advertising, sales, c=profit_margin, s=profit_margin*10, 
                      alpha=0.6, cmap='plasma', edgecolors='black', linewidth=0.5)

z = np.polyfit(advertising, sales, 1)
p = np.poly1d(z)
ax5.plot(advertising, p(advertising), "r--", linewidth=2, label='Trend line')

ax5.set_title('Sales vs Advertising Spend', fontsize=16, fontweight='bold', pad=20)
ax5.set_xlabel('Advertising Spend (₹ in thousands)', fontsize=12, fontweight='bold')
ax5.set_ylabel('Sales (₹ in thousands)', fontsize=12, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax5, label='Profit Margin %')
cbar.set_label('Profit Margin %', fontsize=11, fontweight='bold')
ax5.legend(fontsize=12, frameon=True, shadow=True)
ax5.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chart_05_sales_vs_advertising.png', bbox_inches='tight')
plt.close()
print("✓ Saved chart_05_sales_vs_advertising.png")

# ==================================================================
# CHART 6: HEATMAP - Correlation Matrix
# ==================================================================

print("6. Creating Correlation Heatmap...")
metrics_data = pd.DataFrame({
    'Sales': np.random.randint(50, 100, 100),
    'Marketing': np.random.randint(10, 50, 100),
    'Customer_Satisfaction': np.random.randint(60, 100, 100),
    'Product_Quality': np.random.randint(70, 100, 100),
    'Price': np.random.randint(20, 80, 100)
})

correlation = metrics_data.corr()

fig6, ax6 = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='RdYlGn', center=0, 
            linewidths=2, linecolor='white', fmt='.2f', cbar_kws={'label': 'Correlation'},
            square=True, ax=ax6)

ax6.set_title('Business Metrics Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('chart_06_correlation_heatmap.png', bbox_inches='tight')
plt.close()
print("✓ Saved chart_06_correlation_heatmap.png")

# ==================================================================
# CHART 7: BOX PLOT - Salary Distribution by Department
# ==================================================================

print("7. Creating Salary Distribution...")
departments = ['IT', 'HR', 'Sales', 'Finance', 'Marketing']
salaries_data = []

for dept in departments:
    base = {'IT': 70000, 'HR': 50000, 'Sales': 55000, 'Finance': 65000, 'Marketing': 58000}
    salaries = np.random.normal(base[dept], 15000, 50)
    salaries_data.extend([(dept, sal) for sal in salaries])

salary_df = pd.DataFrame(salaries_data, columns=['Department', 'Salary'])

fig7, ax7 = plt.subplots(figsize=(12, 7))
sns.boxplot(data=salary_df, x='Department', y='Salary', palette='Set2', ax=ax7)
sns.swarmplot(data=salary_df, x='Department', y='Salary', color='black', 
              alpha=0.3, size=3, ax=ax7)

ax7.set_title('Salary Distribution by Department', fontsize=16, fontweight='bold', pad=20)
ax7.set_xlabel('Department', fontsize=12, fontweight='bold')
ax7.set_ylabel('Salary (₹)', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('chart_07_salary_boxplot.png', bbox_inches='tight')
plt.close()
print("✓ Saved chart_07_salary_boxplot.png")

# ==================================================================
# CHART 8: AREA CHART - Revenue Growth
# ==================================================================

print("8. Creating Revenue Growth Trend...")
quarters = ['Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023', 
            'Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024']
product_a = [45, 52, 58, 65, 72, 78, 85, 92]
product_b = [30, 35, 38, 42, 48, 52, 58, 62]
product_c = [15, 18, 22, 25, 28, 32, 35, 38]

fig8, ax8 = plt.subplots(figsize=(14, 7))
ax8.fill_between(range(len(quarters)), product_a, alpha=0.7, label='Product A', color='#3498db')
ax8.fill_between(range(len(quarters)), product_b, alpha=0.7, label='Product B', color='#e74c3c')
ax8.fill_between(range(len(quarters)), product_c, alpha=0.7, label='Product C', color='#2ecc71')

ax8.set_title('Quarterly Revenue by Product Line', fontsize=16, fontweight='bold', pad=20)
ax8.set_xlabel('Quarter', fontsize=12, fontweight='bold')
ax8.set_ylabel('Revenue (₹ in Millions)', fontsize=12, fontweight='bold')
ax8.set_xticks(range(len(quarters)))
ax8.set_xticklabels(quarters, rotation=45)
ax8.legend(fontsize=12, loc='upper left', frameon=True, shadow=True)
ax8.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chart_08_revenue_area.png', bbox_inches='tight')
plt.close()
print("✓ Saved chart_08_revenue_area.png")

# ==================================================================
# CHART 9: VIOLIN PLOT - Performance Metrics
# ==================================================================

print("9. Creating Performance Distribution...")
performance_data = []
teams = ['Team A', 'Team B', 'Team C', 'Team D']

for team in teams:
    scores = np.random.beta(5, 2, 100) * 100
    performance_data.extend([(team, score) for score in scores])

perf_df = pd.DataFrame(performance_data, columns=['Team', 'Score'])

fig9, ax9 = plt.subplots(figsize=(12, 7))
sns.violinplot(data=perf_df, x='Team', y='Score', palette='muted', ax=ax9)

ax9.set_title('Team Performance Score Distribution', fontsize=16, fontweight='bold', pad=20)
ax9.set_xlabel('Team', fontsize=12, fontweight='bold')
ax9.set_ylabel('Performance Score', fontsize=12, fontweight='bold')
ax9.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('chart_09_performance_violin.png', bbox_inches='tight')
plt.close()
print("✓ Saved chart_09_performance_violin.png")

# ==================================================================
# CHART 10: COMBINED DASHBOARD
# ==================================================================

print("10. Creating Executive Dashboard...")

fig10 = plt.figure(figsize=(18, 12))
gs = fig10.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# KPI Cards (text)
ax1 = fig10.add_subplot(gs[0, 0])
ax1.text(0.5, 0.7, '₹12.5M', ha='center', va='center', fontsize=36, fontweight='bold', color='#2ecc71')
ax1.text(0.5, 0.3, 'Total Revenue', ha='center', va='center', fontsize=14, color='gray')
ax1.axis('off')
ax1.set_facecolor('#f0f0f0')

ax2 = fig10.add_subplot(gs[0, 1])
ax2.text(0.5, 0.7, '+23%', ha='center', va='center', fontsize=36, fontweight='bold', color='#3498db')
ax2.text(0.5, 0.3, 'Growth Rate', ha='center', va='center', fontsize=14, color='gray')
ax2.axis('off')
ax2.set_facecolor('#f0f0f0')

ax3 = fig10.add_subplot(gs[0, 2])
ax3.text(0.5, 0.7, '1,245', ha='center', va='center', fontsize=36, fontweight='bold', color='#e74c3c')
ax3.text(0.5, 0.3, 'New Customers', ha='center', va='center', fontsize=14, color='gray')
ax3.axis('off')
ax3.set_facecolor('#f0f0f0')

# Monthly trend
ax4 = fig10.add_subplot(gs[1, :2])
months_dash = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
revenue_dash = [85, 92, 88, 95, 102, 108]
ax4.plot(months_dash, revenue_dash, marker='o', linewidth=3, markersize=10, color='#2ecc71')
ax4.fill_between(months_dash, revenue_dash, alpha=0.3, color='#2ecc71')
ax4.set_title('Monthly Revenue Trend', fontsize=14, fontweight='bold')
ax4.set_ylabel('Revenue (₹M)')
ax4.grid(True, alpha=0.3)

# Category breakdown
ax5 = fig10.add_subplot(gs[1, 2])
categories_dash = ['A', 'B', 'C', 'D']
values_dash = [35, 28, 22, 15]
ax5.pie(values_dash, labels=categories_dash, autopct='%1.0f%%', startangle=90,
        colors=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
ax5.set_title('Category Mix', fontsize=14, fontweight='bold')

# Regional performance
ax6 = fig10.add_subplot(gs[2, :])
regions_dash = ['North', 'South', 'East', 'West', 'Central']
sales_dash = [95, 88, 82, 78, 72]
colors_dash = ['#2ecc71' if x >= 85 else '#f39c12' if x >= 75 else '#e74c3c' for x in sales_dash]
bars = ax6.barh(regions_dash, sales_dash, color=colors_dash, alpha=0.7)
ax6.set_title('Regional Performance', fontsize=14, fontweight='bold')
ax6.set_xlabel('Sales (₹M)')
ax6.grid(True, alpha=0.3, axis='x')

for i, (bar, val) in enumerate(zip(bars, sales_dash)):
    ax6.text(val + 1, i, f'₹{val}M', va='center', fontweight='bold')

fig10.suptitle('Executive Dashboard - Q2 2024', fontsize=20, fontweight='bold', y=0.98)
plt.savefig('chart_10_executive_dashboard.png', bbox_inches='tight')
plt.close()
print("✓ Saved chart_10_executive_dashboard.png")

print("\n" + "="*70)
print(" VISUALIZATION PORTFOLIO COMPLETE!")
print("="*70)
print("\n Created 10 Professional Visualizations:")
print("1. chart_01_stock_trends.png - Multi-line chart")
print("2. chart_02_sales_comparison.png - Grouped bar chart")
print("3. chart_03_market_share.png - Pie chart")
print("4. chart_04_age_distribution.png - Histogram with colors")
print("5. chart_05_sales_vs_advertising.png - Scatter plot")
print("6. chart_06_correlation_heatmap.png - Heatmap")
print("7. chart_07_salary_boxplot.png - Box plot with swarm")
print("8. chart_08_revenue_area.png - Area chart")
print("9. chart_09_performance_violin.png - Violin plot")
print("10. chart_10_executive_dashboard.png - Combined dashboard")

print("\nPortfolio Quality:")
print("✓ Professional color schemes")
print("✓ Clear labels and titles")
print("✓ Grid lines for readability")
print("✓ Legends where appropriate")
print("✓ Value annotations")
print("✓ High-resolution output (300 DPI)")

print("\n These visualizations are PORTFOLIO-READY!")
print("Add them to your GitHub and showcase them to recruiters!")
