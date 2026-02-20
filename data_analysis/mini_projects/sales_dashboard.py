import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime,timedelta

print("="*70)
print("SALES PERFORMANCE DASHBOARD")
print("="*70)

np.random.seed(42)
end_date=datetime.now()
start_date=end_date-timedelta(days=365)
dates=pd.date_range(start=start_date,end=end_date,freq='D')

products=['Laptop','Desktop','Monitor','Keyboard','Mosue','Headphone']
regions=['North','South','East','West','Central']
salespeople=[f'Sales_Person_{i}' for i in range(1,11)]

n_transactions = 2000
data = {
    'Date': np.random.choice(dates, n_transactions),
    'Product': np.random.choice(products, n_transactions),
    'Region': np.random.choice(regions, n_transactions),
    'Salesperson': np.random.choice(salespeople, n_transactions),
    'Units_Sold': np.random.randint(1, 20, n_transactions),
}

# Price per product
price_map = {
    'Laptop': 50000,
    'Desktop': 40000,
    'Monitor': 15000,
    'Keyboard': 2000,
    'Mouse': 800,
    'Headphones': 3000
}

df = pd.DataFrame(data)
df['Unit_Price'] = df['Product'].map(price_map)
df['Total_Sales'] = df['Units_Sold'] * df['Unit_Price']
df['Cost'] = df['Total_Sales'] * 0.6  # 60% cost
df['Profit'] = df['Total_Sales'] - df['Cost']

df['Month'] = df['Date'].dt.to_period('M')
df['Quarter'] = df['Date'].dt.to_period('Q')
df['Year'] = df['Date'].dt.year

df = df.sort_values('Date')

print(f"\nGenerated {len(df)} sales transactions")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print()

print("Sample Data:")
print(df.head(10))
print()

df.to_csv('sales_raw_data.csv', index=False)
print("✓ Saved sales_raw_data.csv\n")

# ANALYSIS 1: EXECUTIVE SUMMARY
print("="*70)
print("EXECUTIVE SUMMARY")
print("="*70)

total_revenue = df['Total_Sales'].sum()
total_profit = df['Profit'].sum()
total_units = df['Units_Sold'].sum()
profit_margin = (total_profit / total_revenue) * 100
avg_transaction = df['Total_Sales'].mean()

print(f"\nTotal Revenue:      ₹{total_revenue:,.0f}")
print(f"Total Profit:       ₹{total_profit:,.0f}")
print(f"Profit Margin:      {profit_margin:.2f}%")
print(f"Total Units Sold:   {total_units:,}")
print(f"Avg Transaction:    ₹{avg_transaction:,.0f}")
print(f"Total Transactions: {len(df):,}")
print()

# ANALYSIS 2: PRODUCT PERFORMANCE
print("="*70)
print("PRODUCT PERFORMANCE ANALYSIS")
print("="*70)

product_summary = df.groupby('Product').agg({
    'Total_Sales': 'sum',
    'Profit': 'sum',
    'Units_Sold': 'sum',
    'Date': 'count'
}).rename(columns={'Date': 'Transactions'})

product_summary['Profit_Margin_%'] = (
    product_summary['Profit'] / product_summary['Total_Sales'] * 100
).round(2)

product_summary = product_summary.sort_values('Total_Sales', ascending=False)

print("\n", product_summary)
print()

print(f"Best Selling Product: {product_summary.index[0]}")
print(f"Most Profitable: {product_summary.sort_values('Profit', ascending=False).index[0]}")
print()

# ANALYSIS 3: REGIONAL PERFORMANCE
print("="*70)
print("REGIONAL PERFORMANCE ANALYSIS")
print("="*70)

regional_summary = df.groupby('Region').agg({
    'Total_Sales': 'sum',
    'Profit': 'sum',
    'Units_Sold': 'sum'
}).sort_values('Total_Sales', ascending=False)

regional_summary['% of Total Sales'] = (
    regional_summary['Total_Sales'] / total_revenue * 100
).round(2)

print("\n", regional_summary)
print()

print(f"Top Region: {regional_summary.index[0]}")
print()

# ANALYSIS 4: TIME-BASED TRENDS
print("="*70)
print("MONTHLY TRENDS ANALYSIS")
print("="*70)

monthly_sales = df.groupby('Month').agg({
    'Total_Sales': 'sum',
    'Profit': 'sum',
    'Units_Sold': 'sum'
})

monthly_sales['Sales_Growth_%'] = monthly_sales['Total_Sales'].pct_change() * 100

print("\nLast 6 Months:")
print(monthly_sales.tail(6))
print()

best_month = monthly_sales['Total_Sales'].idxmax()
worst_month = monthly_sales['Total_Sales'].idxmin()

print(f"Best Month: {best_month} (₹{monthly_sales.loc[best_month, 'Total_Sales']:,.0f})")
print(f"Worst Month: {worst_month} (₹{monthly_sales.loc[worst_month, 'Total_Sales']:,.0f})")
print()

# ANALYSIS 5: SALES TEAM PERFORMANCE
print("="*70)
print("SALES TEAM PERFORMANCE")
print("="*70)

salesperson_summary = df.groupby('Salesperson').agg({
    'Total_Sales': 'sum',
    'Profit': 'sum',
    'Date': 'count'
}).rename(columns={'Date': 'Transactions'})

salesperson_summary['Avg_Transaction_Value'] = (
    salesperson_summary['Total_Sales'] / salesperson_summary['Transactions']
).round(0)

salesperson_summary = salesperson_summary.sort_values('Total_Sales', ascending=False)

print("\nTop 5 Salespeople:")
print(salesperson_summary.head())
print()

# ANALYSIS 6: PIVOT TABLE - SALES BY REGION AND PRODUCT
print("="*70)
print("SALES BY REGION AND PRODUCT (PIVOT TABLE)")
print("="*70)

pivot_region_product = pd.pivot_table(
    df,
    values='Total_Sales',
    index='Region',
    columns='Product',
    aggfunc='sum',
    fill_value=0,
    margins=True
)

print("\n", pivot_region_product.astype(int))
print()

pivot_region_product.to_csv('sales_pivot_region_product.csv')
print("✓ Saved sales_pivot_region_product.csv\n")

# VISUALIZATION
print("="*70)
print("CREATING VISUALIZATIONS...")
print("="*70)

fig = plt.figure(figsize=(16, 12))

# Plot 1: Revenue Trend
plt.subplot(3, 3, 1)
monthly_revenue = df.groupby(df['Date'].dt.to_period('M'))['Total_Sales'].sum()
plt.plot(monthly_revenue.index.to_timestamp(), monthly_revenue.values, 
         marker='o', linewidth=2, color='green')
plt.title('Monthly Revenue Trend', fontsize=12, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Revenue (₹)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Plot 2: Product Sales
plt.subplot(3, 3, 2)
product_sales = df.groupby('Product')['Total_Sales'].sum().sort_values()
plt.barh(product_sales.index, product_sales.values, color='steelblue')
plt.title('Sales by Product', fontsize=12, fontweight='bold')
plt.xlabel('Total Sales (₹)')
plt.grid(True, alpha=0.3, axis='x')

# Plot 3: Regional Distribution
plt.subplot(3, 3, 3)
regional_sales = df.groupby('Region')['Total_Sales'].sum()
plt.pie(regional_sales.values, labels=regional_sales.index, autopct='%1.1f%%',
        startangle=90, colors=plt.cm.Set3.colors)
plt.title('Sales by Region', fontsize=12, fontweight='bold')

# Plot 4: Units Sold by Product
plt.subplot(3, 3, 4)
units_by_product = df.groupby('Product')['Units_Sold'].sum().sort_values(ascending=False)
plt.bar(units_by_product.index, units_by_product.values, color='orange', alpha=0.7)
plt.title('Units Sold by Product', fontsize=12, fontweight='bold')
plt.ylabel('Units')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')

# Plot 5: Profit by Region
plt.subplot(3, 3, 5)
profit_by_region = df.groupby('Region')['Profit'].sum().sort_values()
plt.barh(profit_by_region.index, profit_by_region.values, color='darkgreen', alpha=0.7)
plt.title('Profit by Region', fontsize=12, fontweight='bold')
plt.xlabel('Profit (₹)')
plt.grid(True, alpha=0.3, axis='x')

# Plot 6: Top 5 Salespeople
plt.subplot(3, 3, 6)
top_sales = df.groupby('Salesperson')['Total_Sales'].sum().nlargest(5)
plt.bar(range(len(top_sales)), top_sales.values, color='purple', alpha=0.7)
plt.xticks(range(len(top_sales)), top_sales.index, rotation=45)
plt.title('Top 5 Salespeople', fontsize=12, fontweight='bold')
plt.ylabel('Sales (₹)')
plt.grid(True, alpha=0.3, axis='y')

# Plot 7: Daily Sales Trend (Last 90 days)
plt.subplot(3, 3, 7)
recent_data = df[df['Date'] >= (df['Date'].max() - pd.Timedelta(days=90))]
daily_sales = recent_data.groupby('Date')['Total_Sales'].sum()
plt.plot(daily_sales.index, daily_sales.values, linewidth=1, color='darkblue', alpha=0.6)
plt.title('Daily Sales (Last 90 Days)', fontsize=12, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Sales (₹)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Plot 8: Quarterly Comparison
plt.subplot(3, 3, 8)
quarterly_sales = df.groupby('Quarter')['Total_Sales'].sum()
plt.bar(range(len(quarterly_sales)), quarterly_sales.values, color='teal', alpha=0.7)
plt.xticks(range(len(quarterly_sales)), quarterly_sales.index.astype(str), rotation=45)
plt.title('Quarterly Sales', fontsize=12, fontweight='bold')
plt.ylabel('Sales (₹)')
plt.grid(True, alpha=0.3, axis='y')

# Plot 9: Profit Margin by Product
plt.subplot(3, 3, 9)
product_margin = product_summary['Profit_Margin_%'].sort_values()
colors_margin = ['red' if x < 38 else 'yellow' if x < 42 else 'green' for x in product_margin.values]
plt.barh(product_margin.index, product_margin.values, color=colors_margin, alpha=0.7)
plt.title('Profit Margin % by Product', fontsize=12, fontweight='bold')
plt.xlabel('Profit Margin %')
plt.axvline(x=40, color='black', linestyle='--', linewidth=0.8, label='Target: 40%')
plt.legend()
plt.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('sales_dashboard.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved sales_dashboard.png")
plt.show()

# EXPORT SUMMARY REPORTS
print("\n" + "="*70)
print("EXPORTING SUMMARY REPORTS...")
print("="*70)

product_summary.to_csv('product_performance.csv')
print("✓ Saved product_performance.csv")

regional_summary.to_csv('regional_performance.csv')
print("✓ Saved regional_performance.csv")

monthly_sales.to_csv('monthly_trends.csv')
print("✓ Saved monthly_trends.csv")

salesperson_summary.to_csv('salesperson_performance.csv')
print("✓ Saved salesperson_performance.csv")

#EXECUTIVE SUMMARY
executive_summary = pd.DataFrame({
    'Metric': [
        'Total Revenue (₹)',
        'Total Profit (₹)',
        'Profit Margin (%)',
        'Total Units Sold',
        'Average Transaction (₹)',
        'Total Transactions',
        'Best Product',
        'Top Region',
        'Best Month'
    ],
    'Value': [
        f"{total_revenue:,.0f}",
        f"{total_profit:,.0f}",
        f"{profit_margin:.2f}",
        f"{total_units:,}",
        f"{avg_transaction:,.0f}",
        f"{len(df):,}",
        product_summary.index[0],
        regional_summary.index[0],
        str(best_month)
    ]
})

executive_summary.to_csv('executive_summary.csv', index=False)
print("✓ Saved executive_summary.csv")

print("\n" + "="*70)
print(" SALES DASHBOARD COMPLETE!")
print("="*70)

print("\n Files Created:")
print("1. sales_raw_data.csv - Complete transaction data")
print("2. sales_dashboard.png - Comprehensive visualization")
print("3. sales_pivot_region_product.csv - Pivot analysis")
print("4. product_performance.csv - Product metrics")
print("5. regional_performance.csv - Regional breakdown")
print("6. monthly_trends.csv - Time series analysis")
print("7. salesperson_performance.csv - Team performance")
print("8. executive_summary.csv - Key metrics overview")

print("\n Key Insights:")
print(f"• Total business: ₹{total_revenue/10000000:.2f} Crores")
print(f"• Profit margin: {profit_margin:.2f}% {'' if profit_margin >= 40 else 'Below target'}")
print(f"• Best product: {product_summary.index[0]} (₹{product_summary.iloc[0]['Total_Sales']/100000:.2f}L)")
print(f"• Top region: {regional_summary.index[0]} ({regional_summary.iloc[0]['% of Total Sales']:.1f}% of sales)")
print(f"• Growth trend: {' Positive' if monthly_sales['Sales_Growth_%'].iloc[-1] > 0 else ' Negative'}")

