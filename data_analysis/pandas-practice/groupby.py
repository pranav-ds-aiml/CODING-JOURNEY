import pandas as pd

# Sales data
sales = pd.DataFrame({
    'Product': ['Laptop', 'Phone', 'Laptop', 'Tablet', 'Phone', 'Tablet', 'Laptop'],
    'Region': ['North', 'South', 'North', 'East', 'North', 'South', 'East'],
    'Sales': [50000, 30000, 55000, 25000, 32000, 27000, 52000],
    'Units': [5, 10, 6, 8, 11, 9, 5]
})

print("Sales data:")
print(sales)
print()

# Group by Product
by_product = sales.groupby('Product')['Sales'].sum()
print("Total sales by Product:")
print(by_product)
print()

by_region = sales.groupby('Region')['Sales'].sum()
print("Total sales by Region:")
print(by_region)
print()

summary = sales.groupby('Product').agg({
    'Sales': ['sum', 'mean', 'count'],
    'Units': 'sum'
})
print("Product summary:")
print(summary)
print()

# Group by multiple columns
by_prod_region = sales.groupby(['Product', 'Region'])['Sales'].sum()
print("Sales by Product and Region:")
print(by_prod_region)