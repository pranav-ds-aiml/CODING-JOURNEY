import pandas as pd
import numpy as np

data = {
    'Date': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02', 
             '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-03'],
    'Region': ['North', 'South', 'North', 'South', 'East', 'East', 'North', 'South'],
    'Product': ['Laptop', 'Phone', 'Laptop', 'Phone', 'Tablet', 'Laptop', 'Phone', 'Tablet'],
    'Sales': [50000, 30000, 55000, 32000, 25000, 27000, 35000, 28000],
    'Units': [5, 10, 6, 11, 8, 9, 12, 10]
}

df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])

print("Original Data:")
print(df)
print()

pivot=pd.pivot_table(
    df,
    values='Sales',
    index='Region',
    columns='Product',
    aggfunc='sum',
    fill_value=0
)

print("PIVOT TABLE -SALES BY REGION AND PRODUCT:")
print(pivot)
print()

pivot_multi=pd.pivot_table(
    df,
    values='Sales',
    index='Region',
    columns='Product',
    aggfunc=['sum','mean','count'],fill_value=0

).fillna=0

print("MULTIPLE AGGREGATIONS:")
print(pivot_multi)
print()

pivot_with_totals=pd.pivot_table(
    df,
    values='Sales',
    index='Region',
    columns='Product',
    aggfunc='sum',
    fill_value=0,
    margins=True,
    margins_name='Total'
)

print("PIVOT WITH TOTALS:")
print(pivot_with_totals)
print()