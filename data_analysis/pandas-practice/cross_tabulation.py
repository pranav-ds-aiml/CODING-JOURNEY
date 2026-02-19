import pandas as pd

# Sample data
data = {
    'Customer': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry'],
    'Product': ['Laptop', 'Phone', 'Laptop', 'Tablet', 'Phone', 'Laptop', 'Tablet', 'Phone'],
    'Region': ['North', 'North', 'South', 'South', 'East', 'East', 'North', 'South'],
    'Purchased': [True, True, True, False, True, True, False, True]
}

df = pd.DataFrame(data)

crosstab=pd.crosstab(
    df['Region'],
    df['Product']
)

print("CROSS TABULATION-PRODUCTS BY REGION:")
print(crosstab)
print()

crosstab_pct=pd.crosstab(
    df['Region'],
    df['Product'],
    normalize='all'
)*100

print("CROSS-TABULATION(Percentages):")
print(crosstab_pct.round(2))
print()

crosstab_3way=pd.crosstab(
    df['Region'],df['Product'],df['Purchased']
)

print("THREE WAY CROSS TABULATION:")
print(crosstab_3way)