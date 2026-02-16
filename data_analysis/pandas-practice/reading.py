import pandas as pd

# Sample data
data = {
    'Product': ['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard'],
    'Price': [50000, 30000, 25000, 15000, 3000],
    'Quantity': [5, 10, 8, 12, 20],
    'Category': ['Electronics', 'Electronics', 'Electronics', 'Electronics', 'Accessories']
}
df = pd.DataFrame(data)

df.to_csv('products.csv',index=False)
print("SAVED TO PRODUCTS")

df_read=pd.read_csv('products.csv')
print("\nREAD FROM CSV:")
print(df_read)
print()
print(df)

df.to_excel('products.xlsx', index=False)
print("✓ Saved to products.xlsx")

# Read from Excel
df_excel = pd.read_excel('products.xlsx')
print("\nRead from Excel:")
print(df_excel)
