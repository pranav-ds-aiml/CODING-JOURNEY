import pandas as pd

# DataFrame = 2D table (like Excel sheet)

# Create from dictionary
data = {
    'Name': ['Pranav', 'Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [20, 22, 21, 23, 20],
    'Score': [95, 87, 92, 88, 91],
    'City': ['Bangalore', 'Mumbai', 'Delhi', 'Bangalore', 'Mumbai']
}

df = pd.DataFrame(data)
print("DataFrame:")
print(df)
print()

print("Shape:", df.shape)  # (5, 4) - 5 rows, 4 columns
print("Columns:", df.columns.tolist())
print("Index:", df.index.tolist())
print()

print("First 3 rows:")
print(df.head(3))
print()

print("Last 2 rows:")
print(df.tail(2))
print()

# Info about DataFrame
print("DataFrame info:")
df.info()
print()

# Statistical summary
print("Statistical summary:")
print(df.describe())