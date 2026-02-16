import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'Score': [95, 87, 92, 88, 95],
    'Age': [20, 22, 21, 20, 23]
}
df = pd.DataFrame(data)

print("Original:")
print(df)
print()

# Sort by Score (ascending)
sorted_asc = df.sort_values('Score')
print("Sorted by Score (ascending):")
print(sorted_asc)
print()

sorted_desc = df.sort_values('Score', ascending=False)
print("Sorted by Score (descending):")
print(sorted_desc)
print()

sorted_multi = df.sort_values(['Score', 'Age'], ascending=[False, True])
print("Sorted by Score (desc) then Age (asc):")
print(sorted_multi)
print()

# Add rank column
df['Rank'] = df['Score'].rank(ascending=False, method='min')
print("With Rank:")
print(df.sort_values('Rank'))
