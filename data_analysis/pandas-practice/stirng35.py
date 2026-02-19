import pandas as pd

data = {
    'Name': ['  alice  ', 'BOB', 'charlie', 'DIANA'],
    'Email': ['alice@email.com', 'bob@email.COM', 'charlie@email.com', 'diana@EMAIL.com']
}
df = pd.DataFrame(data)

print("Original:")
print(df)
print()

# Clean names (strip spaces, capitalize)
df['Name_Clean'] = df['Name'].str.strip().str.capitalize()
print("After cleaning names:")
print(df)
print()

# Lowercase emails
df['Email_Lower'] = df['Email'].str.lower()
print("After lowercasing emails:")
print(df)
print()

# Check if contains
df['Has_Charlie'] = df['Name_Clean'].str.contains('Charlie')
print("Check for 'Charlie':")
print(df)
print()

# Extract domain from email
df['Domain'] = df['Email_Lower'].str.split('@').str[1]
print("Extract domain:")
print(df[['Email_Lower', 'Domain']])