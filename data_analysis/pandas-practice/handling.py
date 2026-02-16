import pandas as pd
import numpy as np

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'Age': [25, np.nan, 30, 28, np.nan],
    'Salary': [50000, 60000, np.nan, 55000, 58000],
    'Department': ['IT', 'HR', 'IT', np.nan, 'Finance']
}
df=pd.DataFrame(data)

print("DATA WITH MISIING VALUES")
print(df)
print()

print("Missing values per column:")
print(df.isnull().sum())
print()

print("Any missing values?", df.isnull().any().any())
print()

# Drop rows with any missing value
df_dropped = df.dropna()
print("After dropping rows with NaN:")
print(df_dropped)
print()



df_filled=df.fillna({
    'Age':df['Age'].mean(),
    'Salary':df['Salary'].median(),
    'Department':'Unknown'

})

print("AFTER FILLING NAN")
print(df_filled)
print()

df_ffill = df.ffill()
print("Forward fill:")
print(df_ffill)