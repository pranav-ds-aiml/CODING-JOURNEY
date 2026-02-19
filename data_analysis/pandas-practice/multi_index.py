import pandas as pd

arrays = [
    ['North', 'North', 'South', 'South', 'East', 'East'],
    ['Q1', 'Q2', 'Q1', 'Q2', 'Q1', 'Q2']
]

index = pd.MultiIndex.from_arrays(arrays, names=['Region', 'Quarter'])

data = {
    'Sales': [50000, 55000, 45000, 48000, 52000, 54000],
    'Profit': [10000, 11000, 9000, 9500, 10500, 11000]
}

df = pd.DataFrame(data, index=index)

print("MultiIndex DataFrame:")
print(df)
print()

print("North Region:")
print(df.loc['North'])
print()

print("South Q2:")
print(df.loc[('South', 'Q2')])
print()

print("Total by Region:")
print(df.groupby(level='Region').sum())
print()

print("TOTAL BY QUARTER:")
print(df.groupby(level='Quarter').sum())
print()

df_reset=df.reset_index()
print("RESET TO REGULAR DATAFRAME:")
[print(df_reset)]