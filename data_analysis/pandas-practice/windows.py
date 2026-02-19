import pandas as pd
import numpy as np

dates = pd.date_range('2024-01-01', periods=20, freq='D')
sales = np.random.randint(10000, 50000, 20)

df = pd.DataFrame({
    'Date': dates,
    'Sales': sales
})

print("ORIGINAL DATA:")
print(df)
print()

df['Rolling_Mean_3']=df['Sales'].rolling(window=3).mean()
df['Rolling_Mean_7']=df['Sales'].rolling(window=7).mean()

print("WITH ROLLING NAMES:")
print(df)
print()

df['Rolling_Sum_3'] = df['Sales'].rolling(window=3).sum()
df['Rolling_Min_3'] = df['Sales'].rolling(window=3).min()
df['Rolling_Max_3'] = df['Sales'].rolling(window=3).max()

print("WITH MULTIPLE ROLLING CALCULATIONS:")
print(df[['Date','Sales','Rolling_Mean_3','Rolling_Min_3','Rolling_Max_3']])
print()

df['Cumulative_Sum']=df['Sales'].expanding.sum()
df['Cumulative_Mean'] = df['Sales'].expanding().mean()

print("WITH EXPANDING CALCULATIONS")
print(df[['Date','Sales','Cumulative_Sum','Cumulative_Mean']].tail(10))
