import pandas as pd

data={
    'NAME':['PRANAV','ALICE','BOB','CHARLIE',"DIANA"],
    'AGE':[20,30,21,23,20],
    'SCORE':[95,87,92,88,91],
    'CITY':['BANGALORE','MUMBAI','DELHI','BANGALORE',"MUMBAI"],
}

df=pd.DataFrame(data)

print(df)

print("NAMES OF COLUMN:")
print(df['NAME'])
print()

print("NAME AND SCORE:")
print(df[['NAME','SCORE']])
print()

print("FIRST ROW:")
print(df.iloc[0])
print()

print("FIRST 3 ROWS")
print(df.iloc[0:3])
print()

print("ROW AT INDEX 2:")
print(df.loc[2])
print()

print("NAME OF PERSON INDEX 1:")
print(df.loc[1,'NAME'])
print()

print("PEOPLE WITH SCORE>90")
print(df[df['SCORE']>90])
print()

print("People in Bangalore:")
print(df[df['CITY'] == 'BANGALORE'])
print()

print("Age 20 AND Score > 90:")
print(df[(df['AGE'] == 20) & (df['SCORE'] > 90)])

