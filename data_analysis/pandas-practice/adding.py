import pandas as pd

data = {
    'Name': ['Pranav', 'Alice', 'Bob'],
    'Score': [95, 87, 92]
}
df = pd.DataFrame(data)

print("Original:")
print(df)
print()

# Add new column
df['Grade'] = ['A+', 'B+', 'A']
print("After adding Grade:")
print(df)
print()

df['Percentage']=df['Score']
df['Pass']=df['Score']>=90
print(df)
print()

new_student=pd.DataFrame({
    'Name': ['Charlie'],
    'Score': [88],
    'Grade': ['B+'],
    'Percentage': [88],
    'Pass': [False]
})

df=pd.concat([df,new_student],ignore_index=True)
print("AFTER ADDING NEW STUDENT")
print(df)
print()

df=df.drop('Pass',axis=1)
print(df)
print()

df=df.drop(0,axis=0)
print(df)
