import pandas as pd

students = pd.DataFrame({
    'Name': ['Amit', 'Priya', 'Raj', 'Sneha', 'Vikram', 'Ananya'],
    'Math': [85, 92, 78, 95, 88, 90],
    'Science': [88, 85, 92, 90, 85, 93],
    'English': [90, 88, 85, 92, 90, 88],
    'City': ['Delhi', 'Mumbai', 'Delhi', 'Bangalore', 'Mumbai', 'Bangalore']
})
df=pd.DataFrame(students)

print("FIRST 3 ROWS:")
print(df.iloc[0:3])
print()
print(df[['Name','Math']])

df['Total']=students['Math'] + students['Science'] + students['English']
print(df)
print()

df['Average']=df['Total']/3
print(df)
print()

print(df[df['City']=='Bangalore'])
print()
print(df[(df['Math']>90) & (df['Science']>85)])

sorted_values=df.sort_values('Average',ascending=False)
print(sorted_values)
print()

top_student=df.loc[df['Average'].idxmax()]
print(top_student)
print()

df.to_csv('students.csv',index=False)
print("SAVED TO STUDENTS.CSV")