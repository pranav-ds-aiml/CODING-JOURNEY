import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Math': [85, 92, 78, 95],
    'Science': [88, 85, 92, 90],
    'English': [90, 88, 85, 92]
}

df=pd.DataFrame(data)

df['Math_Grade']=df['Math'].apply(lambda x:'A' if x>=90 else 'B' if x>=80 else 'C')


print(("WITH MATH GRADE:"))
print(df)
print()

def calculate_grade(row):
    avg=(row['Math']+row['Science']+row['English'])/3
    if avg>=90:
        return 'A+'
    elif avg>=85:
        return 'A'
    elif avg>=80:
        return 'B+'
    else:
        return 'B'

df['Overall_Grade']=df.apply(calculate_grade,axis=1)

print("WITH OVERALL GRADE:")
print(df)
print()

grade_points={'A+':4.0,'A':3.7,'B+':3.3,'B':3.0,'C':2.0}
df['GPA']=df['Overall_Grade'].map(grade_points)

print("WITH GPA:")
print(df)

