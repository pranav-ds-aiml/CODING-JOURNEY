import pandas as pd

# Two DataFrames to merge
students = pd.DataFrame({
    'StudentID': [1, 2, 3, 4],
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [20, 21, 22, 20]
})

scores = pd.DataFrame({
    'StudentID': [1, 2, 3, 5],
    'Score': [95, 87, 92, 88],
    'Subject': ['Math', 'Math', 'Math', 'Math']
})

print("Students:")
print(students)
print()

print("Scores:")
print(scores)
print()

inner = pd.merge(students, scores, on='StudentID', how='inner')
print("Inner join:")
print(inner)
print()

left = pd.merge(students, scores, on='StudentID', how='left')
print("Left join:")
print(left)
print()

right = pd.merge(students, scores, on='StudentID', how='right')
print("Right join:")
print(right)
print()

# Outer join (all records from both)
outer = pd.merge(students, scores, on='StudentID', how='outer')
print("Outer join:")
print(outer)
