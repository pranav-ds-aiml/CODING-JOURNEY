import pandas as pd
import numpy as np

# Create employee data
employees = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank'],
    'Department': ['IT', 'HR', 'IT', 'Finance', 'HR', 'IT'],
    'Salary': [60000, 50000, 65000, 55000, np.nan, 70000],
    'Years': [3, 5, 2, 4, 6, 1],
    'City': ['Bangalore', 'Mumbai', 'Bangalore', 'Delhi', 'Mumbai', 'Bangalore']
})

dept_avg = employees.groupby('Department')['Salary'].transform('mean')
employees['Salary'] = employees['Salary'].fillna(dept_avg)
print("1. After filling Salary:")
print(employees)
print()

def categorize_salary(sal):
 if sal < 55000:
  return 'Low'
 elif sal <= 65000:
  return 'Medium'
 else:
  return 'High'
employees['Salary_Category'] = employees['Salary'].apply(categorize_salary)
print("2. With Salary Category:")
print(employees)
print()

avg_by_dept = employees.groupby('Department')['Salary'].mean()
print("3. Average Salary by Department:")
print(avg_by_dept)
print()

count=employees.groupby(['Department','Size']).size()
print("4. Count by Department and City:")
print(count)
print()

sorted_emp=employees.sort_values('Salary',ascending=False)
print("5>Sorted by Salary:")
print(sorted_emp)
print()

top_3 = employees.nlargest(3, 'Salary')
print("6. Top 3 earners:")
print(top_3[['Name', 'Salary']])
print()

it_dept = employees[employees['Department'] == 'IT']
print("7. IT Department:")
print(it_dept)
print()

employees['New_Salary'] = employees['Salary'] * 1.1
print("8. After 10% increase:")
print(employees[['Name', 'Salary', 'New_Salary']])
print()

bangalore_high = employees[(employees['City'] == 'Bangalore') & (employees['Salary'] > 60000)]
print("9. Bangalore with Salary > 60000:")
print(bangalore_high)
print()

total_by_dept = employees.groupby('Department')['Salary'].sum()
print("10. Total Salary by Department:")
print(total_by_dept)
