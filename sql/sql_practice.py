import sqlite3
import pandas as pd

conn = sqlite3.connect('company.db')

print("="*70)
print("SQL PRACTICE PROBLEMS")
print("="*70)

# Problem 1
print("\n1. Find all employees hired in 2021:")
query = """
SELECT first_name, last_name, hire_date 
FROM employees 
WHERE hire_date LIKE '2021%'
"""
print(pd.read_sql_query(query, conn))
print()

# Problem 2
print("2. Find the second highest salary:")
query = """
SELECT DISTINCT salary 
FROM employees 
ORDER BY salary DESC 
LIMIT 1 OFFSET 1
"""
print(pd.read_sql_query(query, conn))
print()

# Problem 3
print("3. Count employees by hire year:")
query = """
SELECT 
    SUBSTR(hire_date, 1, 4) as hire_year,
    COUNT(*) as employee_count
FROM employees
GROUP BY hire_year
ORDER BY hire_year
"""
print(pd.read_sql_query(query, conn))
print()

# Problem 4
print("4. Find employees without managers:")
query = """
SELECT first_name, last_name, department 
FROM employees 
WHERE manager_id IS NULL
"""
print(pd.read_sql_query(query, conn))
print()

# Problem 5
print("5. Department with highest average salary:")
query = """
SELECT 
    department,
    AVG(salary) as avg_salary
FROM employees
GROUP BY department
ORDER BY avg_salary DESC
LIMIT 1
"""
print(pd.read_sql_query(query, conn))
print()

# Problem 6
print("6. Find duplicate email domains:")
query = """
SELECT 
    SUBSTR(email, INSTR(email, '@') + 1) as domain,
    COUNT(*) as count
FROM employees
GROUP BY domain
"""
print(pd.read_sql_query(query, conn))
print()

# Problem 7
print("7. Employees with name longer than 6 characters:")
query = """
SELECT first_name, last_name 
FROM employees 
WHERE LENGTH(first_name) > 6
"""
print(pd.read_sql_query(query, conn))
print()

# Problem 8
print("8. Total budget vs total salary by department:")
query = """
SELECT 
    d.department_name,
    d.budget as department_budget,
    SUM(e.salary) as total_salaries,
    d.budget - SUM(e.salary) as remaining_budget
FROM departments d
LEFT JOIN employees e ON d.department_name = e.department
GROUP BY d.department_name, d.budget
"""
print(pd.read_sql_query(query, conn))
print()

# Problem 9
print("9. Find employees earning more than their department average:")
query = """
SELECT 
    e.first_name,
    e.last_name,
    e.department,
    e.salary,
    dept_avg.avg_sal as dept_avg_salary
FROM employees e
JOIN (
    SELECT department, AVG(salary) as avg_sal
    FROM employees
    GROUP BY department
) dept_avg ON e.department = dept_avg.department
WHERE e.salary > dept_avg.avg_sal
ORDER BY e.department, e.salary DESC
"""
print(pd.read_sql_query(query, conn))
print()

# Problem 10
print("10. Running total of salaries by hire date:")
query = """
SELECT 
    first_name,
    last_name,
    hire_date,
    salary,
    (SELECT SUM(salary) 
     FROM employees e2 
     WHERE e2.hire_date <= e1.hire_date) as running_total
FROM employees e1
ORDER BY hire_date
"""
print(pd.read_sql_query(query, conn))
print()

conn.close()

print("="*70)
print("✅ PRACTICE PROBLEMS COMPLETE!")
print("="*70)