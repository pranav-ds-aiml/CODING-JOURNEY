import sqlite3
import pandas as pd

# Create database and connection
conn = sqlite3.connect('company.db')
cursor = conn.cursor()

print("="*70)
print("SQL BASICS - LEARNING BY DOING")
print("="*70)
print("\n✓ Connected to database 'company.db'\n")

# ==================================================================
# CREATE TABLES
# ==================================================================

print("Creating tables...")

# Employees table
cursor.execute('''
CREATE TABLE IF NOT EXISTS employees (
    employee_id INTEGER PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT UNIQUE,
    department TEXT,
    salary REAL,
    hire_date TEXT,
    manager_id INTEGER
)
''')

# Departments table
cursor.execute('''
CREATE TABLE IF NOT EXISTS departments (
    department_id INTEGER PRIMARY KEY,
    department_name TEXT NOT NULL,
    location TEXT,
    budget REAL
)
''')

# Projects table
cursor.execute('''
CREATE TABLE IF NOT EXISTS projects (
    project_id INTEGER PRIMARY KEY,
    project_name TEXT NOT NULL,
    department TEXT,
    budget REAL,
    start_date TEXT,
    end_date TEXT
)
''')

conn.commit()
print("✓ Tables created\n")

# ==================================================================
# INSERT SAMPLE DATA
# ==================================================================

print("Inserting sample data...")

# Clear existing data
cursor.execute('DELETE FROM employees')
cursor.execute('DELETE FROM departments')
cursor.execute('DELETE FROM projects')

# Insert employees
employees_data = [
    (1, 'Rajesh', 'Kumar', 'rajesh.k@company.com', 'IT', 75000, '2020-01-15', None),
    (2, 'Priya', 'Sharma', 'priya.s@company.com', 'HR', 55000, '2019-03-20', None),
    (3, 'Amit', 'Singh', 'amit.s@company.com', 'IT', 68000, '2021-06-10', 1),
    (4, 'Sneha', 'Patel', 'sneha.p@company.com', 'Finance', 72000, '2020-08-05', None),
    (5, 'Vikram', 'Reddy', 'vikram.r@company.com', 'IT', 65000, '2022-01-12', 1),
    (6, 'Ananya', 'Gupta', 'ananya.g@company.com', 'Sales', 60000, '2021-09-18', None),
    (7, 'Rahul', 'Verma', 'rahul.v@company.com', 'Sales', 58000, '2022-03-25', 6),
    (8, 'Kavya', 'Menon', 'kavya.m@company.com', 'HR', 52000, '2021-11-30', 2),
    (9, 'Arjun', 'Nair', 'arjun.n@company.com', 'Finance', 70000, '2020-05-14', 4),
    (10, 'Divya', 'Iyer', 'divya.i@company.com', 'IT', 71000, '2021-07-22', 1)
]

cursor.executemany('INSERT INTO employees VALUES (?,?,?,?,?,?,?,?)', employees_data)

# Insert departments
departments_data = [
    (1, 'IT', 'Bangalore', 500000),
    (2, 'HR', 'Mumbai', 200000),
    (3, 'Finance', 'Delhi', 300000),
    (4, 'Sales', 'Pune', 400000)
]

cursor.executemany('INSERT INTO departments VALUES (?,?,?,?)', departments_data)

# Insert projects
projects_data = [
    (1, 'Website Redesign', 'IT', 150000, '2024-01-01', '2024-06-30'),
    (2, 'HR Portal', 'HR', 80000, '2024-02-01', '2024-05-31'),
    (3, 'Financial System Upgrade', 'Finance', 200000, '2024-01-15', '2024-12-31'),
    (4, 'Sales CRM Implementation', 'Sales', 120000, '2024-03-01', '2024-08-31'),
    (5, 'Mobile App Development', 'IT', 180000, '2024-02-15', '2024-09-30')
]

cursor.executemany('INSERT INTO projects VALUES (?,?,?,?,?,?)', projects_data)

conn.commit()
print("✓ Sample data inserted\n")

# ==================================================================
# LESSON 1: SELECT - Basic Queries
# ==================================================================

print("="*70)
print("LESSON 1: SELECT STATEMENTS")
print("="*70)

# Query 1: Select all columns
print("\n1. SELECT * FROM employees (all columns, all rows):")
query = "SELECT * FROM employees"
result = pd.read_sql_query(query, conn)
print(result)
print()

# Query 2: Select specific columns
print("2. SELECT first_name, last_name, salary FROM employees:")
query = "SELECT first_name, last_name, salary FROM employees"
result = pd.read_sql_query(query, conn)
print(result)
print()

# Query 3: Select with alias
print("3. SELECT with column aliases:")
query = """
SELECT 
    first_name AS 'First Name',
    last_name AS 'Last Name',
    salary AS 'Annual Salary'
FROM employees
"""
result = pd.read_sql_query(query, conn)
print(result)
print()

# ==================================================================
# LESSON 2: WHERE - Filtering Data
# ==================================================================

print("="*70)
print("LESSON 2: WHERE CLAUSE (Filtering)")
print("="*70)

# Query 4: Simple WHERE
print("\n4. Employees in IT department:")
query = "SELECT * FROM employees WHERE department = 'IT'"
result = pd.read_sql_query(query, conn)
print(result)
print()

# Query 5: WHERE with comparison
print("5. Employees with salary > 65000:")
query = "SELECT first_name, last_name, salary FROM employees WHERE salary > 65000"
result = pd.read_sql_query(query, conn)
print(result)
print()

# Query 6: WHERE with AND
print("6. IT employees with salary > 65000:")
query = """
SELECT first_name, last_name, department, salary 
FROM employees 
WHERE department = 'IT' AND salary > 65000
"""
result = pd.read_sql_query(query, conn)
print(result)
print()

# Query 7: WHERE with OR
print("7. Employees in IT OR Sales:")
query = """
SELECT first_name, last_name, department 
FROM employees 
WHERE department = 'IT' OR department = 'Sales'
"""
result = pd.read_sql_query(query, conn)
print(result)
print()

# Query 8: WHERE with IN
print("8. Employees in IT, Sales, or HR (using IN):")
query = """
SELECT first_name, last_name, department 
FROM employees 
WHERE department IN ('IT', 'Sales', 'HR')
"""
result = pd.read_sql_query(query, conn)
print(result)
print()

# Query 9: WHERE with BETWEEN
print("9. Employees with salary between 60000 and 70000:")
query = """
SELECT first_name, last_name, salary 
FROM employees 
WHERE salary BETWEEN 60000 AND 70000
"""
result = pd.read_sql_query(query, conn)
print(result)
print()

# Query 10: WHERE with LIKE (pattern matching)
print("10. Employees whose name starts with 'A':")
query = """
SELECT first_name, last_name 
FROM employees 
WHERE first_name LIKE 'A%'
"""
result = pd.read_sql_query(query, conn)
print(result)
print()

# ==================================================================
# LESSON 3: ORDER BY - Sorting
# ==================================================================

print("="*70)
print("LESSON 3: ORDER BY (Sorting)")
print("="*70)

# Query 11: ORDER BY ascending
print("\n11. Employees sorted by salary (ascending):")
query = """
SELECT first_name, last_name, salary 
FROM employees 
ORDER BY salary
"""
result = pd.read_sql_query(query, conn)
print(result)
print()

# Query 12: ORDER BY descending
print("12. Employees sorted by salary (descending):")
query = """
SELECT first_name, last_name, salary 
FROM employees 
ORDER BY salary DESC
"""
result = pd.read_sql_query(query, conn)
print(result)
print()

# Query 13: ORDER BY multiple columns
print("13. Employees sorted by department, then salary:")
query = """
SELECT first_name, last_name, department, salary 
FROM employees 
ORDER BY department, salary DESC
"""
result = pd.read_sql_query(query, conn)
print(result)
print()

# ==================================================================
# LESSON 4: LIMIT - Restricting Results
# ==================================================================

print("="*70)
print("LESSON 4: LIMIT (Top N Records)")
print("="*70)

# Query 14: LIMIT
print("\n14. Top 5 highest paid employees:")
query = """
SELECT first_name, last_name, salary 
FROM employees 
ORDER BY salary DESC 
LIMIT 5
"""
result = pd.read_sql_query(query, conn)
print(result)
print()

# ==================================================================
# LESSON 5: Aggregate Functions
# ==================================================================

print("="*70)
print("LESSON 5: AGGREGATE FUNCTIONS")
print("="*70)

# Query 15: COUNT
print("\n15. Total number of employees:")
query = "SELECT COUNT(*) as total_employees FROM employees"
result = pd.read_sql_query(query, conn)
print(result)
print()

# Query 16: SUM
print("16. Total salary expense:")
query = "SELECT SUM(salary) as total_salary FROM employees"
result = pd.read_sql_query(query, conn)
print(result)
print()

# Query 17: AVG
print("17. Average salary:")
query = "SELECT AVG(salary) as average_salary FROM employees"
result = pd.read_sql_query(query, conn)
print(result)
print()

# Query 18: MIN and MAX
print("18. Minimum and maximum salary:")
query = """
SELECT 
    MIN(salary) as min_salary,
    MAX(salary) as max_salary
FROM employees
"""
result = pd.read_sql_query(query, conn)
print(result)
print()

# ==================================================================
# LESSON 6: GROUP BY
# ==================================================================

print("="*70)
print("LESSON 6: GROUP BY (Grouping Data)")
print("="*70)

# Query 19: GROUP BY simple
print("\n19. Number of employees in each department:")
query = """
SELECT 
    department,
    COUNT(*) as employee_count
FROM employees
GROUP BY department
"""
result = pd.read_sql_query(query, conn)
print(result)
print()

# Query 20: GROUP BY with multiple aggregates
print("20. Department salary statistics:")
query = """
SELECT 
    department,
    COUNT(*) as employee_count,
    AVG(salary) as avg_salary,
    MIN(salary) as min_salary,
    MAX(salary) as max_salary
FROM employees
GROUP BY department
"""
result = pd.read_sql_query(query, conn)
print(result)
print()

# Query 21: GROUP BY with HAVING
print("21. Departments with more than 2 employees:")
query = """
SELECT 
    department,
    COUNT(*) as employee_count
FROM employees
GROUP BY department
HAVING COUNT(*) > 2
"""
result = pd.read_sql_query(query, conn)
print(result)
print()

# ==================================================================
# LESSON 7: JOINS
# ==================================================================

print("="*70)
print("LESSON 7: JOINS (Combining Tables)")
print("="*70)

# First, let's create a mapping table
cursor.execute('''
CREATE TABLE IF NOT EXISTS employee_departments (
    employee_id INTEGER,
    department_name TEXT,
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
)
''')

cursor.execute('DELETE FROM employee_departments')

# Map employees to departments
emp_dept_data = [
    (1, 'IT'), (2, 'HR'), (3, 'IT'), (4, 'Finance'),
    (5, 'IT'), (6, 'Sales'), (7, 'Sales'), (8, 'HR'),
    (9, 'Finance'), (10, 'IT')
]

cursor.executemany('INSERT INTO employee_departments VALUES (?,?)', emp_dept_data)
conn.commit()

# Query 22: INNER JOIN
print("\n22. Employees with their department details (INNER JOIN):")
query = """
SELECT 
    e.first_name,
    e.last_name,
    e.salary,
    d.department_name,
    d.location,
    d.budget
FROM employees e
INNER JOIN employee_departments ed ON e.employee_id = ed.employee_id
INNER JOIN departments d ON ed.department_name = d.department_name
"""
result = pd.read_sql_query(query, conn)
print(result.head(10))
print()

# Query 23: Self JOIN (employees with their managers)
print("23. Employees with their managers (SELF JOIN):")
query = """
SELECT 
    e.first_name || ' ' || e.last_name AS employee,
    m.first_name || ' ' || m.last_name AS manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.employee_id
WHERE e.manager_id IS NOT NULL
"""
result = pd.read_sql_query(query, conn)
print(result)
print()

# ==================================================================
# LESSON 8: Subqueries
# ==================================================================

print("="*70)
print("LESSON 8: SUBQUERIES")
print("="*70)

# Query 24: Subquery in WHERE
print("\n24. Employees earning more than average:")
query = """
SELECT 
    first_name,
    last_name,
    salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees)
ORDER BY salary DESC
"""
result = pd.read_sql_query(query, conn)
print(result)
print()

# Query 25: Subquery in SELECT
print("25. Employees with salary difference from average:")
query = """
SELECT 
    first_name,
    last_name,
    salary,
    (SELECT AVG(salary) FROM employees) as avg_salary,
    salary - (SELECT AVG(salary) FROM employees) as difference
FROM employees
ORDER BY difference DESC
"""
result = pd.read_sql_query(query, conn)
print(result)
print()

# ==================================================================
# LESSON 9: CASE Statements
# ==================================================================

print("="*70)
print("LESSON 9: CASE STATEMENTS (Conditional Logic)")
print("="*70)

# Query 26: CASE statement
print("\n26. Salary categories:")
query = """
SELECT 
    first_name,
    last_name,
    salary,
    CASE 
        WHEN salary >= 70000 THEN 'High'
        WHEN salary >= 60000 THEN 'Medium'
        ELSE 'Low'
    END as salary_category
FROM employees
ORDER BY salary DESC
"""
result = pd.read_sql_query(query, conn)
print(result)
print()

# ==================================================================
# LESSON 10: Advanced Queries
# ==================================================================

print("="*70)
print("LESSON 10: COMPLEX QUERIES")
print("="*70)

# Query 27: Window functions simulation
print("\n27. Rank employees by salary within department:")
query = """
SELECT 
    first_name,
    last_name,
    department,
    salary,
    (SELECT COUNT(*) 
     FROM employees e2 
     WHERE e2.department = e1.department 
     AND e2.salary > e1.salary) + 1 as rank_in_dept
FROM employees e1
ORDER BY department, salary DESC
"""
result = pd.read_sql_query(query, conn)
print(result)
print()

# Query 28: Complex business query
print("28. Department statistics with project count:")
query = """
SELECT 
    d.department_name,
    COUNT(DISTINCT e.employee_id) as employee_count,
    AVG(e.salary) as avg_salary,
    COUNT(DISTINCT p.project_id) as project_count,
    d.budget as department_budget
FROM departments d
LEFT JOIN employees e ON d.department_name = e.department
LEFT JOIN projects p ON d.department_name = p.department
GROUP BY d.department_name, d.budget
ORDER BY employee_count DESC
"""
result = pd.read_sql_query(query, conn)
print(result)
print()

print("="*70)
print("✅ SQL FUNDAMENTALS COMPLETE!")
print("="*70)

# Close connection
conn.close()