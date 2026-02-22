import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("SQL + PYTHON: EMPLOYEE ANALYTICS DASHBOARD")
print("="*70)

conn = sqlite3.connect('company.db')

# ANALYSIS 1: Salary Distribution
print("\n1. Analyzing salary distribution...")
query="SELECT department,salary FROM employees"
df=pd.read_sql_query(query,conn)

plt.figure(figsize=(12,6))
sns.boxplot(data=df,x='department',y='salary',palette='Set2')
plt.title('Salary Distribution by Department', fontsize=14, fontweight='bold')
plt.ylabel('Salary (₹)')
plt.xlabel('Department')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('sql_analysis_1_salary_dist.png', dpi=300)
plt.close()
print("✓ Saved sql_analysis_1_salary_dist.png")

# ANALYSIS 2: Department Comparison

print("2. Creating department comparison...")

query="""
SELECT
      department,
      COUNT(*) as employee_count,
      AVG(salary) as avg_salary,
      SUM(salary) as total_salary
FROM employees
GROUP BY  department
"""

dept_df=pd.read_sql_query(query,conn)
fig,axes=plt.subplots(1,2,figsize=(14,6))

#Employee Count
axes[0].bar(dept_df['department'], dept_df['employee_count'], 
            color='steelblue', alpha=0.7)
axes[0].set_title('Employees per Department', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].set_xlabel('Department')
axes[0].grid(True, alpha=0.3, axis='y')

#Average Salary
axes[1].barh(dept_df['department'], dept_df['avg_salary'], 
             color='coral', alpha=0.7)
axes[1].set_title('Average Salary by Department', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Average Salary (₹)')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('sql_analysis_2_dept_comparison.png', dpi=300)
plt.close()
print("✓ Saved sql_analysis_2_dept_comparison.png")

#ANALYSIS 3
print("3. Analyzing hiring trends...")

query = """
SELECT 
    SUBSTR(hire_date, 1, 4) as year,
    COUNT(*) as hires
FROM employees
GROUP BY year
ORDER BY year
"""
hire_df = pd.read_sql_query(query, conn)

plt.figure(figsize=(10, 6))
plt.plot(hire_df['year'], hire_df['hires'], marker='o', 
         linewidth=2, markersize=10, color='green')
plt.title('Hiring Trend Over Years', fontsize=14, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Number of Hires')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sql_analysis_3_hiring_trend.png', dpi=300)
plt.close()
print("✓ Saved sql_analysis_3_hiring_trend.png")

print("\n4. Generating summary report...")

# Executive summary
exec_query = """
SELECT 
    COUNT(*) as total_employees,
    AVG(salary) as avg_salary,
    MIN(salary) as min_salary,
    MAX(salary) as max_salary,
    SUM(salary) as total_payroll
FROM employees
"""
exec_df = pd.read_sql_query(exec_query, conn)

# Department breakdown
dept_query = """
SELECT 
    department,
    COUNT(*) as count,
    AVG(salary) as avg_sal,
    MIN(salary) as min_sal,
    MAX(salary) as max_sal
FROM employees
GROUP BY department
ORDER BY count DESC
"""
dept_breakdown = pd.read_sql_query(dept_query, conn)

# Save reports
exec_df.to_csv('sql_report_executive_summary.csv', index=False)
dept_breakdown.to_csv('sql_report_department_breakdown.csv', index=False)

print("✓ Saved sql_report_executive_summary.csv")
print("✓ Saved sql_report_department_breakdown.csv")

print("\n" + "="*70)
print("EXECUTIVE SUMMARY")
print("="*70)
print(exec_df.to_string(index=False))

print("\n" + "="*70)
print("DEPARTMENT BREAKDOWN")
print("="*70)
print(dept_breakdown.to_string(index=False))

print("\n" + "="*70)
print(" SQL ANALYTICS DASHBOARD COMPLETE!")
print("="*70)
print("\n Files Created:")
print("1. sql_analysis_1_salary_dist.png")
print("2. sql_analysis_2_dept_comparison.png")
print("3. sql_analysis_3_hiring_trend.png")
print("4. sql_report_executive_summary.csv")
print("5. sql_report_department_breakdown.csv")

conn.close()