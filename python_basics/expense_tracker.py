import csv
from datetime import datetime, timedelta
import os

# File to store expenses
EXPENSE_FILE = "expenses.csv"

# Categories and their monthly budgets
CATEGORIES = {
    "Food": 5000,
    "Transport": 2000,
    "Entertainment": 3000,
    "Shopping": 4000,
    "Bills": 6000,
    "Other": 2000
}

def create_sample_data():
    """Create sample expenses for testing"""
    if os.path.exists(EXPENSE_FILE):
        return  # Don't overwrite existing data
    
    print("📊 Creating sample data for demo...")
    
    sample_expenses = [
        {'date': datetime.now().strftime("%Y-%m-%d"), 'category': 'Food', 'amount': 450, 'description': 'Grocery shopping'},
        {'date': datetime.now().strftime("%Y-%m-%d"), 'category': 'Food', 'amount': 300, 'description': 'Restaurant dinner'},
        {'date': (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"), 'category': 'Transport', 'amount': 150, 'description': 'Uber ride'},
        {'date': (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"), 'category': 'Entertainment', 'amount': 800, 'description': 'Movie tickets'},
        {'date': (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"), 'category': 'Shopping', 'amount': 1200, 'description': 'New shoes'},
        {'date': (datetime.now() - timedelta(days=4)).strftime("%Y-%m-%d"), 'category': 'Bills', 'amount': 2500, 'description': 'Electricity bill'},
        {'date': (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"), 'category': 'Food', 'amount': 200, 'description': 'Coffee and snacks'},
        {'date': (datetime.now() - timedelta(days=6)).strftime("%Y-%m-%d"), 'category': 'Transport', 'amount': 100, 'description': 'Metro card recharge'},
    ]
    
    with open(EXPENSE_FILE, 'w', newline='') as file:
        fieldnames = ['date', 'category', 'amount', 'description']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_expenses)
    
    print("✓ Sample data created! You can now test all features.\n")

def load_expenses():
    """Load expenses from CSV file"""
    expenses = []
    if os.path.exists(EXPENSE_FILE):
        try:
            with open(EXPENSE_FILE, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    row['amount'] = float(row['amount'])
                    expenses.append(row)
        except Exception as e:
            print(f"Error loading expenses: {e}")
    return expenses

def save_expense(expense):
    """Save a single expense to CSV"""
    file_exists = os.path.exists(EXPENSE_FILE)
    
    try:
        with open(EXPENSE_FILE, 'a', newline='') as file:
            fieldnames = ['date', 'category', 'amount', 'description']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(expense)
    except Exception as e:
        print(f"Error saving expense: {e}")

def add_expense():
    """Add a new expense"""
    print("\n=== ADD EXPENSE ===")
    
    try:
        # Show categories
        print("\nCategories:")
        for i, (cat, budget) in enumerate(CATEGORIES.items(), 1):
            print(f"{i}. {cat} (Budget: ₹{budget}/month)")
        
        # Get category
        choice = int(input("\nChoose category (1-6): "))
        if choice < 1 or choice > 6:
            print("Invalid choice! Please choose 1-6.")
            input("\nPress Enter to continue...")
            return
            
        category = list(CATEGORIES.keys())[choice - 1]
        
        # Get amount
        amount = float(input("Enter amount: ₹"))
        if amount <= 0:
            print("Amount must be positive!")
            input("\nPress Enter to continue...")
            return
        
        # Get description
        description = input("Enter description: ")
        
        # Create expense
        expense = {
            'date': datetime.now().strftime("%Y-%m-%d"),
            'category': category,
            'amount': amount,
            'description': description
        }
        
        # Save it
        save_expense(expense)
        
        print("\n✓ Expense added successfully!")
        
        # Check if over budget
        check_budget_alert(category)
        
    except ValueError:
        print("\n❌ Invalid input! Please enter valid numbers.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    input("\nPress Enter to continue...")

def check_budget_alert(category):
    """Alert if category spending exceeds budget"""
    try:
        expenses = load_expenses()
        current_month = datetime.now().strftime("%Y-%m")
        
        # Calculate spending for this category this month
        monthly_spending = sum(
            exp['amount'] 
            for exp in expenses 
            if exp['category'] == category and exp['date'].startswith(current_month)
        )
        
        budget = CATEGORIES[category]
        
        if monthly_spending > budget:
            print(f"\n⚠️  ALERT! {category} spending (₹{monthly_spending:.2f}) exceeds budget (₹{budget})!")
        elif monthly_spending > budget * 0.8:
            print(f"\n⚠️  Warning: {category} at {(monthly_spending/budget*100):.0f}% of budget")
    except Exception as e:
        print(f"Error checking budget: {e}")

def view_all_expenses():
    """View all expenses"""
    try:
        expenses = load_expenses()
        
        if not expenses:
            print("\n📭 No expenses recorded yet!")
            print("Tip: Use option 1 to add your first expense.")
            input("\nPress Enter to continue...")
            return
        
        print("\n=== ALL EXPENSES ===")
        print(f"{'Date':<12} {'Category':<15} {'Amount':>10} {'Description':<30}")
        print("-" * 70)
        
        total = 0
        for exp in expenses:
            print(f"{exp['date']:<12} {exp['category']:<15} ₹{exp['amount']:>9.2f} {exp['description']:<30}")
            total += exp['amount']
        
        print("-" * 70)
        print(f"{'TOTAL':>27} ₹{total:>9.2f}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    input("\nPress Enter to continue...")

def monthly_summary():
    """Show monthly spending summary"""
    try:
        expenses = load_expenses()
        
        if not expenses:
            print("\n📭 No expenses recorded yet!")
            input("\nPress Enter to continue...")
            return
        
        current_month = datetime.now().strftime("%Y-%m")
        
        # Filter this month's expenses
        monthly_expenses = [exp for exp in expenses if exp['date'].startswith(current_month)]
        
        if not monthly_expenses:
            print(f"\n📭 No expenses for {datetime.now().strftime('%B %Y')} yet!")
            input("\nPress Enter to continue...")
            return
        
        print(f"\n=== MONTHLY SUMMARY ({datetime.now().strftime('%B %Y')}) ===\n")
        
        # Calculate category totals
        category_totals = {}
        grand_total = 0
        
        for exp in monthly_expenses:
            cat = exp['category']
            category_totals[cat] = category_totals.get(cat, 0) + exp['amount']
            grand_total += exp['amount']
        
        # Display
        print(f"{'Category':<15} {'Spent':>10} {'Budget':>10} {'Remaining':>12} {'Status'}")
        print("-" * 60)
        
        for cat, budget in CATEGORIES.items():
            spent = category_totals.get(cat, 0)
            remaining = budget - spent
            percentage = (spent / budget * 100) if budget > 0 else 0
            
            status = "✓" if spent <= budget else "⚠️ OVER"
            
            print(f"{cat:<15} ₹{spent:>9.2f} ₹{budget:>9.2f} ₹{remaining:>11.2f} {status}")
        
        print("-" * 60)
        total_budget = sum(CATEGORIES.values())
        print(f"{'TOTAL':<15} ₹{grand_total:>9.2f} ₹{total_budget:>9.2f} ₹{total_budget-grand_total:>11.2f}")
        
        # Visual bar chart
        print("\n--- Spending Visualization ---")
        for cat, budget in CATEGORIES.items():
            spent = category_totals.get(cat, 0)
            percentage = int((spent / budget * 100) if budget > 0 else 0)
            bar_length = min(percentage, 100) // 2  # Max 50 chars
            bar = "█" * bar_length
            print(f"{cat:<15} {bar} {percentage}%")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    input("\nPress Enter to continue...")

def category_breakdown():
    """Show expenses by category"""
    try:
        print("\nCategories:")
        for i, cat in enumerate(CATEGORIES.keys(), 1):
            print(f"{i}. {cat}")
        
        choice = int(input("\nChoose category (1-6): "))
        if choice < 1 or choice > 6:
            print("Invalid choice! Please choose 1-6.")
            input("\nPress Enter to continue...")
            return
            
        category = list(CATEGORIES.keys())[choice - 1]
        
        expenses = load_expenses()
        cat_expenses = [exp for exp in expenses if exp['category'] == category]
        
        if not cat_expenses:
            print(f"\n📭 No expenses in {category}!")
            input("\nPress Enter to continue...")
            return
        
        print(f"\n=== {category.upper()} EXPENSES ===")
        print(f"{'Date':<12} {'Amount':>10} {'Description':<30}")
        print("-" * 55)
        
        total = 0
        for exp in cat_expenses:
            print(f"{exp['date']:<12} ₹{exp['amount']:>9.2f} {exp['description']:<30}")
            total += exp['amount']
        
        print("-" * 55)
        print(f"{'TOTAL':>22} ₹{total:>9.2f}")
        
    except ValueError:
        print("\n❌ Invalid input! Please enter a number.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    input("\nPress Enter to continue...")

def main_menu():
    """Main menu loop"""
    # Create sample data if no file exists
    create_sample_data()
    
    while True:
        try:
            # Clear screen (optional - works on most terminals)
            print("\n" * 2)
            
            print("="*50)
            print("💰 EXPENSE TRACKER")
            print("="*50)
            print("1. Add expense")
            print("2. View all expenses")
            print("3. Monthly summary")
            print("4. Category breakdown")
            print("5. Exit")
            print("="*50)
            
            choice = input("\nChoose option (1-5): ")
            
            if choice == "1":
                add_expense()
            elif choice == "2":
                view_all_expenses()
            elif choice == "3":
                monthly_summary()
            elif choice == "4":
                category_breakdown()
            elif choice == "5":
                print("\n💾 All data saved to expenses.csv")
                print("Goodbye! 👋\n")
                input("Press Enter to exit...")
                break
            else:
                print("\n❌ Invalid choice! Please choose 1-5.")
                input("\nPress Enter to continue...")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            input("\nPress Enter to continue...")

# Run the program
if __name__ == "__main__":
    print("="*50)
    print("  EXPENSE TRACKER")
    print("  VS Code Edition")
    print("="*50)
    print()
    main_menu()
