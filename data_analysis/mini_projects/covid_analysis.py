import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("="*50)
print("COVID-19 DATA ANALYSIS")
print("="*60)

url = "https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv"
print("\nLOADING COVID-19 DATA FROM ONLINE SOUCRCES...")
df=pd.read_csv(url)

print("DATA LOADED SUCCESSFULLY")
print(f"TOTAL RECORDS:{len(df)}")
print()

print("="*60)
print("DATA EXPLORATION")
print("="*60)

print("\nFIRST 10 ROWS")
print(df.head(10))
print()

print("DATASET INFO")
print(df.info())
print()

print("STATISTICAL SUMMARY")
print(df.describe())
print()

print("COLUMNS:",df.columns.tolist())
print()

print("MISSING VALUES")
print(df.isnull().sum())
print()

df['Date']=pd.to_datetime(df['Date'])

df=df.sort_values('Date')

print("Date range:", df['Date'].min(), "to", df['Date'].max())
print()

print("="*60)
print("ANALYSIS 1: TOP 10 COUNTRIES BY TOTAL CASES")
print("="*60)

latest_date = df['Date'].max()
latest_data = df[df['Date'] == latest_date]

top_10_cases = latest_data.nlargest(10, 'Confirmed')[['Country', 'Confirmed', 'Deaths', 'Recovered']]
print(f"\nAs of {latest_date.date()}:")
print(top_10_cases)
print()

print("="*60)
print("ANALYSIS 2: GLOBAL TRENDS")
print("="*60)

global_daily = df.groupby('Date').agg({
    'Confirmed': 'sum',
    'Deaths': 'sum',
    'Recovered': 'sum'
}).reset_index()

print("\nGLOBAL TOTALS (latest):")
print(global_daily.tail(1))
print()

global_daily['New_Cases']=global_daily['Confirmed'].diff()
global_daily['New_Deaths']=global_daily['Deaths'].diff()

print("PEAK NEW CASES DAY:")
peak_day=global_daily.loc[global_daily['New_Cases'].idxmax()]
print(f"DATE: {peak_day['Date'].date()}")
print(f"New Cases:{int(peak_day['New_Cases']):,}")
print()

print("="*60)
print("ANALYSIS 3: INDIA SPECIFIC ANALYSIS")
print("="*60)

india_data = df[df['Country'] == 'India'].copy()

if len(india_data) > 0:
    india_data['New_Cases'] = india_data['Confirmed'].diff()
    india_data['New_Deaths'] = india_data['Deaths'].diff()
    india_data['Active'] = india_data['Confirmed'] - india_data['Deaths'] - india_data['Recovered']
    
    print("\nIndia - Latest statistics:")
    latest_india = india_data.iloc[-1]
    print(f"Total Confirmed: {int(latest_india['Confirmed']):,}")
    print(f"Total Deaths: {int(latest_india['Deaths']):,}")
    print(f"Total Recovered: {int(latest_india['Recovered']):,}")
    print(f"Active Cases: {int(latest_india['Active']):,}")
    print()
    
    print("India - Peak day:")
    peak_india = india_data.loc[india_data['New_Cases'].idxmax()]
    print(f"Date: {peak_india['Date'].date()}")
    print(f"New cases: {int(peak_india['New_Cases']):,}")
    print()

print("="*60)
print("ANALYSIS 4: MORTALITY RATE ANALYSIS")
print("="*60)

latest_data['Mortality_Rate'] = (latest_data['Deaths'] / latest_data['Confirmed'] * 100).round(2)

significant = latest_data[latest_data['Confirmed'] > 1000]

highest_mortality = significant.nlargest(10, 'Mortality_Rate')[['Country', 'Confirmed', 'Deaths', 'Mortality_Rate']]
print("\nTop 10 countries by mortality rate (>1000 cases):")
print(highest_mortality)
print()

lowest_mortality = significant.nsmallest(10, 'Mortality_Rate')[['Country', 'Confirmed', 'Deaths', 'Mortality_Rate']]
print("Top 10 countries by lowest mortality rate (>1000 cases):")
print(lowest_mortality)
print()

print("="*60)
print("ANALYSIS 5: RECOVERY RATE ANALYSIS")
print("="*60)

latest_data['Recovery_Rate'] = (latest_data['Recovered'] / latest_data['Confirmed'] * 100).round(2)

significant = latest_data[latest_data['Confirmed'] > 1000]
best_recovery = significant.nlargest(10, 'Recovery_Rate')[['Country', 'Confirmed', 'Recovered', 'Recovery_Rate']]

print("\nTop 10 countries by recovery rate (>1000 cases):")
print(best_recovery)
print()

print("="*60)
print("CREATING VISUALIZATIONS...")
print("="*60)

plt.figure(figsize=(14,10))

#Plot 1: Global cumulative cases
plt.subplot(2, 2, 1)
plt.plot(global_daily['Date'], global_daily['Confirmed'], label='Confirmed', linewidth=2)
plt.plot(global_daily['Date'], global_daily['Deaths'], label='Deaths', linewidth=2)
plt.plot(global_daily['Date'], global_daily['Recovered'], label='Recovered', linewidth=2)
plt.title('Global COVID-19 Cumulative Cases', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Number of Cases')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Plot 2: Global daily new cases
plt.subplot(2, 2, 2)
plt.bar(global_daily['Date'], global_daily['New_Cases'], alpha=0.7, color='orange')
plt.title('Global Daily New Cases', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('New Cases')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Plot 3: Top 10 countries
plt.subplot(2, 2, 3)
top_10 = latest_data.nlargest(10, 'Confirmed')
plt.barh(top_10['Country'], top_10['Confirmed'])
plt.title('Top 10 Countries by Total Cases', fontsize=14, fontweight='bold')
plt.xlabel('Total Cases')
plt.ylabel('Country')
plt.grid(True, alpha=0.3, axis='x')

# Plot 4: India trend
if len(india_data) > 0:
    plt.subplot(2, 2, 4)
    plt.plot(india_data['Date'], india_data['Confirmed'], label='Confirmed', linewidth=2)
    plt.plot(india_data['Date'], india_data['Active'], label='Active', linewidth=2)
    plt.plot(india_data['Date'], india_data['Recovered'], label='Recovered', linewidth=2)
    plt.title('India COVID-19 Trends', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Number of Cases')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('covid_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved visualization as 'covid_analysis.png'")
plt.show()

print("\n" + "="*60)
print("SAVING RESULTS...")
print("="*60)

top_10_cases.to_csv('top_10_countries.csv', index=False)
print("✓ Saved top_10_countries.csv")

# Save India data
if len(india_data) > 0:
    india_data.to_csv('india_covid_data.csv', index=False)
    print("✓ Saved india_covid_data.csv")

summary = pd.DataFrame({
    'Metric': ['Total Confirmed', 'Total Deaths', 'Total Recovered', 
               'Global Mortality Rate %', 'Global Recovery Rate %'],
    'Value': [
        f"{int(global_daily.iloc[-1]['Confirmed']):,}",
        f"{int(global_daily.iloc[-1]['Deaths']):,}",
        f"{int(global_daily.iloc[-1]['Recovered']):,}",
        f"{(global_daily.iloc[-1]['Deaths'] / global_daily.iloc[-1]['Confirmed'] * 100):.2f}",
        f"{(global_daily.iloc[-1]['Recovered'] / global_daily.iloc[-1]['Confirmed'] * 100):.2f}"
    ]
})

summary.to_csv('covid_summary.csv', index=False)
print("✓ Saved covid_summary.csv")

print("\n" + "="*60)
print("✅ COVID-19 ANALYSIS COMPLETE!")
print("="*60)
print("\nFiles created:")
print("1. covid_analysis.png - Visualization")
print("2. top_10_countries.csv - Top 10 countries data")
print("3. india_covid_data.csv - India detailed data")
print("4. covid_summary.csv - Summary statistics")


