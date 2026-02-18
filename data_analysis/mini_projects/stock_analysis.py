import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime,timedelta

print("="*50)
print("STOCK MARKET ANALYSIS")
print("="*50)

def generate_stock_data(ticker,days=365,start_price=100):
    """SYNTHETIC STOCK DATA"""
    dates=pd.date_range(end=datetime.now(),periods=days,freq='D')

    returns=np.random.randn(days)*2
    prices=start_price*(1+returns/100).cumprod()

    trend=np.linspace(0,20,days)
    prices=prices+trend

    volume=np.random.randint(100000,1000000,days)

    df = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.randn(days) * 0.01),
        'High': prices * (1 + np.abs(np.random.randn(days)) * 0.02),
        'Low': prices * (1 - np.abs(np.random.randn(days)) * 0.02),
        'Close': prices,
        'Volume': volume,
        'Ticker': ticker
    })
    
    return df

print("\nGENERATING STOCK DATA FOR POPULAR TECH COMPANIES...")
tickers=['AAPL','GOOGl','MSFT','NVDA','TSLA']
start_prices=[150,140,300,500,200]


all_stocks=[]
for ticker,start_price in zip(tickers,start_prices):
    stock_data=generate_stock_data(ticker,days=365,start_price=start_price)
    all_stocks.append(stock_data)
    print(f"GENERATED DATA FOR {ticker}")

df=pd.concat(all_stocks,ignore_index=True)
print(f"\nTOTAL RECORDS:{len(df)}")
print()

print("="*60)
print("DATA EXPLORATION")
print("="*60)

print("\nFirst 10 rows:")
print(df.head(10))
print()

print("Dataset info:")
df.info()
print()


# ANALYSIS 1: Current prices
print("="*60)
print("ANALYSIS 1: CURRENT STOCK PRICES")
print("="*60)

current_prices = df.groupby('Ticker').last()[['Close', 'Volume']]
current_prices = current_prices.sort_values('Close', ascending=False)
print("\nCurrent prices (latest):")
print(current_prices)
print()

# ANALYSIS 2: Calculate returns
print("="*60)
print("ANALYSIS 2: RETURNS ANALYSIS")
print("="*60)

returns_data = []

for ticker in tickers:
    stock = df[df['Ticker'] == ticker].copy()
    stock = stock.sort_values('Date')
    
    first_price = stock.iloc[0]['Close']
    last_price = stock.iloc[-1]['Close']
    
    total_return = ((last_price - first_price) / first_price) * 100
    
    returns_data.append({
        'Ticker': ticker,
        'Start_Price': first_price,
        'End_Price': last_price,
        'Total_Return_%': total_return
    })

returns_df = pd.DataFrame(returns_data)
returns_df = returns_df.sort_values('Total_Return_%', ascending=False)

print("\n1-Year Returns:")
print(returns_df)
print()

best_performer = returns_df.iloc[0]
print(f"Best performer: {best_performer['Ticker']} (+{best_performer['Total_Return_%']:.2f}%)")
print()

# ANALYSIS 3: Volatility (risk)
print("="*60)
print("ANALYSIS 3: VOLATILITY ANALYSIS")
print("="*60)

volatility_data = []

for ticker in tickers:
    stock = df[df['Ticker'] == ticker].copy()
    stock = stock.sort_values('Date')
    
    # Calculate daily returns
    stock['Daily_Return'] = stock['Close'].pct_change() * 100
    
    volatility = stock['Daily_Return'].std()
    
    volatility_data.append({
        'Ticker': ticker,
        'Volatility_%': volatility,
        'Risk_Level': 'High' if volatility > 3 else 'Medium' if volatility > 2 else 'Low'
    })

volatility_df = pd.DataFrame(volatility_data)
volatility_df = volatility_df.sort_values('Volatility_%', ascending=False)

print("\nVolatility (Risk) Analysis:")
print(volatility_df)
print()

# ANALYSIS 4: Moving averages
print("="*60)
print("ANALYSIS 4: MOVING AVERAGES")
print("="*60)


focus_stock = 'NVDA'
nvda = df[df['Ticker'] == focus_stock].copy()
nvda = nvda.sort_values('Date')

nvda['MA_50'] = nvda['Close'].rolling(window=50).mean()
nvda['MA_200'] = nvda['Close'].rolling(window=200).mean()

print(f"\n{focus_stock} Moving Averages:")
print(nvda[['Date', 'Close', 'MA_50', 'MA_200']].tail(10))
print()

#Golden cross / Death cross
latest = nvda.iloc[-1]
if latest['MA_50'] > latest['MA_200']:
    signal = "BULLISH (Golden Cross)"
else:
    signal = "BEARISH (Death Cross)"

print(f"{focus_stock} Signal: {signal}")
print(f"50-day MA: ${latest['MA_50']:.2f}")
print(f"200-day MA: ${latest['MA_200']:.2f}")
print()

# ANALYSIS 5: Trading volume trends
print("="*60)
print("ANALYSIS 5: VOLUME ANALYSIS")
print("="*60)

volume_summary = df.groupby('Ticker')['Volume'].agg(['mean', 'max', 'min'])
volume_summary = volume_summary.sort_values('mean', ascending=False)

print("\nAverage trading volume:")
print(volume_summary)
print()

print("="*60)
print("CREATING VISUALIZATIONS...")
print("="*60)

fig=plt.figure(figsize=(16,12))

# Plot 1: Price comparison
plt.subplot(3, 2, 1)
for ticker in tickers:
    stock = df[df['Ticker'] == ticker].sort_values('Date')
    # Normalize to percentage change
    normalized = (stock['Close'] / stock['Close'].iloc[0] - 1) * 100
    plt.plot(stock['Date'], normalized, label=ticker, linewidth=2)
plt.title('Stock Price Performance (Normalized %)', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Return %')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Plot 2: Current prices
plt.subplot(3, 2, 2)
current = df.groupby('Ticker').last()['Close'].sort_values()
plt.barh(current.index, current.values, color='steelblue')
plt.title('Current Stock Prices', fontsize=14, fontweight='bold')
plt.xlabel('Price ($)')
plt.grid(True, alpha=0.3, axis='x')

# Plot 3: Returns
plt.subplot(3, 2, 3)
colors = ['green' if x > 0 else 'red' for x in returns_df['Total_Return_%']]
plt.bar(returns_df['Ticker'], returns_df['Total_Return_%'], color=colors, alpha=0.7)
plt.title('1-Year Total Returns', fontsize=14, fontweight='bold')
plt.ylabel('Return %')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.3, axis='y')

# Plot 4: Volatility
plt.subplot(3, 2, 4)
plt.bar(volatility_df['Ticker'], volatility_df['Volatility_%'], color='orange', alpha=0.7)
plt.title('Volatility (Risk) Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Volatility %')
plt.grid(True, alpha=0.3, axis='y')

# Plot 5: NVDA with moving averages
plt.subplot(3, 2, 5)
plt.plot(nvda['Date'], nvda['Close'], label='Close Price', linewidth=2)
plt.plot(nvda['Date'], nvda['MA_50'], label='50-day MA', linewidth=1.5, linestyle='--')
plt.plot(nvda['Date'], nvda['MA_200'], label='200-day MA', linewidth=1.5, linestyle='--')
plt.title(f'{focus_stock} Price with Moving Averages', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Plot 6: Volume comparison
plt.subplot(3, 2, 6)
avg_volume = df.groupby('Ticker')['Volume'].mean().sort_values()
plt.barh(avg_volume.index, avg_volume.values / 1_000_000, color='purple', alpha=0.7)
plt.title('Average Trading Volume', fontsize=14, fontweight='bold')
plt.xlabel('Volume (Millions)')
plt.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('stock_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved visualization as 'stock_analysis.png'")
plt.show()

print("\n" + "="*60)
print("SAVING RESULTS...")
print("="*60)

returns_df.to_csv('stock_returns.csv', index=False)
print("✓ Saved stock_returns.csv")

volatility_df.to_csv('stock_volatility.csv', index=False)
print("✓ Saved stock_volatility.csv")

nvda.to_csv('nvda_detailed.csv', index=False)
print("✓ Saved nvda_detailed.csv")

#Creating Summary Report
summary_report = pd.DataFrame({
    'Metric': [
        'Best Performer',
        'Worst Performer',
        'Most Volatile',
        'Least Volatile',
        'Highest Price',
        'Lowest Price'
    ],
    'Value': [
        f"{returns_df.iloc[0]['Ticker']} (+{returns_df.iloc[0]['Total_Return_%']:.2f}%)",
        f"{returns_df.iloc[-1]['Ticker']} ({returns_df.iloc[-1]['Total_Return_%']:.2f}%)",
        f"{volatility_df.iloc[0]['Ticker']} ({volatility_df.iloc[0]['Volatility_%']:.2f}%)",
        f"{volatility_df.iloc[-1]['Ticker']} ({volatility_df.iloc[-1]['Volatility_%']:.2f}%)",
        f"{current_prices.index[0]} (${current_prices.iloc[0]['Close']:.2f})",
        f"{current_prices.index[-1]} (${current_prices.iloc[-1]['Close']:.2f})"
    ]
})

summary_report.to_csv('stock_summary.csv', index=False)
print("✓ Saved stock_summary.csv")

print("\n" + "="*60)
print("✅ STOCK ANALYSIS COMPLETE!")
print("="*60)
print("\nKey Insights:")
print(f"• Best performer: {returns_df.iloc[0]['Ticker']} with {returns_df.iloc[0]['Total_Return_%']:.2f}% return")
print(f"• Most volatile: {volatility_df.iloc[0]['Ticker']} (higher risk, potentially higher reward)")
print(f"• {focus_stock} is currently {signal}")
print("\nFiles created:")
print("1. stock_analysis.png - Comprehensive visualization")
print("2. stock_returns.csv - Returns data")
print("3. stock_volatility.csv - Risk analysis")
print("4. nvda_detailed.csv - Detailed NVIDIA data")
print("5. stock_summary.csv - Summary report")