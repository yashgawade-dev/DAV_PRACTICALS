import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# 1. Create 100 days of random walk data (simulating a stock price)
np.random.seed(0)
dates = pd.date_range(start='2024-01-01', periods=100)
price_changes = np.random.randn(100)
stock_price = 100 + np.cumsum(price_changes) # Cumulative sum to create a trend

df = pd.DataFrame({'Price': stock_price}, index=dates)

# 2. Calculate Rolling Statistics (Moving Averages)
df['7-Day MA'] = df['Price'].rolling(window=7).mean()
df['30-Day MA'] = df['Price'].rolling(window=30).mean()

# 3. Visualization - Two Subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle('Time Series Analysis', fontsize=16, fontweight='bold')

# Plot 1: Normal Time Series Plot
axes[0].plot(df.index, df['Price'], label='Original Daily Price', color='blue', alpha=0.7, linewidth=2)
axes[0].plot(df.index, df['7-Day MA'], color='orange', label='7-Day Moving Average', linewidth=2)
axes[0].plot(df.index, df['30-Day MA'], color='red', label='30-Day Moving Average', linewidth=2)
axes[0].set_title('Time Series: Simulated Stock Price with Moving Averages', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Date', fontsize=10)
axes[0].set_ylabel('Price ($)', fontsize=10)
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)

# Plot 2: Autocorrelation Function (ACF)
plot_acf(df['Price'].dropna(), lags=40, ax=axes[1], title='Autocorrelation Function (ACF) - Stock Price')
axes[1].set_xlabel('Lag', fontsize=10)
axes[1].set_ylabel('ACF', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()