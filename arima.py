import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore') # Suppresses annoying statsmodels warnings in IDLE

# 1. Generate some sine wave-like historical data with noise
np.random.seed(42)
time = np.arange(50)
data = 20 + 5 * np.sin(time / 3) + np.random.randn(50)

# 2. Fit ARIMA Model (Order: p=2, d=1, q=2)
model = ARIMA(data, order=(2, 1, 2))
model_fit = model.fit()

print("--- ARIMA Model Summary ---")
print(model_fit.summary().tables[1]) # Print the coefficient table

# 3. Forecast the next 10 steps
forecast_steps = 10
forecast = model_fit.forecast(steps=forecast_steps)

# 4. Plot original data and forecast
plt.figure(figsize=(9, 4))
plt.plot(time, data, label='Historical Data', marker='o')

future_time = np.arange(50, 50 + forecast_steps)
plt.plot(future_time, forecast, color='red', label='ARIMA Forecast', marker='x', linestyle='--')

plt.title('ARIMA Time Series Forecasting')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()