import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

# 1. Dummy Data: [Age, Experience, Education_Level] -> Target: Salary
data = {
    'Age': [25, 30, 35, 40, 45, 50, 55, 60, 28, 32],
    'Experience': [2, 5, 8, 12, 15, 20, 25, 30, 3, 6],
    'Education_Level': [1, 2, 2, 3, 3, 2, 3, 4, 1, 2], # 1:Bachelors, 2:Masters, 3:PhD
    'Salary': [45000, 60000, 75000, 110000, 130000, 125000, 150000, 180000, 50000, 65000]
}
df = pd.DataFrame(data)

# 2. Assign Features (X) and Target (y)
X = df[['Age', 'Experience', 'Education_Level']]
y = df['Salary']

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# 4. Build and Train the Model
mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train)

# 5. Evaluate
y_pred = mlr_model.predict(X_test)
print("Intercept:", mlr_model.intercept_)
print("Coefficients (Age, Exp, Edu):", mlr_model.coef_)
print("R-Squared Score on Test Data:", r2_score(y_test, y_pred))

# 6. Test with a custom input
custom_pred = mlr_model.predict([[35, 7, 2]])
print("\nPredicted Salary for [Age 35, 7 yrs Exp, Masters (Level 2)]: $", round(custom_pred[0], 2))

# 7. Plotting Multiple Linear Regression Results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Multiple Linear Regression Analysis', fontsize=16, fontweight='bold')

# Plot 1: Actual vs Predicted Values
axes[0, 0].scatter(y_test, y_pred, color='blue', alpha=0.7, edgecolors='k')
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Salary ($)', fontsize=10)
axes[0, 0].set_ylabel('Predicted Salary ($)', fontsize=10)
axes[0, 0].set_title('Actual vs Predicted Salary', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Residuals Plot
residuals = y_test - y_pred
axes[0, 1].scatter(y_pred, residuals, color='green', alpha=0.7, edgecolors='k')
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Predicted Salary ($)', fontsize=10)
axes[0, 1].set_ylabel('Residuals ($)', fontsize=10)
axes[0, 1].set_title('Residuals Plot', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Feature Coefficients
features = ['Age', 'Experience', 'Education_Level']
coefficients = mlr_model.coef_
colors = ['green' if c > 0 else 'red' for c in coefficients]
axes[1, 0].bar(features, coefficients, color=colors, alpha=0.7, edgecolor='k')
axes[1, 0].set_ylabel('Coefficient Value', fontsize=10)
axes[1, 0].set_title('Feature Coefficients (Impact on Salary)', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Distribution of Residuals
axes[1, 1].hist(residuals, bins=5, color='purple', alpha=0.7, edgecolor='k')
axes[1, 1].set_xlabel('Residuals ($)', fontsize=10)
axes[1, 1].set_ylabel('Frequency', fontsize=10)
axes[1, 1].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()