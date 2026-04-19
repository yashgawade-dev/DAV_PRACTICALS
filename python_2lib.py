import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set up a wider figure to show off two plots clearly
plt.figure(figsize=(12, 5))

# --- Experiment 8a: Matplotlib (Multiple Lines) ---
plt.subplot(1, 2, 1) # 1 row, 2 cols, 1st plot
x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x), label='Sine Wave', color='blue', linestyle='-')
plt.plot(x, np.cos(x), label='Cosine Wave', color='orange', linestyle='--')
plt.title('Matplotlib: Trigonometric Functions')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.legend()
plt.grid(True)

# --- Experiment 8b: Seaborn (Boxplot) ---
plt.subplot(1, 2, 2) # 1 row, 2 cols, 2nd plot
# Using Seaborn's internal 'iris' dataset
iris = sns.load_dataset("iris")
sns.boxplot(x="species", y="sepal_length", data=iris, palette="Set2")
plt.title('Seaborn: Iris Dataset Boxplot')
plt.xlabel('Species')
plt.ylabel('Sepal Length (cm)')

# Show both plots together
plt.tight_layout()
plt.show()