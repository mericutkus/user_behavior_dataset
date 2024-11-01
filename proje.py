import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load the dataset
data = pd.read_csv("C:\\Users\\meric\\Downloads\\archive (11)\\user_behavior_dataset.csv")

# 2. Display basic information about the dataset
print("Dataset Info:")
print(data.info())
print("\nDataset Head:")
print(data.head())
print("\nDataset Description:")
print(data.describe())

# Check for missing values in the dataset
missing_values = data.isnull().sum()
print("Missing values in each column:\n", missing_values)

# 3. Set up the plotting style
plt.style.use("seaborn-v0_8-dark-palette")
plt.figure(figsize=(16, 20))

# 4. List of numerical columns (ensure these columns exist in your dataset)
numeric_columns = [
    "App Usage Time (min/day)", "Screen On Time (hours/day)", "Battery Drain (mAh/day)",
    "Number of Apps Installed", "Data Usage (MB/day)", "Age", "User Behavior Class"
]

# 5. Loop through each numerical column to plot histograms
for i, col in enumerate(numeric_columns, start=1):
    if col in data.columns:
        plt.subplot(4, 2, i)
        sns.histplot(data[col], kde=True, bins=30, color="skyblue")
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)

plt.tight_layout()
plt.show()

# 6. Plot boxplots for each numerical column to visualize potential outliers
plt.figure(figsize=(16, 20))

# 7. Loop through each numerical column to plot boxplots
for i, col in enumerate(numeric_columns, start=1):
    if col in data.columns:
        plt.subplot(4, 2, i)
        sns.boxplot(x=data[col], color="lightcoral")
        plt.title(f"Boxplot of {col}")
        plt.xlabel(col)

plt.tight_layout()
plt.show()

# 8. Plot a correlation heatmap for numerical features
plt.figure(figsize=(12, 10))
correlation_matrix = data[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", square=True)
plt.title("Correlation Heatmap")
plt.show()

# 9. Define the target variable and independent variable
X = data[['Battery Drain (mAh/day)']]  # Independent variable
y = data['Data Usage (MB/day)']  # Target variable

# 10 Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 11 Create the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 12 Make predictions
y_pred = model.predict(X_test)

# 13 Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Values')
plt.scatter(X_test, y_pred, color='red', label='Predicted Values')
plt.plot(X_test, y_pred, color='red')
plt.title('Actual vs Predicted Values')
plt.xlabel('Battery Drain (mAh/day)')
plt.ylabel('Data Usage (MB/day)')
plt.legend()
plt.show()

# 14 Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)