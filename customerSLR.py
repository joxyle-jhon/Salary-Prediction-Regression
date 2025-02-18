import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

# Load dataset
df = pd.read_csv('Telco-Customer-Churn.csv')

# Convert 'TotalCharges' to numeric, forcing errors to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Handle missing values
df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

# Define features and target
X = df[['TotalCharges']]
y = df['MonthlyCharges']  # Use a continuous variable instead of churn

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate model performance metrics
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Absolute Percentage Error:", mape)
print("Mean Squared Error:", mse)

# Visualization
plt.figure(figsize=(8, 5))
sns.set_theme(style="whitegrid")
sns.scatterplot(x=X_test['TotalCharges'], y=y_test, label="Actual", color="blue", alpha=0.6)
sns.lineplot(x=X_test['TotalCharges'], y=y_pred, label="Regression Line", color="red")
plt.xlabel("Total Charges")
plt.ylabel("Monthly Charges")
plt.title("Linear Regression: Total Charges vs Monthly Charges")
plt.legend()
plt.show()
