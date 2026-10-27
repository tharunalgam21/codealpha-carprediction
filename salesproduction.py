# TASK 4: SALES PREDICTION USING PYTHON

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 2. Load Dataset
# Change the file name if your dataset has a different name
df = pd.read_csv("Advertising.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)


# 3. Data Cleaning

# Remove unnecessary index column if present
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicate rows
df.drop_duplicates(inplace=True)

print("\nDataset after cleaning:")
print(df.head())


# 4. Basic Data Analysis

print("\nStatistical Summary:")
print(df.describe())

print("\nData Types:")
print(df.dtypes)


# 5. Correlation Analysis

print("\nCorrelation Matrix:")
print(df.corr(numeric_only=True))

plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="Blues")
plt.title("Correlation Matrix")
plt.show()


# 6. Data Visualization

# Advertising Spend vs Sales

plt.figure(figsize=(7, 5))
plt.scatter(df["TV"], df["Sales"])
plt.xlabel("TV Advertising Spend")
plt.ylabel("Sales")
plt.title("TV Advertising vs Sales")
plt.show()


plt.figure(figsize=(7, 5))
plt.scatter(df["Radio"], df["Sales"])
plt.xlabel("Radio Advertising Spend")
plt.ylabel("Sales")
plt.title("Radio Advertising vs Sales")
plt.show()


plt.figure(figsize=(7, 5))
plt.scatter(df["Newspaper"], df["Sales"])
plt.xlabel("Newspaper Advertising Spend")
plt.ylabel("Sales")
plt.title("Newspaper Advertising vs Sales")
plt.show()


# 7. Select Features and Target

X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]


# 8. Split Data into Training and Testing

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# 9. Create Linear Regression Model

model = LinearRegression()

model.fit(X_train, y_train)


# 10. Predict Sales

y_pred = model.predict(X_test)

print("\nPredicted Sales:")
print(y_pred)


# 11. Model Evaluation

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("------------------------")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)


# 12. Compare Actual vs Predicted Sales

comparison = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": y_pred
})

print("\nActual vs Predicted Sales:")
print(comparison.head(20))


# 13. Plot Actual vs Predicted Sales

plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual Sales vs Predicted Sales")

plt.show()


# 14. Advertising Impact Analysis

print("\nAdvertising Impact on Sales")

print("TV coefficient:", model.coef_[0])
print("Radio coefficient:", model.coef_[1])
print("Newspaper coefficient:", model.coef_[2])


# 15. Predict Future Sales

# Example advertising budget
future_advertising = pd.DataFrame({
    "TV": [150],
    "Radio": [30],
    "Newspaper": [20]
})

future_sales = model.predict(future_advertising)

print("\nFuture Advertising Budget:")
print(future_advertising)

print("\nPredicted Future Sales:")
print(future_sales[0])


# 16. Feature Importance

importance = pd.DataFrame({
    "Advertising Platform": ["TV", "Radio", "Newspaper"],
    "Coefficient": model.coef_
})

print("\nAdvertising Impact:")
print(importance)


# 17. Plot Advertising Impact

plt.figure(figsize=(7, 5))

plt.bar(
    importance["Advertising Platform"],
    importance["Coefficient"]
)

plt.xlabel("Advertising Platform")
plt.ylabel("Impact on Sales")
plt.title("Advertising Platform Impact on Sales")

plt.show()


# 18. Final Result

print("\n================================")
print("SALES PREDICTION COMPLETED")
print("================================")

print("R2 Score:", round(r2, 4))
print("Predicted Future Sales:", round(future_sales[0], 2))

if model.coef_[0] > model.coef_[1] and model.coef_[0] > model.coef_[2]:
    print("TV advertising has the highest impact on sales.")
elif model.coef_[1] > model.coef_[2]:
    print("Radio advertising has the highest impact on sales.")
else:
    print("Newspaper advertising has the highest impact on sales.")
