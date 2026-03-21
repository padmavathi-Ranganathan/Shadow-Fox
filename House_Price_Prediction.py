
# import pandas as pd

# df = pd.read_csv("C:/Users/acer/Downloads/HousingData/HousingData.csv")
# print(df.head())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("C:/Users/acer/Downloads/HousingData/HousingData.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())

df.replace('NA', np.nan, inplace=True)
df = df.astype(float)
df = df.fillna(df.mean())

X = df.drop("MEDV", axis=1)
y = df["MEDV"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)
print("Accuracy Percentage:", r2 * 100, "%")

result = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})
print(result.head())

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()