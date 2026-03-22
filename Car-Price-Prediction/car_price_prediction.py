import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

df = pd.read_csv(r"C:\Users\acer\Downloads\car.csv")
if 'Car_Name' in df.columns:
    df.drop(['Car_Name'], axis=1, inplace=True)

df['Years_Old'] = 2026 - df['Year']
df.drop(['Year'], axis=1, inplace=True)

le_fuel = LabelEncoder()
df['Fuel_Type'] = le_fuel.fit_transform(df['Fuel_Type'])
le_seller = LabelEncoder()
df['Seller_Type'] = le_seller.fit_transform(df['Seller_Type'])
le_trans = LabelEncoder()
df['Transmission'] = le_trans.fit_transform(df['Transmission'])

X = df[['Present_Price', 'Kms_Driven', 'Owner', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Years_Old']]
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

joblib.dump(rf, "car_price_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_fuel, "le_fuel.pkl")
joblib.dump(le_seller, "le_seller.pkl")
joblib.dump(le_trans, "le_trans.pkl")

df['Predicted_Selling_Price'] = rf.predict(X)

y_pred = rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"\nModel Evaluation:\nRMSE: {rmse:.2f} lakhs\nR2 Score: {r2:.2f}\n")

plt.figure(figsize=(8,5))
importance = rf.feature_importances_
feature_names = X.columns
sns.barplot(x=importance, y=feature_names)
plt.title("Feature Importance")
plt.show()

plt.figure(figsize=(8,6))
numeric_df = df.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

print("\nPredicted Selling Prices for first 10 cars in dataset:")
print(df[['Present_Price','Kms_Driven','Owner','Fuel_Type','Seller_Type','Transmission','Years_Old','Selling_Price','Predicted_Selling_Price']].head(10))

def predict_new_car():
    present_price = float(input("Showroom Price (in lakhs): "))
    kms_driven = float(input("Kilometers Driven: "))
    owner = int(input("Number of Previous Owners: "))
    fuel_input = input("Fuel Type (Petrol/Diesel/CNG): ").capitalize()
    fuel_encoded = le_fuel.transform([fuel_input])[0]
    seller_input = input("Seller Type (Dealer/Individual): ").capitalize()
    seller_encoded = le_seller.transform([seller_input])[0]
    trans_input = input("Transmission Type (Manual/Automatic): ").capitalize()
    trans_encoded = le_trans.transform([trans_input])[0]
    years_old = int(input("Years of Service: "))
    user_data = [present_price, kms_driven, owner, fuel_encoded, seller_encoded, trans_encoded, years_old]
    predicted_price = rf.predict([user_data])[0]
    print(f"\nApproximate Selling Price: {predicted_price:.2f} lakhs\n")

# predict_new_car()