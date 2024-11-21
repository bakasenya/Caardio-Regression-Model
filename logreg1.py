import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import datetime
import math

# Reading the data into pd
data = pd.read_csv("cardio_train.csv", sep=";", encoding="utf-8")

# Printing Dimesnions and columns
print(f"Dimensions of data frame: {data.shape}")
print(f"Column names of data frame: {data.columns.to_frame()}")
print("\n")
# Extracting age in days
age_values = data["age"].values
print(f"Age values: {age_values}")
print("\n")
# Calculating age in years
age_year = data["age"] / 365.25
current_year = datetime.datetime.now().year
year_of_birth = current_year - age_year
print(f"Year of birth: {year_of_birth.astype(int)}")
print("\n")
# Calculating mean of the age in years
mean_age_year = age_year.mean()
max_age = age_year.max()
min_age = age_year.min()

print("Mean age in years:", mean_age_year.round(2))
print("Maximum age in years:", math.ceil(max_age))
print("Minimum age in years:", math.ceil(min_age))
print("\n")
# Splitting data into training and test data
train_data, test_data = train_test_split(data, test_size=0.2, random_state=420)
print("Training data samples:", len(train_data))
print("Test data samples:", len(test_data))
print("\n")

# Scale the data using standard scaler
scaler = StandardScaler()

features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
            'cholesterol', 'gluc', 'smoke', 'alco', 'active']  
target = 'cardio'

train_data_scaled = train_data.copy()
test_data_scaled = test_data.copy()

train_data_scaled[features] = scaler.fit_transform(train_data[features])
test_data_scaled[features] = scaler.transform(test_data[features])

print(train_data_scaled, "\n")
print(test_data_scaled, "\n")

# Creating a Logistic Regression Model
x_train = train_data_scaled[features]
y_train = train_data[target]
x_test = test_data_scaled[features]
y_test = test_data[target]

model = LogisticRegression(random_state=420, max_iter=1000)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Compare the predicted output
comparison = pd.DataFrame({
    "Real Values": y_test,
    "Predicted Values": y_pred
})

print("Comparison between real and predicated values")
print(comparison, "\n")

# Calculating feature importance 
feature_importance = model.coef_[0]

importance_of_features = pd.DataFrame({
    "Features": features,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)

print("Feature Importance")
print(importance_of_features)