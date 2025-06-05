from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd

# Selecting Data for Modeling
housing_file_path = 'data/housing.csv'
housing_data = pd.read_csv(housing_file_path)

# Selecting The Prediction Target ( median_house_value )
y= housing_data.median_house_value

# Housing "Features"
housing_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                    'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity']
X = housing_data[housing_features] # X - Input features

# One-hot encoding for categorical variable
X = pd.get_dummies(X)

# Split the data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1)

# Model Definition (Random Forest Regressor)
housing_model = RandomForestRegressor(random_state=1)

# Model Fitting (no validation split)
housing_model.fit(X_train, y_train)

# Prediction Making
predictions = housing_model.predict(X_valid)
mae = mean_absolute_error(y_valid, predictions)

# Model Evaluation

print("Make predictions for the first 5 houses: ")
print(X.head())
print("The predictions are:")
print(housing_model.predict(X.head()))