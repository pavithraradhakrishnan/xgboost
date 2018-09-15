import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import precision_score, mean_squared_error

from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib
from numpy import loadtxt
import xgboost as xgb
from sklearn.datasets import dump_svmlight_file
# Load the data set
from xgboost import XGBClassifier

df = pd.read_csv("ml_house_data_set.csv")

# Remove the fields from the data set that we don't want to include in our model
del df['house_number']
del df['unit_number']
del df['street_name']
del df['zip_code']

# Replace categorical data with one-hot encoded data
features_df = pd.get_dummies(df, columns=['garage_type', 'city'])

# Remove the sale price from the feature data
del features_df['sale_price']


#print(df['sale_price'])
# Create the X and y arrays
X = features_df.values
y = df['sale_price'].values

# Split the data set in a training set (70%) and a test set (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# use DMatrix for xgbosot
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)


# set xgboost params
params = {}
# the number of classes that exist in this datset
num_round = 20  # the number of training iterations

#------------- numpy array ------------------
# training and testing - numpy matrices
xg_reg = xgb.XGBRegressor(learning_rate = 0.1,
                max_depth = 6, n_estimators = 100000,min_samples_leaf=9,
    max_features=0.1,
    loss='huber',random_state=0)
#xg_reg = XGBClassifier()
xg_reg.fit(X_train,y_train)
joblib.dump(xg_reg, 'trained_house_classifier_model_xgboost1.pkl')
preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse)) 
model = joblib.load('trained_house_classifier_model_xgboost1.pkl')
house_to_value = [
    # House features
    1980,  # year_built
    1,  # stories
    3,  # num_bedrooms
    2,  # full_bathrooms
    0,  # half_bathrooms
    1401,  # livable_sqft
    1406,  # total_sqft
    396,  # garage_sqft
    0,  # carport_sqft
    False,  # has_fireplace
    False,  # has_pool
    True,  # has_central_heating
    True,  # has_central_cooling

    # Garage type: Choose only one
    1,  # attached
    0,  # detached
    0,  # none

    # City: Choose only one
    0,  # Amystad
    0,  # Brownport
    0,  # Chadstad
    0,  # Clarkberg
    0,  # Coletown
    0,  # Davidfort
    0,  # Davidtown
    0,  # East Amychester
    0,  # East Janiceville
    0,  # East Justin
    0,  # East Lucas
    0,  # Fosterberg
    0,  # Hallfort
    0,  # Jeffreyhaven
    0,  # Jenniferberg
    0,  # Joshuafurt
    0,  # Julieberg
    0,  # Justinport
    0,  # Lake Carolyn
    0,  # Lake Christinaport
    0,  # Lake Dariusborough
    1,  # Lake Jack
    0,  # Lake Jennifer
    0,  # Leahview
    0,  # Lewishaven
    0,  # Martinezfort
    0,  # Morrisport
    0,  # New Michele
    0,  # New Robinton
    0,  # North Erinville
    0,  # Port Adamtown
    0,  # Port Andrealand
    0,  # Port Daniel
    0,  # Port Jonathanborough
    0,  # Richardport
    0,  # Rickytown
    0,  # Scottberg
    0,  # South Anthony
    0,  # South Stevenfurt
    0,  # Toddshire
    0,  # Wendybury
    0,  # West Ann
    0,  # West Brittanyview
    0,  # West Gerald
    0,  # West Gregoryview
    0,  # West Lydia
    0  # West Terrence
]

# scikit-learn assumes you want to predict the values for lots of houses at once, so it expects an array.
# We just want to look at a single house, so it will be the only item in our array.
homes_to_value = [
    house_to_value
]

# Run the model and make a prediction for each house in the homes_to_value array
predicted_home_values = model.predict(homes_to_value)

# Since we are only predicting the price of one house, just look at the first prediction returned
predicted_value = predicted_home_values[0]

print("This house has an estimated value of ${:,.2f}".format(predicted_value))
feature_labels = np.array(['year_built', 'stories', 'num_bedrooms', 'full_bathrooms', 'half_bathrooms', 'livable_sqft', 'total_sqft', 'garage_sqft', 'carport_sqft', 'has_fireplace', 'has_pool', 'has_central_heating', 'has_central_cooling', 'garage_type_attached', 'garage_type_detached', 'garage_type_none', 'city_Amystad', 'city_Brownport', 'city_Chadstad', 'city_Clarkberg', 'city_Coletown', 'city_Davidfort', 'city_Davidtown', 'city_East Amychester', 'city_East Janiceville', 'city_East Justin', 'city_East Lucas', 'city_Fosterberg', 'city_Hallfort', 'city_Jeffreyhaven', 'city_Jenniferberg', 'city_Joshuafurt', 'city_Julieberg', 'city_Justinport', 'city_Lake Carolyn', 'city_Lake Christinaport', 'city_Lake Dariusborough', 'city_Lake Jack', 'city_Lake Jennifer', 'city_Leahview', 'city_Lewishaven', 'city_Martinezfort', 'city_Morrisport', 'city_New Michele', 'city_New Robinton', 'city_North Erinville', 'city_Port Adamtown', 'city_Port Andrealand', 'city_Port Daniel', 'city_Port Jonathanborough', 'city_Richardport', 'city_Rickytown', 'city_Scottberg', 'city_South Anthony', 'city_South Stevenfurt', 'city_Toddshire', 'city_Wendybury', 'city_West Ann', 'city_West Brittanyview', 'city_West Gerald', 'city_West Gregoryview', 'city_West Lydia', 'city_West Terrence'])

importance = xg_reg.feature_importances_

# Sort the feature labels based on the feature importance rankings from the model
feauture_indexes_by_importance = importance.argsort()

# Print each feature label, from most important to least important (reverse order)
for index in feauture_indexes_by_importance:
    print("{} - {:.2f}%".format(feature_labels[index], (importance[index] * 100.0)))


mse = mean_absolute_error(y_train, xg_reg.predict(X_train))
print("Training Set Mean Absolute Error: %.4f" % mse)

# Find the error rate on the test set
mse = mean_absolute_error(y_test, xg_reg.predict(X_test))
print("Test Set Mean Absolute Error: %.4f" % mse)


