# Standart data analysis libraries
import numpy as np
import pandas as pd

# Models
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error as RMSE

# read data
df = pd.read_csv('internship_train.csv')

# make a numpy array with feature that we need for modeling and target variable
x = df['6'].values
target = df['target'].values

# Modeling
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(x.reshape(-1, 1))
poly_reg_model = LinearRegression()
model = poly_reg_model.fit(poly_features, target)

# Prediction
test = pd.read_csv('internship_hidden_test.csv')
poly_test_feature = poly.fit_transform(test['6'].values.reshape(-1,1))
prediction = model.predict(poly_test_feature)
pd.DataFrame({'prediction':prediction}).to_csv('Solution_task 3.csv')
