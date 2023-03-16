# Standart data analysis libraries
import numpy as np
import pandas as pd
# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
# Counter
from collections import Counter
# Models
from xgboost import XGBRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor

#Scale data
from sklearn.preprocessing import StandardScaler

# For model selections
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error

df = pd.read_csv('internship_train.csv')

assert df.shape[1] == 54
df = df.drop('8', axis = 1)

most_cor = ['7', '26', '39', '31', '25', '28', '10', '44', '17', '21']

X = df.drop('target', axis = 1)
X = X[most_cor]
y = df.target

scaler_x = StandardScaler()
scaler_y = StandardScaler()

X = scaler_x.fit_transform(pd.DataFrame(X))
y = scaler_y.fit_transform(pd.DataFrame(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle = True)
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size = 0.5, random_state = 0, shuffle = True)
y_train = y_train.ravel()

model = XGBRegressor(max_depth = 3, learning_rate = 0.1)

model.fit(X_train, y_train)

data_test = pd.read_csv('internship_hidden_test.csv')

data_test = data_test[most_cor]

solution = scaler_y.inverse_transform(model.predict(data_test).reshape(-1,1))

pd.DataFrame(solution).to_csv('solution.csv')
