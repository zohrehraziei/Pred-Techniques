
# Zohreh Raziei - zohrehraziei@gmail.com

"""

      Predict : Regression Methods(using scikit-learn package)
      A prediction task for rental home company 
      Author: Zohreh Raziei - raziei.z@husky.neu.edu

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import fbeta_score, make_scorer


# Custom Loss function to evaluate the accuracy
def custom_loss_func_1(y_true, y_pred):
     diff = np.abs(y_true - y_pred).mean()
     return np.log1p(diff)
 
# Custom Loss function to evaluate the accuracy
def custom_loss_func_2(y_true, y_pred):
     diff = np.median(np.abs(y_true - y_pred), axis=0)
     return np.log1p(diff)

# Import dataset
data_train = pd.read_csv('Data Science.csv')
data_test  = pd.read_csv('Data Science.csv')

# separate dependent and independent vars
# and building the test and train
X_tr = data_train.loc[:, ~data_train.columns.isin(['SaleDollarCnt','TransDate','ZoneCodeCounty']) ].values
Y_tr = data_train['SaleDollarCnt'].values
#X_ts for the final prediction
X_ts = data_test.loc[:, ~data_test.columns.isin(['SaleDollarCnt','TransDate','ZoneCodeCounty'])].values
Xtr_len = len(X_tr)
Xts_len = len(X_ts)


# missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_tr[:, :])
X_tr[:, :] = imputer.transform(X_tr[:, :])
imputer = imputer.fit(X_ts[:, :])
X_ts[:, :] = imputer.transform(X_ts[:, :])


# Splitting Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_tr, Y_tr, test_size = 0.2, random_state = 0)


# Fit Linear Regression to x_train
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

# Predicting 
y_pred = lin_reg.predict(x_test)

print('First Way: Linear Regression: ')
score = make_scorer(custom_loss_func_1, greater_is_better=False)
print('Loss function 1 output: ', custom_loss_func_1(y_pred, y_test))
print('score: ', score(lin_reg, x_test, y_test))

score = make_scorer(custom_loss_func_2, greater_is_better=False)
print('Loss function 2 output: ', custom_loss_func_2(y_pred, y_test))
print('score: ', score(lin_reg, x_test, y_test))
print('--------------------------------------------------------')


# Fit Decision Tree Regression to x_train
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x_train, y_train)

# Predicting a new result
y_pred = regressor.predict(x_test)

print('Second Way: Decision Tree Regression: ')
score = make_scorer(custom_loss_func_1, greater_is_better=False)
print('Loss function 1 output: ', custom_loss_func_1(y_pred, y_test))
print('score: ', score(regressor, x_test, y_test))

score = make_scorer(custom_loss_func_2, greater_is_better=False)
print('Loss function 2 output: ', custom_loss_func_2(y_pred, y_test))
print('score: ', score(regressor, x_test, y_test))
print('--------------------------------------------------------')


# Fit Random Forest Regression to the x train
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(x_train, y_train)


# Predicting 
y_pred = regressor.predict(x_test)

print('Third Way: Random Forest Regression: ')
score = make_scorer(custom_loss_func_1, greater_is_better=False)
print('Loss function 1 output: ', custom_loss_func_1(y_pred, y_test))
print('score: ', score(regressor, x_test, y_test))

score = make_scorer(custom_loss_func_2, greater_is_better=False)
print('Loss function 2 output: ', custom_loss_func_2(y_pred, y_test))
print('score: ', score(regressor, x_test, y_test))
print('--------------------------------------------------------')


#####       Final Prediction:
 
y_pred_final = regressor.predict(X_ts)


#CSV results
from pandas import DataFrame

column1 = 'PropertyID'
column2 = 'SaleDollarCnt'

dic = dict(zip([column1,column2],[data_test['PropertyID'].values,y_pred_final.tolist()]))
print(dic)
df = DataFrame(dic)
export_csv = df.to_csv ('Resutls_Z.csv', columns = [column1,column2])


