
# Zohreh Raziei- zohrehraziei@gmail.com

"""

      Predict 1 : Logistic Regression (using scikit-learn package)
      Author: Zohreh Raziei - raziei.z@husky.neu.edu

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
data_train = pd.read_csv('Code_train.csv')
data_test  = pd.read_csv('Code_test.csv')

# clear some specific columns
data_train['x41'] = data_train['x41'].str.replace('$', '')
data_train['x41'] = data_train['x41'].astype(float)
data_train['x45'] = data_train['x45'].str.replace('%', '')
data_train['x45'] = data_train['x45'].astype(float)
data_train['x35'] = data_train['x35'].str.replace('wednesday', 'wed')
data_train['x35'] = data_train['x35'].str.replace('thurday', 'thur')
data_test['x41'] = data_test['x41'].str.replace('$', '')
data_test['x41'] = data_test['x41'].astype(float)
data_test['x45'] = data_test['x45'].str.replace('%', '')
data_test['x45'] = data_test['x45'].astype(float)
data_test['x35'] = data_test['x35'].str.replace('wednesday', 'wed')
data_test['x35'] = data_test['x35'].str.replace('thurday', 'thur')


# separate dependent and independent vars
X_tr = data_train.iloc[:, :-1].values
Y_tr = data_train.iloc[:, -1].values
X_ts = data_test.iloc[:, :].values
Xtr_len = len(X_tr)
Xts_len = len(X_ts)

# Encoding categorical data: LabelEncoder
# [34, 35, 68, 93]
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X_tr[:, 34] = labelencoder_X.fit_transform(X_tr[:, 34].astype(str))
X_tr[:, 35] = labelencoder_X.fit_transform(X_tr[:, 35].astype(str))
X_tr[:, 68] = labelencoder_X.fit_transform(X_tr[:, 68].astype(str))
X_tr[:, 93] = labelencoder_X.fit_transform(X_tr[:, 93].astype(str))
X_ts[:, 34] = labelencoder_X.fit_transform(X_ts[:, 34].astype(str))
X_ts[:, 35] = labelencoder_X.fit_transform(X_ts[:, 35].astype(str))
X_ts[:, 68] = labelencoder_X.fit_transform(X_ts[:, 68].astype(str))
X_ts[:, 93] = labelencoder_X.fit_transform(X_ts[:, 93].astype(str))

# missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_tr[:, :])
X_tr[:, :] = imputer.transform(X_tr[:, :])
imputer = imputer.fit(X_ts[:, :])
X_ts[:, :] = imputer.transform(X_ts[:, :])

# Encoding categorical data: OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
a = np.concatenate((X_tr, X_ts))
onehotencoder = OneHotEncoder(categorical_features = [34,35,68,93],sparse=True)
a = onehotencoder.fit_transform(a).toarray()
X_tr = a[:len(X_tr),:]
X_ts = a[len(X_tr):,:]


# Splitting Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_tr, Y_tr, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
X_ts = sc.fit_transform(X_ts)


# Fit the Logistic Regression to the x train
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

# Predicting 
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Probability estimate of all classes(two class) for the test data:
y_pred_prob = classifier.predict_proba(X_ts)



#CSV results
from pandas import DataFrame

column = ['Probability of belonging to class 1 - Logistic Regression']

dic = dict(zip(column,[y_pred_prob[:,1].tolist()]))

df = DataFrame(dic)
export_csv = df.to_csv ('LogisticRegression.csv', columns = column) 


