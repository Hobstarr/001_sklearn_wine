import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
sns.set_style('whitegrid')

import joblib

# Download data.
dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep = ';')

# Give some information about each column using classical statistics. 
for entry in data.columns:
    data[entry].describe()

# Created a data / labels set.
y = data.pop('quality')
x = data.copy()

# Create a train/test split that keeps label distribution, with a size train: 0.8, test 0.2. 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify = y)

# Scale data to a zero mean and standard deviation of one.
scaler = preprocessing.StandardScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Create a pipeline that includes a scaler, and a RF regressor.
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators = 100, random_state=123))

# Print a simple list all parameters in the pipeline.
for entry in pipeline.get_params().keys():
    print(entry)

# Declare the hyper parameters we are going to change through cross validation to determine the best model.
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}

clf = GridSearchCV(pipeline, hyperparameters, cv = 10)

clf.fit(x_train, y_train)

print(clf.best_params_)
print(clf.refit)

y_preds = clf.predict(x_test)

print(r2_score(y_test, y_preds))
print(mean_squared_error(y_test, y_preds))

sns.histplot(y_test - y_preds, bins = 50, label = 'true_rating - predicted_rating')
plt.title('Difference between predcition results and true results for wine quality')
plt.axvline(x = 0.0, color = 'r', ls = ':')
plt.legend()
plt.show()