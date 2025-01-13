import sklearn
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection
import joblib
import numpy as np
import pickle

print(joblib.__version__)

X_train = pd.read_csv('data/processed_data/X_train_scaled.csv')
X_test = pd.read_csv('data/processed_data/X_test_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv')
y_test = pd.read_csv('data/processed_data/y_test.csv')
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

regr = RandomForestRegressor()

parameters = {'max_depth': np.arange(1,8,1)}

#--do grid search
grid_clf = model_selection.GridSearchCV(estimator=regr, param_grid=parameters)
grid_clf.fit(X_train, y_train)

#--Save the trained model to a file
model_filename = './models/gridsearch_rf.pkl'
pickle.dump(grid_clf, open(model_filename, 'wb'))
print("Model trained and saved successfully.")