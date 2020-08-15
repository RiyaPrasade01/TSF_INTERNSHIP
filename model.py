import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv("http://bit.ly/w-data")
print("Shape of dataset: {}".format(data.shape))

data.head(10)

data.isnull().sum()

data.info()

sns.scatterplot(data['Hours'], data['Scores'])

x = np.array(data['Hours']).reshape(-1,1)
y = np.array(data['Scores']).reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3)
print("x_train size: ", x_train.shape)
print("x_test size: ", x_test.shape)
print("y_train size: ", y_train.shape)
print("x_test size:", y_test.shape)

lr = LinearRegression()
lr.fit(x_train, y_train)

y_hat1 = lr.predict(x_test)

print("mean squared error: ", mean_squared_error(y_test, y_hat1))
plt.scatter(y_test, x_test, label = "Actual Value")
plt.scatter(y_hat1, x_test, label = "Predicted Value")
plt.xlabel('Score')
plt.ylabel('Hours')
plt.legend()

parameters= [{'alpha': [0.001,0.1,1, 10, 100, 1000, 10000, 100000, 100000]}]
rr=Ridge()
grid = GridSearchCV(rr, parameters,cv=4)


grid.fit(x_train, y_train)
best_model = grid.best_estimator_
best_model


y_hat2 = best_model.predict(x_test)


print("mean squared error: ", mean_squared_error(y_test, y_hat2))
plt.scatter(y_test, x_test, label = "Actual Value")
plt.scatter(y_hat2, x_test, label = "Predicted Value")
plt.xlabel('Score')
plt.ylabel('Hours')
plt.legend()


xgbr = xgb.XGBRegressor()
xgbr.fit(x_train,y_train)


y_hat3 = xgbr.predict(x_test)


print("mean squared error: ", mean_squared_error(y_test, y_hat3))
plt.scatter(y_test, x_test, label = "Actual Value")
plt.scatter(y_hat3, x_test, label = "Predicted Value")
plt.xlabel('Score')
plt.ylabel('Hours')
plt.legend()

pd.DataFrame({"Model":['sklearn,LinearRigression', 'Ridge Regressior','Gradient Boost Regressor'],
             "Mean Squared Error":[mean_squared_error(y_test, y_hat1),
                                   mean_squared_error(y_test, y_hat2),
                                   mean_squared_error(y_test, y_hat3)]})

regressor = LinearRegression()
regressor.fit(x, y)
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[2, 9, 6]]))
#result = best_model.predict(np.array(9.25).reshape(-1,1))
#print("Score of a student who studies 9.25 hours a day: {}%".format(round(result[0][0], 1)))


#In this task we have predicted the percentage of marks a student is expected to score based upon the amount of time they spend for studying. We have test three models for accuracy; among the three models we found that Ridge Regressior and Linear Regressor have similar accuracy with Rigde Regressor having the least value of MSE. Whereas XGB Regressor have showed greater error while predicting. XGB Models are more sophesticated and advanced than the other two, it is used main for big datasets. Therefore XGB models are model recomended for this project the available datas are very small in number.

