#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np
import xgboost as xgb
#from xgboost.xgbclassifier import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[28]:


pip install xgboost


# In[33]:


data = pd.read_csv("http://bit.ly/w-data")
print("Shape of dataset: {}".format(data.shape))


# In[35]:


data.head(10)


# In[36]:



# Number of missing values in the dataset
data.isnull().sum()


# In[37]:


# information on data set
data.info()


# In[38]:


# data visualised using scatterplot. Linear dependents of both the variables can be clearly seen
sns.scatterplot(data['Hours'], data['Scores'])


# In[39]:


# converting features into x and score to y
x = np.array(data['Hours']).reshape(-1,1)
y = np.array(data['Scores']).reshape(-1,1)


# In[40]:


# splitting the data into train and test datasets
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3)
print("x_train size: ", x_train.shape)
print("x_test size: ", x_test.shape)
print("y_train size: ", y_train.shape)
print("x_test size:", y_test.shape)


# In[41]:


# Initialising model and fitting train dataset
lr = LinearRegression()
lr.fit(x_train, y_train)


# In[42]:



# Predicting scores using test dataset
y_hat1 = lr.predict(x_test)


# In[43]:


# checking the meansquared error of the prediction. Then visualising actual values and predicted values
print("mean squared error: ", mean_squared_error(y_test, y_hat1))
plt.scatter(y_test, x_test, label = "Actual Value")
plt.scatter(y_hat1, x_test, label = "Predicted Value")
plt.xlabel('Score')
plt.ylabel('Hours')
plt.legend()


# In[44]:


# tuning the model
parameters= [{'alpha': [0.001,0.1,1, 10, 100, 1000, 10000, 100000, 100000]}]
rr=Ridge()
grid = GridSearchCV(rr, parameters,cv=4)


# In[45]:


# model with the best alpha parameter to chosen and train datasets are fitted to the model
grid.fit(x_train, y_train)
best_model = grid.best_estimator_
best_model


# In[46]:


# predicting score using test dataset
y_hat2 = best_model.predict(x_test)


# In[47]:


# checking the meansquared error of the prediction. Then visualising actual values and predicted values
print("mean squared error: ", mean_squared_error(y_test, y_hat2))
plt.scatter(y_test, x_test, label = "Actual Value")
plt.scatter(y_hat2, x_test, label = "Predicted Value")
plt.xlabel('Score')
plt.ylabel('Hours')
plt.legend()


# In[48]:


# Initialising the model and fitting train datasets to the model.
xgbr = xgb.XGBRegressor()
xgbr.fit(x_train,y_train)


# In[49]:


# predicting scores using test dataset
y_hat3 = xgbr.predict(x_test)


# In[51]:


# checking the meansquared error of the prediction. Then visualising actual values and predicted values
print("mean squared error: ", mean_squared_error(y_test, y_hat3))
plt.scatter(y_test, x_test, label = "Actual Value")
plt.scatter(y_hat3, x_test, label = "Predicted Value")
plt.xlabel('Score')
plt.ylabel('Hours')
plt.legend()


# In[52]:


pd.DataFrame({"Model":['sklearn,LinearRigression', 'Ridge Regressior','Gradient Boost Regressor'],
             "Mean Squared Error":[mean_squared_error(y_test, y_hat1),
                                   mean_squared_error(y_test, y_hat2),
                                   mean_squared_error(y_test, y_hat3)]})


# In[53]:


result = best_model.predict(np.array(9.25).reshape(-1,1))
print("Score of a student who studies 9.25 hours a day: {}%".format(round(result[0][0], 1)))


# In[54]:


#In this task we have predicted the percentage of marks a student is expected to score based upon the amount of time they spend for studying. We have test three models for accuracy; among the three models we found that Ridge Regressior and Linear Regressor have similar accuracy with Rigde Regressor having the least value of MSE. Whereas XGB Regressor have showed greater error while predicting. XGB Models are more sophesticated and advanced than the other two, it is used main for big datasets. Therefore XGB models are model recomended for this project the available datas are very small in number.


# In[ ]:




