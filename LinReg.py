#!/usr/bin/env python
# coding: utf-8

# In[49]:


#Import the required libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm


# In[31]:


#Import the Boston dataset from the scikit learn library
from sklearn.datasets import load_boston
boston_dataset = load_boston()


# In[27]:


#Create a pandas dataframe and store the data
#To store features names as columns in the dataframe
df_boston = pd.DataFrame(boston_dataset.data)
df_boston.columns = boston_dataset.feature_names


# In[25]:


#Append price, the target value, as a new column in the data set.
df_boston['Price'] = boston_dataset.target


# In[8]:


#View the top five observations
df_boston.head()


# In[24]:


#Assign features on the X axis

X_features = boston_dataset.data


# In[10]:


#Assign target on the Y axis
Y_target = boston_dataset.target


# In[21]:


#Import linear model, which is the estimator
from sklearn.linear_model import LinearRegression
linReg = LinearRegression()


# In[41]:


#Fit the data into the estimator
model = LinearRegression().fit(X_features, Y_target)


# In[42]:


#Print the intercept
print('intercept:', model.intercept_)


# In[43]:


# Print the Slope:
print('slope:', model.coef_) 


# In[46]:


# Obtain the coefficient of determination by calling the model with the score() function, then print the coefficient:
r_sq = model.score(X_features, Y_target)
print('coefficient of determination:', r_sq)


# In[48]:


# Predict a Response and print it:
y_pred = model.predict(X_features)
print('Predicted response:', y_pred, sep='\n')


# In[53]:


sm.add_constant(X_features)
mod = sm.OLS(Y_target, X_features)
res = mod.fit()

print(res.summary())


# In[54]:


#Train the model by splitting the whole dataset into train and test datasets
from sklearn.model_selection import train_test_split


# In[55]:


X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y_target)


# In[56]:


#Print the shape of the dataset
print(boston_dataset.data.shape)


# In[57]:


#Print the shapes of the training and testing datasets
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# In[58]:


#Fit the training sets into the model
linReg.fit(X_train, Y_train)


# In[144]:


#Caculate the mean squared erorr (MSE) or residual sum of squares, r2 score
from sklearn.metrics import mean_squared_error, r2_score
predictions = model.predict(X_features)
rmse = mean_squared_error(Y_target, predictions, squared=False)
R2 = r2_score(Y_target, predictions)

print('The r2 is:',R2)
print('The rmse is:', rmse)


# In[145]:


print('Mean Square Error (MSE): %.2f ' % np.mean((linReg.predict(X_test)-Y_test) ** 2))


# In[146]:


#Calculate the variance score
print('Variance Score: %.2f ' % linReg.score(X_test, Y_test))


# In[ ]:




