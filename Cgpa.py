#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[38]:


df=pd.read_csv("CGPA")
df


# In[39]:


df.info()


# There are sum null values in some of the columns

# In[40]:


df.isnull().sum()


# The above cell represents the how many null values are present in each column.
# To fill the null values we use mode value 

# In[41]:


for i in df.columns:
    mode=df[i].mode()[0]
    df[i]=df[i].fillna(mode)
## filling null values 


# In[42]:


df.info()


# There is no null values anymore  in the data set

# In[43]:


sns.heatmap(df.isnull())


# In[44]:


df.describe()


# There is no effect of CGPA based on the Seat Number so we can drop the  seat number coulumn

# In[45]:


df.drop(columns='Seat No.',axis=1,inplace=True)


# In[46]:


X=df.drop(columns='CGPA')
y=df['CGPA']


# In[47]:


X


# In[48]:


y


# In[49]:


from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.preprocessing import OneHotEncoder


# In[50]:


enc=OneHotEncoder()


# In[51]:


X=enc.fit_transform(X)


# In[52]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# Linear Regression

# In[53]:


l=LinearRegression()


# In[54]:


l.fit(X_train,y_train)


# In[55]:


y_pred=l.predict(X_test)


# In[56]:


r2score=r2_score(y_test,y_pred)
r2score


# In[57]:


from sklearn.ensemble import RandomForestRegressor


# In[58]:


r=RandomForestRegressor()


# In[59]:


r.fit(X_train,y_train)


# In[60]:


r_pred=r.predict(X_test)


# In[62]:


r2score=r2_score(y_test,r_pred)


# In[63]:


r2score


# with hyper parameter tuning

# In[76]:


param_grid = {
    'n_estimators': [50, 100, 200,150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}


# In[77]:


grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)


# In[78]:


print("Best Parameters:", grid_search.best_params_)


# In[79]:


y_pred=grid_search.predict(X_test)


# In[80]:


r=r2_score(y_test,y_pred)


# In[81]:


r


# In[ ]:




