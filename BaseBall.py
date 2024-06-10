#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('BaseBall')


# In[3]:


df


# In[4]:


df.shape


# it has 30 rows and 17 columns

# In[5]:


df.replace(" ","")


# In[6]:


df.info()


# In[7]:


df.isna().sum()


# No NUll values in the data set

# In[8]:


df.describe()


# as we can see 
# spread of the data is high 
# in some columns mean is greater than median so the data is skewed to right

# In[9]:


for i in df.columns:
    print(i)
    print(df[i].value_counts)
    print('\n')
    


# In[10]:


df.corr()


# In[11]:


plt.figure(figsize=(20,10))
p=1
for i in df.columns:
    plt.subplot(5,6,p)
    sns.scatterplot(data=df,x='W',y='RA')
    p+=1
plt.show()


# In[12]:


sns.scatterplot(data=df,x='W',y='RA')


# In[13]:


sns.scatterplot(data=df,x='W',y='ER')


# In[14]:


sns.scatterplot(data=df,x='W',y='ERA')


# In[15]:


sns.scatterplot(data=df,x='ERA',y='RA')


# In[16]:


sns.scatterplot(data=df,x='RA',y='ERA')


# In[17]:


sns.histplot(data=df,x='ER')


# In[ ]:





# as we can see from the above table RA ERA ER are negetively corelated with the target variable
# There is multi coliniarity between the variables

# In[18]:


df.skew()


# checking for outliers

# In[19]:


df.plot(kind='box',subplots=True,layout=(3,6))


# we can see that outliers are present in E column 
# TO remove the outliers we use Zscore

# In[20]:


from scipy.stats import zscore


# In[21]:


Z=np.abs(zscore(df))
Z


# In[22]:


baseball=df[(Z<3).all(axis=1)]


# In[23]:


baseball.shape


# In[24]:


X=baseball.drop(columns='W',axis=1)
y=baseball['W']


# In[25]:


X.shape


# In[26]:


plt.figure(figsize=(20,15))
sns.heatmap(baseball.corr(),annot=True,cmap='PiYG')


# In[27]:


sns.scatterplot(data=df,x='SHO',y='ER')


# In[28]:


sns.scatterplot(data=df,x='SV',y='RA')


# In[34]:


from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import xgboost as xgb


# In[30]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# In[31]:


regressor=[
    RandomForestRegressor(),
    AdaBoostRegressor(),
    GradientBoostingRegressor(),
    ExtraTreesRegressor(),
    LinearRegression(),
    Ridge(),
    Lasso(),
    KNeighborsRegressor(),
    DecisionTreeRegressor()  
]


# In[32]:


for i in regressor:
    i.fit(X_train,y_train)
    ypred=i.predict(X_test)
    print(i)
    print(r2_score(y_test,ypred))
    print(mean_absolute_error(y_test,ypred))


# Ridge regressor is working well so now we are going to work on hyper parameter tuning

# In[ ]:


alpha=1.0, *, fit_intercept=True, copy_X=True, max_iter=None, tol=0.0001, solver='auto', positive=False, random_state=None


# In[38]:


params={
    'copy_X':[True,False],
    'alpha':[1.0,1.5,2,0.5],
    'positive':[True,False],
    'fit_intercept':[True,False]
}


# In[39]:


model=GridSearchCV(Ridge(),params)


# In[40]:


grid=model.fit(X_train,y_train)


# In[41]:


print(grid.best_params_)


# In[42]:


print(grid.best_estimator_)


# In[45]:


R=Ridge(alpha= 2, copy_X= True, fit_intercept= True, positive= True)


# In[46]:


R.fit(X_train,y_train)


# In[47]:


pred=R.predict(X_test)


# In[48]:


print("r2_score is = ",r2_score(y_test,pred)*100)


# In[ ]:





# In[ ]:




