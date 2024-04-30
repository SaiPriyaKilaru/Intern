#!/usr/bin/env python
# coding: utf-8

# Health Insurance Dataset

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("Insurance.csv")


# In[37]:


df


# this dataset consist of 1338 rows and 7 columns are present

# In[4]:


for i in df:
    print(df[i].value_counts())


# In[5]:


df.describe()


# In[ ]:





# In[6]:


df.corr()


# In[7]:


df.isna().sum()


# There is no Null values in this dataset

# In[8]:


for i in df:
    print(df[i].value_counts())
    print('/n')


# In[9]:


sns.pairplot(df)


# In[10]:


plt.boxplot(df['bmi'])


# In[11]:


np.mean(df['bmi'])


# In[12]:


np.median(df['bmi'])


# In[13]:


np.mean(df)


# In[14]:


np.median(df['age'])


# In[15]:


np.median(df['children'])


# In[16]:


np.median(df['charges'])


# There are some outliers are present in bmi column however mean and medians of age,bmi,childrens column are  equal  so there is no skewness present in the data,Hence most of the data present under the bell shaped curve

# In[17]:


sns.boxplot(df['charges'])


# In[18]:


from sklearn.preprocessing import LabelEncoder


# In[19]:


le=LabelEncoder()
df['sex']=le.fit_transform(df['sex'])
df['smoker']=le.fit_transform(df['smoker'])
df['region']=le.fit_transform(df['region'])


# In[20]:


df


# In[21]:


plt.title('Linear line')
plt.xlabel("df['age']")
plt.ylabel("df['region']")
plt.bar(df['charges'],df['age'])
plt.show()


# In[22]:


sns.lmplot(x='bmi',y='charges',data=df)


# In[23]:


sns.lmplot(x='age',y='charges',data=df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[24]:


plt.hist(df['region'])


# In[25]:


plt.hist(df['sex'])


# In[ ]:





# In[26]:


sns.stripplot(x='age',y='charges',data=df)


# as the above graph says if the charges of insurence is increaseing with respect to the age

# In[27]:


sns.stripplot(x='region',y='charges',data=df)


# In[28]:


sns.stripplot(x='children',y='charges',data=df)


# In[29]:


plt.hist(df['children'])


# In[30]:


plt.scatter('children', 'charges',data=df)
plt.xlabel('children')
plt.ylabel('charges')
plt.show()


# In[31]:


p=1
plt.figure(figsize=(20,25))
for i in df:
    if p<=7:
        plt.subplot(5,4,p)
        sns.boxplot(df[i])
        plt.xlabel(i)
    p+=1
plt.show()


# outliers are present in the charges and bmi columns
# By using the Z score we will remove outliers

# In[58]:


p=1
plt.figure(figsize=(20,25))
for i in df:
    if p<=7:
        ax=plt.subplot(6,4,p)
        sns.distplot(df[i])
        plt.xlabel(i)
    p+=1
plt.show()


# In[59]:


df1.skew()


# In[60]:


df1.corr()


# charges are co related with age around 30percent,smoker with 78 percentage  and bmi around 20percentage

# In[34]:


from scipy.stats import zscore
out_fetures=df[['bmi']]
z=np.abs(zscore(out_fetures))
z


# In[35]:


np.where(z>3)


# In[39]:


df1=df[(z<3).all(axis=1)]


# In[40]:


df1


# In[ ]:





# In[ ]:





# In[41]:


X=df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[42]:


y


# In[43]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


# In[44]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()


# In[61]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=0)


# In[62]:


lr.fit(X_train,y_train)


# In[63]:


y_pred=lr.predict(X_test)


# In[64]:


test_error_score = r2_score(y_test, y_pred)


# In[65]:


test_error_score


# In[ ]:





# In[ ]:


mean_squared_error


# In[66]:


from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


# In[67]:


r=Ridge()


# In[68]:


r.fit(X_train,y_train)


# In[69]:


y_pr=r.predict(X_test)


# In[70]:


test_score = r2_score(y_test, y_pr)


# In[71]:


test_score


# In[56]:


l=Lasso()


# In[57]:


l.fit(X_train,y_train)


# In[64]:


ll=l.predict(X_test)


# In[65]:


test_score = r2_score(y_test,ll)


# In[66]:


test_score


# In[ ]:




