#!/usr/bin/env python
# coding: utf-8

# In[105]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_curve


# In[4]:


df=pd.read_csv('avocado.csv')


# In[6]:


df


# we can rename the unnammed column as Week

# In[7]:


df.rename(columns={'Unnamed: 0': 'Week'}, inplace=True)


# In[8]:


df


# In[9]:


df.describe()


# In[10]:


df.isna().sum()


# In[11]:


df.info()


# In[12]:


for i in df.columns:
    print(df[i].value_counts())
    print('\n')


# In[13]:


sorted_df = df.sort_values(by='Week')


# In[14]:


sorted_df


# In[15]:


df['type'].value_counts()


# In[16]:


df['region'].value_counts()


# In[17]:


df.replace(" ","")


# In[18]:


df.isna().sum()


# 1.There is no Null Values in the data set 
# 2.This data set is balanced so first we can focus on classification Problem
# 3. As we can see from the region column it is a multi class classification  

# In[19]:


sns.barplot(x='year',y='AveragePrice',data=df)


# In[20]:


sns.barplot(x='year',y='Total Volume',data=df)


# In[21]:


sns.barplot(x='region',y='AveragePrice',data=df)


# In[22]:


plt.figure(figsize=(20,16))
sns.barplot(x='region',y='Total Volume',data=df)
plt.xticks(rotation=45)
plt.show()


# In[23]:


plt.figure(figsize=(20,16))
sns.barplot(x='region',y='AveragePrice',data=df)
plt.xticks(rotation=45)
plt.show()


# In[24]:


df['Date']=pd.to_datetime(df['Date'],dayfirst=True)


# In[25]:


df['region'].value_counts()


# In[26]:


df.info()


# In[27]:


sorted_df = df.sort_values(by='Date')


# In[28]:


sorted_df


# In[29]:


df.corr()


# In[30]:


skew=('Total Volume','4046','4225','4770','Total Bags','Small Bags','Large Bags','XLarge Bags')
for i in skew:
    if df.skew().loc[i]>0.55:
        df[i]=np.log1p(df[i])


# In[31]:


df.skew()


# In[32]:


df.hist(figsize=(14,14),grid=True,layout=(4,4))


# In[33]:


sns.set(style='whitegrid')
sns.lmplot(x='year',y='AveragePrice',data=df,height=6)
plt.show()


# we can see from the above graph price of the single avacado slightly incresed year by year

# In[34]:


def dt_to_int(dt_obj):
    return dt_obj.toordinal()

df['Date'] = df['Date'].apply(dt_to_int)


# In[35]:


L=LabelEncoder()


# In[36]:


df['type']=L.fit_transform(df['type'])


# In[37]:


df['region']=L.fit_transform(df['region'])


# In[38]:


df


# In[77]:


df.plot(kind='box',subplots=True,layout=(4,4),figsize=(12,10))


# In[67]:


sns.set(style='whitegrid')
sns.(x='year',y='Total Volume',data=df,height=6)
plt.show()


# In[79]:


sns.lineplot(x='year',y='AveragePrice',hue='type', data=df)


# In[81]:


plt.figure(figsize=(20,10))
p=1
for i in df.columns:
    plt.subplot(5,4,p)
    sns.distplot(df[i],color='r',hist=True)
    plt.xlabel(i)
    p+=1
plt.show()
    


# In[82]:


sns.barplot(x='year',y='Total Volume',hue='type',data=df)


# from the above graph we can see that conventional type of fruits sold more than organic however the price of organic type is high

# Training and testing the model for classification

# In[90]:


X=df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[91]:


y


# In[92]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# In[94]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[106]:


model=[
    RandomForestClassifier(),
    DecisionTreeClassifier(),
    AdaBoostClassifier(),
    LogisticRegression(),
    KNeighborsClassifier(),
    ExtraTreesClassifier(),
]


# In[107]:


for i in model:
    i.fit(X_train,y_train)
    y_pred=i.predict(X_test)
    print(i)
    print('\tAccuracy Score :',accuracy_score(y_test,y_pred))
    print('\t Confusion Matrix : ',confusion_matrix(y_test,y_pred))
    score=cross_val_score(i,X_train,y_train)
    print(score)
    print(score.mean())
    print("Difference btw accurancy and CV score is  ",accuracy_score(y_test, y_pred) - score.mean())


# Extra Tree Classifier gives the best performance

# In[108]:


etc=ExtraTreesClassifier()
etc.fit(X_train,y_train)
predetc=etc.predict(X_test)
print(accuracy_score(y_test,predetc))
print(confusion_matrix(y_test,predetc))


# In[109]:


import joblib
joblib.dump(etc,'Avacado Region and Price')


# Regression Problem

# Importing Libraries

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error


# In[39]:


df


# In[41]:


X=df.iloc[:,3:]
X


# In[43]:


y=df.iloc[:,2]
y


# In[44]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[45]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[46]:


model=[
    LinearRegression(),
    RandomForestRegressor(),
    AdaBoostRegressor(),
    ExtraTreesRegressor(),
    DecisionTreeRegressor()
    
]


# In[47]:


for i in model:
    i.fit(X_train,y_train)
    ypred=i.predict(X_test)
    print(i)
    print(r2_score(y_test,ypred))
    


# ExtraTree Regressor performed well in this  case also
# 

# In[49]:


et=ExtraTreesRegressor()
et.fit(X_train,y_train)
ypred=et.predict(X_test)
print(r2_score(y_test,ypred))
print(mean_squared_error(y_test,ypred))
print(mean_absolute_error(y_test,ypred))


# In[53]:


s=r2_score(y_test,ypred)
for i in range(2,11):
    score=cross_val_score(et,X,y,cv=i)
    lsc=score.mean()
    print('At CV=',i)
    print('Cross validation score is =',lsc*100)
    print('accuracy score is =',s*100)
    print('\n')


# In[54]:


import joblib
joblib.dump(et,'Avacado Region and Price')


# In[ ]:





# In[ ]:




