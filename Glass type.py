#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# Loading the DataSet

# In[2]:


df=pd.read_csv('Glasstype.csv')


# In[3]:


df.iloc[120:180]


# In this dataset 214 rows and 11 columns are present

# In[4]:


df.isnull().sum()


# There is no null values in the data set

# In[5]:


df.describe()


# In[6]:


df.corr()


# as per the above table K and Ca are less corelated with glass type so we can drop the coloumns K and Ca

# In[7]:


df=df.drop(columns=['K','Ca','ID_number'])


# we can drop id number column as well because it doesnt effect the output variable 

# In[8]:


sns.pairplot(df)


# In[9]:


p=1
plt.figure(figsize=(20,22))
for i in df:
    if p<=8:
        plt.subplot(5,4,p)
        sns.boxplot(df[i])
        plt.xlabel(i)
    p+=1
plt.show()


# There are some outliers are present in this data set in Ri,AL,Fe,Si columns

# In[10]:


p=1
plt.figure(figsize=(20,22))
for i in df:
    if p<=8:
        plt.subplot(5,4,p)
        sns.distplot(df[i])
        plt.xlabel(i)
    p+=1
plt.show()


# In[11]:


plt.figure(figsize=(12, 8))
sns.countplot(x='Type of glass', data=df, palette='viridis')
plt.title('Count of Glass type')
plt.xlabel('Type of glass')
plt.ylabel('Count')
plt.show()


# target variable values are in assending order to get effecctive results we need to sort the variables
# i am using the sorting values based on Na and Ba variables

# In[17]:


sorted_df = df.sort_values(by=['Na','Ba'])


# In[18]:


sorted_df.tail(50)


# In[21]:


X=sorted_df.drop(columns='Type of glass',axis=1)
y=sorted_df['Type of glass']


# In[50]:


from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.ensemble import RandomForestClassifier


# In[44]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[45]:


rm=RandomForestClassifier()


# In[46]:


rm.fit(X_train,y_train)


# In[47]:


y_pred=rm.predict(X_test)


# In[48]:


a=confusion_matrix(y_pred,y_test)


# In[49]:


a


# In[51]:


accuracy=accuracy_score(y_pred,y_test)
accuracy


# Decision Tree

# In[52]:


from sklearn.tree import DecisionTreeClassifier


# In[53]:


dt=DecisionTreeClassifier()


# In[67]:


params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10]
}


# In[68]:


grid=GridSearchCV(dt,param_grid=params,cv=5)


# In[69]:


grid.fit(X_train,y_train)


# In[70]:


y_predcv=grid.predict(X_test)


# In[71]:


accuracy=accuracy_score(y_predcv,y_test)
accuracy


# In[72]:


cv=cross_val_score(rm,X,y,cv=5)


# In[73]:


cv


# In[74]:


cv.mean()


# Support Vector Machine Classifier

# In[75]:


from sklearn import svm


# In[76]:


clf=svm.SVC()


# In[77]:


clf.fit(X_train,y_train)


# In[78]:


ypred=clf.predict(X_test)


# In[79]:


acc=accuracy_score(y_test,ypred)
acc


# In[80]:


params = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
}


# In[84]:


grid_search = GridSearchCV(clf, param_grid=params, cv=5)


# In[86]:


grid_search.fit(X_train, y_train)


# In[87]:


best_model = grid_search.best_estimator_


# In[89]:


y_pred = best_model.predict(X_test)


# In[91]:


accuracy = accuracy_score(y_test, y_pred)
accuracy


# In[96]:


from sklearn.ensemble import GradientBoostingClassifier


# In[98]:


clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=42).fit(X_train, y_train)
clf.score(X_test, y_test)


# In[ ]:




