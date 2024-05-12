#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# Loading the DataSet

# In[4]:


df=pd.read_csv('Titanic.csv')


# In[5]:


df


# Data set has 891 Rows and 12 Columns

# In[6]:


df.describe()


# In[7]:


df.info()


# There are some null  values present in the dataset 
# age and cabin and embarked coulumns has missing values.
# 

# In[8]:


df['Age']=df['Age'].fillna(df['Age'].median())


# In[9]:


df.info()


# missing values are covered with the median of the column 

# In[10]:


sns.heatmap(df.corr())


# In[11]:


df.drop(columns=['PassengerId','Ticket','Name','Cabin'],axis=1,inplace=True)


# In[12]:


from sklearn.preprocessing import LabelEncoder


# In[13]:


le=LabelEncoder()
df['Sex']=le.fit_transform(df['Sex'])
df['Embarked']=le.fit_transform(df['Embarked'])


# In[14]:


df


# In[15]:


sns.pairplot(df)


# In[16]:


p=1
plt.figure(figsize=(20,25))
for i in df:
    if p<=8:
        plt.subplot(5,4,p)
        sns.boxplot(df[i])
        plt.xlabel(i)
    p+=1
plt.show()


# In[17]:


df['Survived'].value_counts()


# its a balanced data set

# In[ ]:





# In[18]:


p=1
plt.figure(figsize=(20,25))
for i in df:
    if p<9:
        plt.subplot(5,4,p)
        sns.boxplot(df[i])
        plt.xlabel(i)
    p+=1
plt.show()


# some outliers are present in the sibsp and fare and age columns we can remove those by using Z score 

# In[19]:


p=1
plt.figure(figsize=(20,25))
for i in df:
    if p<=7:
        plt.subplot(5,4,p)
        sns.distplot(df[i])
        plt.xlabel(i)
    p+=1
plt.show()


# Without Removing outliers

# In[20]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score
from sklearn.model_selection import cross_val_score,GridSearchCV,train_test_split


# In[21]:


X=df.iloc[:,1:]
y=df.iloc[:,:1]


# In[22]:


y


# In[23]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# In[24]:


l=LogisticRegression()


# In[25]:


df


# In[26]:


l.fit(X_train,y_train)


# In[27]:


y_pred=l.predict(X_test)


# In[28]:


score=accuracy_score(y_pred,y_test)
score


# In[29]:


roc_score=roc_auc_score(y_pred,y_test)


# In[30]:


roc_score


# In[31]:


confusion_matrix(y_pred,y_test)


# Support vector Machine Classifier:

# In[32]:


from sklearn import svm
clf=svm.SVC()


# In[33]:


clf.fit(X_train,y_train)


# In[34]:


y_predsvm=clf.predict(X_test)


# In[35]:


score=accuracy_score(y_predsvm,y_test)
score


# In[36]:


confusion_matrix(y_predsvm,y_test)


# Decision Tree

# In[37]:


from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)


# In[38]:


a=cross_val_score(clf, X, y, cv=10)


# In[39]:


np.mean(a)


# In[40]:


a


# RandomForestClassifier

# In[41]:


from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()


# In[42]:


param_grid={
    'n_estimators':[40,80,120,150],
    'max_depth':[None,5,10],
    'min_samples_split':[2,5,10]
}


# In[48]:


grid=GridSearchCV(estimator=model,param_grid=param_grid,cv=10)
grid.fit(X_train,y_train)


# In[49]:


ypred=grid.predict(X_test)


# In[47]:


cross_val_scores = cross_val_score(grid.best_estimator_, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cross_val_scores


# In[50]:


accuracy_score=accuracy_score(ypred,y_test)


# In[51]:


accuracy_score


# In[45]:


grid.best_params_


# In[52]:


confusion_matrix(ypred,y_test)


# Randomforest classifier and logistic regression gives the best accuracy

# In[ ]:




