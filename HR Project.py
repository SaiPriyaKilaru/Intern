#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import warnings 
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")


# In[3]:


df


# In this data set consist of 1470 rows and 35 columns 
# it is a Binary classification Dataset

# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.isna().sum()


# In[7]:


df.describe()


# In[8]:


df.columns


# These are the columns present in the dataset

# In[9]:


df.nunique()


# From the above cell output we can understand that In Employee Count,Over 18 and Standard Hours has only one unique value for all the employees so it doesnt influence on the attrition column, Therefore we can drop these colums

# In[10]:


for i in df.columns:
    print(df[i].value_counts)
    print('\n')
    print(df[i].unique())
    print('\n')


# These are the unique values in each and every column

# EDA

# In[11]:


df.head()


# In[12]:


df.drop(columns=['EmployeeCount','EmployeeNumber','StandardHours','Over18'],inplace=True,axis=1)


# In[13]:


df.shape


# In[14]:


plt.figure(figsize=(20,60))
p=1
columns_to_plot = df.columns[:20]
for i in columns_to_plot:
    plt.subplot(5,4,p)
    sns.countplot(x=i,data=df,hue='Attrition')
    plt.xlabel(i)
    p+=1
plt.show()


# In[12]:


from sklearn.preprocessing import LabelEncoder


# In[13]:


L=LabelEncoder()


# In[14]:


df['Attrition']=L.fit_transform(df['Attrition'])


# In[15]:


df['BusinessTravel']=L.fit_transform(df['BusinessTravel'])


# In[16]:


df['Department']=L.fit_transform(df['Department'])


# In[17]:


df['Gender']=L.fit_transform(df['Gender'])


# In[18]:


df['EducationField']=L.fit_transform(df['EducationField'])


# In[19]:


df['JobRole']=L.fit_transform(df['JobRole'])


# In[20]:


df['OverTime']=L.fit_transform(df['OverTime'])


# In[21]:


df['MaritalStatus']=L.fit_transform(df['MaritalStatus'])


# In[22]:


df.info()


# In[23]:


sns.countplot(x='Attrition',data=df)


# The above dataset is imbalance 
# To balance that we can use different sampling techniques like SMOTE,ADASYN

# In[24]:


from imblearn.over_sampling import SMOTE,ADASYN
smt=SMOTE()
asyn=ADASYN()


# In[25]:


from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression


# In[26]:


X=df.drop(columns=['Attrition'],axis=1)


# In[27]:


X


# In[33]:


X=X.drop(columns=['HourlyRate'])


# In[34]:


y=df['Attrition']


# In[35]:


X_res,y_res=smt.fit_resample(X,y)


# In[36]:


print(X_res.shape)
print(y_res.shape)


# In[37]:


X_train,X_test,y_train,y_test=train_test_split(X_res,y_res,test_size=0.3,random_state=42)


# In[45]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier,ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_auc_score


# In[51]:


model=[
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    AdaBoostClassifier(),
    ExtraTreesClassifier(),
    XGBClassifier()
    
]


# In[52]:


for i in model:
    i.fit(X_train,y_train)
    ypred=i.predict(X_test)
    print(i)
    print(accuracy_score(ypred,y_test))
    print(confusion_matrix(ypred,y_test))
    print(classification_report(y_test,ypred))
    score=cross_val_score(i,X_train,y_train,cv=5)
    print(score)
    print("Difference btw accurancy and CV score is  ",accuracy_score(y_test, ypred) - score.mean())
    print("-"*50)


# In[50]:


df.replace(" ","")


# In[56]:


pip install --upgrade pycaret


# In[55]:


from pycaret.classification import *
s=setup(attrition, target = 'Attrition')


# In[54]:


get_ipython().system('pip install pycaret')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




