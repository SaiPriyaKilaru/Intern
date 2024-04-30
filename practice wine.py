#!/usr/bin/env python
# coding: utf-8

# Wine Data set 
# Classification problem

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[18]:


df=pd.read_csv('wine.csv')


# In[19]:


df


# 1599 rows and 12 columns are present in this data set

# In[5]:


df.isna().sum()


# There is no null values in this data set

# In[20]:


df.corr()


# as we can see from the above data Residual Sugars,free sulpher dioxide,ph are not corelated with the target variable  so we can drop from the original dataset

# In[6]:


df.info()


# checking the data types of features

# In[25]:


df1=df.drop(['residual sugar','free sulfur dioxide','pH'],axis=1)


# In[26]:


df1


# In[ ]:





# These are the columns present in the dataset

# In[8]:


df.value_counts


# In[28]:


p=1
plt.figure(figsize=(20,25))
for i in df1:
    if p<=8:
        plt.subplot(5,4,p)
        sns.boxplot(df1[i])
        plt.xlabel(i)
    p+=1
plt.show()


# outliers are present in almost all columns so by using z score we remove outliers
# 

# In[34]:


from scipy.stats import zscore
out_fea=df1[['fixed acidity','volatile acidity','citric acid','chlorides','total sulfur dioxide','density','sulphates','alcohol']]
z=np.abs(zscore(out_fea))


# In[35]:


z


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[29]:


df1.columns


# In[ ]:





# In[ ]:





# In[40]:


Target=[]
for i in df1['quality']:
    if(i>=7):
        Target.append('1')
    else:
        Target.append('0')


# In[41]:


df['quality']=Target


# considering quality score >=7 is 1 <7 is 0

# In[42]:


df.head(25)


# changeing the values of quality(target) variable column into 1 and 0 which turns into binary classification problem

# In[12]:


X=df.iloc[:,:-1]
Y=df.iloc[:,-1]


# In[13]:


(unique,count)=np.unique(df['quality'],return_counts=True)



# In[14]:


sns.barplot(x=unique,y=count)
print(unique,count)
plt.title('Target variable counts')
plt.xlabel('class')
plt.ylabel('numner of values')
plt.show()


# as we can see the from the above bar graph  no values are very high and less values are very less around 250
# Therefore this data set is imbalnced

# To balance the imbalanced data set  we use different sampling techniques those are under and over sampling.
# In the above case we have only 1599 records in which 1382 records are bad category and 217 records are in good category.
# if we use under sampling technique only 434 records are in consideration eventually our accuracy will effect on this hence we decided to use oversampling method called SMOTE Technique

# In[15]:


X.shape


# In[16]:


Y.shape


# In[17]:


df.corr()


# In[ ]:





# In[42]:


good=[]
bad=[]


# In[43]:


good=df[df1['quality']=='1']
bad=df[df1['quality']=='0']


# In[44]:


from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import numpy as np
from xgboost import XGBClassifier
import warnings


# In[46]:


from imblearn.combine import SMOTETomek


# In[47]:


smk=SMOTETomek(random_state=42)


# In[62]:


df1


# In[ ]:





# In[47]:


from imblearn.over_sampling import SMOTE


# smk=SMOTE()

# In[48]:


smk=SMOTE()


# In[49]:


sm = SMOTE(random_state=42, k_neighbors=5)


# In[68]:


X_res, y_res = sm.fit_resample(X, Y)


# In[69]:


X_res


# In[63]:


from scipy.stats import zscore
out_fea=X_res[['fixed acidity','volatile acidity','citric acid','chlorides','total sulfur dioxide','density','sulphates','alcohol']]
z=np.abs(zscore(out_fea))


# In[64]:


z


# In[65]:


np.where(z>3)


# In[ ]:





# In[54]:


sns.pairplot(X_res)


# In[53]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[54]:


X_train,X_test,y_train,y_test=train_test_split(X_res, y_res,test_size=0.3,random_state=42)


# In[55]:


import sklearn 
sklearn.linear_model.LogisticRegression


# In[56]:


rf = RandomForestClassifier(n_estimators=100, random_state=42)


# In[57]:


rf.fit(X_train,y_train)


# In[58]:


y_pred=rf.predict(X_test)


# In[72]:


p=1
plt.figure(figsize=(20,25))
for i in df1:
    if p<=10:
        ax=plt.subplot(6,4,p)
        sns.distplot(df[i])
        plt.xlabel(i)
    p+=1
plt.show()


# In[ ]:





# In[60]:


accuracy=accuracy_score(y_test,y_pred)


# In[61]:


accuracy


# In[92]:


confusionmatrix=confusion_matrix(y_test,y_pred)


# In[93]:


confusionmatrix


# In[ ]:


ls=LogisticRegression()


# In[94]:


from sklearn.metrics import roc_auc_score
auc_roc=roc_auc_score(y_test,y_pred)


# In[95]:


auc_roc


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


sns.heatmap(df.corr())


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


# In[ ]:


df['Target']=df['quality']


# In[ ]:


df.unique


# In[ ]:


# Plot box plots for each numerical column
for col in numerical_columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(df[col].dropna())
    plt.title(col)
    plt.ylabel('Value')
    plt.show()


# In[ ]:


for col in columns:
    plt.figure(figsize(6,4))
    sns.bloxplot(df[col])
    plt.title(col)
    plt.ylabel("value")
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:


df.head(15)


# In[ ]:





# In[ ]:




