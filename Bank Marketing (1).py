#!/usr/bin/env python
# coding: utf-8

# In[1]:


import  numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# Loading the datasets Train and Test

# In[2]:


traindf=pd.read_csv('TrainDatacsv')
testdf=pd.read_csv('Testcsv')


# In[3]:


traindf


# This Training Dataset consist of 31647 rows and 18 columns

# In[4]:


testdf


# This Test Dataset Consist of 13564 rows and 17 columns

# In[5]:


traindf.isna().sum()


# In[6]:


for i in traindf.columns:
    if traindf[i].dtype == 'object':
        print(i)
        print(traindf[i].unique())


# These are the categorical columns present in the data set and unique values in each column

# In[7]:


traindf.info()


# In[8]:


traindf.describe()


# In[9]:


traindf.drop(columns='default',inplace=True)


# In[10]:


traindf


# In[ ]:





# In[ ]:





# Separating nuemerical and categorical columns

# In[11]:


categorical = []
numerical = []

for i in traindf.columns:
    if traindf[i].dtype == 'object':
        categorical.append(i)
    elif traindf[i].dtype == 'int64':
        numerical.append(i)


# In[12]:


categorical_df = traindf[categorical]
numerical_df = traindf[numerical]


# In[13]:


categorical_df


# In[14]:


numerical_df


# In[15]:


numerical_df.corr()


# In[16]:


sns.histplot(traindf['age'],kde=True,bins=20)
plt.title('Histogram of age')
plt.xlabel('age')
plt.ylabel('frequency')
plt.show()


# as per the above plot we can understand that deposit starts from the age  of 20 and end is around 82 
# however most of the deposits made between the age of 30-50

# In[17]:


plt.figure(figsize=(10,5))
sns.barplot(x=traindf['job'].value_counts().index,y=traindf['job'].value_counts())
plt.title('Bar Plot of Jobs')
plt.xlabel('Job')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[18]:


plt.figure(figsize=(10,5))
sns.countplot(x='job',hue='subscribed',data=traindf,order=traindf['job'].value_counts().index)
plt.title('Bar Plot of Jobs')
plt.xlabel('Job')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[19]:


for i in categorical_df:
    plt.figure(figsize=(10,5))
    sns.barplot(x=traindf[i].value_counts().index,y=traindf[i].value_counts())
    plt.xticks(rotation=45)
    plt.show()


# In[20]:


for i in categorical_df:
    plt.figure(figsize=(10,5))
    sns.countplot(x=i,hue='subscribed',data=traindf,order=traindf[i].value_counts().index)
    plt.xticks(rotation=45)
    plt.show()


# from the above graph we can drop the default feature because it has high number of nos than yes

# In[21]:


for i in categorical_df:
    sns.catplot(x='subscribed',hue=i,kind='count',data=traindf)
    plt.show()


# In[22]:


plt.figure(figsize=(20,60),facecolor='white')
p=1
for i in numerical_df:
    ax=plt.subplot(12,3,p)
    sns.distplot(traindf[i])
    plt.xlabel(i)
    p+=1
plt.show()


# In[23]:


plt.figure(figsize=(20,60),facecolor='white')
p=1
for i in numerical_df:
    ax=plt.subplot(12,3,p)
    sns.boxplot(traindf[i])
    plt.xlabel(i)
    p+=1
plt.show()


# age,duration,campain,pdays and balance has some outliers

# In[24]:


sns.heatmap(numerical_df.corr())


# no feature is heavily corelated with other feature

# In[25]:


sns.countplot(x='subscribed',data=traindf)


# This is unbalanced dataset 

# In[26]:


df=traindf.copy()


# In[27]:


df.head()


# In[28]:


df.groupby(['subscribed','pdays']).size()


# it dosent mean anything to the subscription so we can drop the column

# In[29]:


df.drop(columns='pdays',inplace=True,axis=1)


# In[30]:


df.groupby('age',sort=True)['age'].count()


# its start with the age of 18 to 95 we cant remove any  rows from the age column 

# In[31]:


df.groupby(['subscribed','balance'],sort=True)['balance'].count()


# we cant remove the outliers from the balance column because the highest balance increses the chance of subcription

# In[32]:


df.groupby(['subscribed','campaign'],sort=True)['campaign'].count()


# finding outliers

# In[33]:


from scipy.stats import zscore
campaign=df['campaign']


# In[34]:


z_scores = zscore(campaign)

threshold=3


# In[35]:


outliers = campaign[abs(z_scores) > threshold]
print(outliers)


# there are 604 columns are outliers present in the campaign colum so i am removing those columns resulted dataframe named as cleaned_data

# In[36]:


cleaned_data = df[abs(z_scores) <= threshold]


# In[37]:


cleaned_data.shape


# In[38]:


df.shape


# In[39]:


df.groupby(['subscribed','duration'],sort=True)['duration'].count()


# if the duration is high chances of subricption is high so we cant remove outliers from the data

# In[40]:


cleaned_data.drop(columns='ID',inplace=True)


# In[41]:


cleaned_data


# In[42]:


from sklearn.preprocessing import LabelEncoder


# In[43]:


l=LabelEncoder()


# In[44]:


cleaned_data['housing']=l.fit_transform(cleaned_data['housing'])


# In[45]:


cleaned_data['loan']=l.fit_transform(cleaned_data['loan'])


# In[46]:


cleaned_data['subscribed']=l.fit_transform(cleaned_data['subscribed'])


# In[47]:


cleaned_data=pd.get_dummies(cleaned_data, columns=['job', 'marital', 'education', 'contact', 'day', 'month','poutcome'])


# In[48]:


cleaned_data


# In[ ]:





# In[49]:


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


# In[50]:


# Assuming you have already created the 'cleaned_data' DataFrame

# Drop the 'subscribed' column from cleaned_data to create X
X =cleaned_data.drop(columns='subscribed')

# Assign the 'subscribed' column to y
y =cleaned_data['subscribed']


# In[51]:


smote = SMOTE(random_state=42)


# In[52]:


X_resampled, y_resampled = smote.fit_resample(X, y)


# In[88]:


X_resampled


# In[89]:


y_resampled.value_counts()


# now its a balanced data set

# In[90]:


y_resampled


# For the Test Data

# In[56]:


testdf


# In[57]:


testdf.drop(columns=['ID','default','pdays'],inplace=True)


# In[58]:


testdf


# In[59]:


testdf['housing']=l.fit_transform(testdf['housing'])


# In[60]:


testdf['loan']=l.fit_transform(testdf['loan'])


# In[61]:


testdf=pd.get_dummies(testdf, columns=['job', 'marital', 'education', 'contact', 'day', 'month','poutcome'])


# In[62]:


testdf


# Training the Dataset

# In[63]:


X_train,X_test,y_train,y_test=train_test_split(X_resampled,y_resampled,test_size=0.3,random_state=42)


# In[64]:


X_train


# In[65]:


X_test


# In[72]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier


# In[74]:


from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score
import time


# In[83]:


model={
    'RandomForest':RandomForestClassifier(),
    'DecisionTree':DecisionTreeClassifier(),
    'AdaBoost':AdaBoostClassifier(),
    'XGB':XGBClassifier()
}


# In[84]:


from time import time
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score


head=12
for name, classifier in model.items():
    start=time()
    print(name)
    classifier.fit(X_train,y_train)
    y_pred=classifier.predict(X_test)
    train_time=time()-start
    start = time()
    predict_time = time()-start
    print(i)
    print("\tAccuracy Score:", accuracy_score(y_test, y_pred))
    print("\tConfusion Matrix:", confusion_matrix(y_test, y_pred))
    score=cross_val_score(classifier,X_train,y_train)
    print(score)
    print(score.mean())
    


# both Xg boost and random forest gives the best results
