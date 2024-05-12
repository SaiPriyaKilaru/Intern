#!/usr/bin/env python
# coding: utf-8

# In[89]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[90]:


df=pd.read_csv('happyness csv')


# In[91]:


df


# Dataset has 158 Rows and 12 Columns

# In[92]:


df.isnull().sum()


# There is no missing values in the above dataset

# In[93]:


df['Region'].value_counts()


# In[94]:


df.describe()


# In[95]:


df.info()


# In[96]:


sns.pairplot(df)


# In[97]:


df.corr()


# In[98]:


sns.boxplot(df['Standard Error'])


# In[99]:


sns.boxplot(df['Economy (GDP per Capita)'])


# In[100]:


sns.boxplot(df['Family'])


# In[101]:


sns.boxplot(df['Health (Life Expectancy)'])


# In[102]:


sns.boxplot(df['Freedom'])


# In[103]:


sns.boxplot(df['Trust (Government Corruption)'])


# In[104]:


sns.boxplot(df['Generosity'])


# In[105]:


sns.boxplot(df['Dystopia Residual'])


# There are some outliers are present in Trust (Government Corruption) and Standard Error
# we can remove by using Z score 

# In[107]:


a=(df['Region']).value_counts()


# In[108]:


plt.figure(figsize=(12, 8))
sns.countplot(x='Region', data=df, palette='viridis')
plt.title('Count of Happiness Scores')
plt.xlabel('Happiness Score')
plt.ylabel('Count')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()


# In[109]:


from scipy.stats import zscore


# In[110]:


zscore=zscore(df['Standard Error'])


# In[111]:


zscore


# In[112]:


outliers=abs(zscore)>3.0


# In[113]:


print(outliers)


# In[114]:


z_scores = (df['Trust (Government Corruption)'] - df['Trust (Government Corruption)'].mean()) / df['Trust (Government Corruption)'].std()


# In[115]:


print(outliers)


# In[116]:


threshold = 3.0
outliers = df[abs(z_scores) > threshold]


# In[117]:


outliers.head(28)


# In[118]:


df.drop(columns='Happiness Rank',inplace=True,axis=1)


# In[119]:


df=df.drop([27,153])


# In[120]:


df.head()


# In[121]:


df.corr()


# As per the corelation table Happiness Score is Highly positively corelated with the Economy,family,health, and moderately corelated with Dystopia Resudual ,freedom and Trust.
# Below figues shows the analytical graphs

# In[122]:


sns.set(style="ticks")
plt.figure(figsize=(10, 6))
sns.lmplot(x='Happiness Score', y='Freedom', data=df, height=6)
plt.title('Relationship between Happiness Score and Freedom')
plt.xlabel('Happiness Score')
plt.ylabel('Freedom)')
plt.grid(True)
plt.show()


# In[123]:


sns.set(style="ticks")
plt.figure(figsize=(10, 6))
sns.lmplot(x='Happiness Score', y='Trust (Government Corruption)', data=df, height=6)
plt.title('Relationship between Happiness Score and Trust (Government Corruption)')
plt.xlabel('Happiness Score')
plt.ylabel('Trust (Government Corruption))')
plt.grid(True)
plt.show()


# In[124]:


sns.set(style='ticks')
plt.figure(figsize=(10,6))
sns.lmplot(x='Happiness Score',y='Dystopia Residual',data=df,height=6)
plt.title('Relationship between Happiness Score and Dystopia Residual')
plt.xlabel('Happiness Score')
plt.ylabel('Dystopia Residual')
plt.grid(True)
plt.show()


# In[125]:


sns.set(style='ticks')
plt.figure(figsize=(10,6))
sns.lmplot(x='Happiness Score',y='Health (Life Expectancy)',data=df,height=6)
plt.title('Relationship between Happiness Score and Health (Life Expectancy)')
plt.xlabel('Happiness Score')
plt.ylabel('Health (Life Expectancy)')
plt.grid(True)
plt.show()


# In[126]:


sns.set(style='ticks')
plt.figure(figsize=(10,6))
sns.lmplot(x='Happiness Score',y='Family',data=df,height=6)
plt.xlabel('Happiness Score')
plt.ylabel('Family')
plt.grid(True)
plt.show()


# In[127]:


region_happiness = df.groupby('Region')['Happiness Score'].mean().sort_values()
plt.figure(figsize=(10, 6))
region_happiness.plot(kind='bar', color='skyblue')
plt.title('Average Happiness Score by Region')
plt.xlabel('Region')
plt.ylabel('Average Happiness Score')
plt.show()


# In[128]:


plt.figure(figsize=(12, 8))
sns.violinplot(x='Happiness Score', y='Region', data=df)
plt.title('Violin Plot of Happiness Score by Region')
plt.xlabel('Happiness Score')
plt.ylabel('Region')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()


# as from the above figures region also effect the happiness score so we use annova test 

# In[129]:


from scipy.stats import f_oneway
grouped_data = []
for name, group in df.groupby('Region'):
    grouped_data.append(group['Happiness Score'])
f_statistic, p_value = f_oneway(*grouped_data)
print("F-statistic:", f_statistic)
print("p-value:", p_value)


# i am dropping the  country column from the data frame because region is in consideration

# In[130]:


X=df.drop(columns='Happiness Score')
y=df['Happiness Score']


# In[131]:


X.drop(columns='Country',inplace=True,axis=1)
X


# In[132]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[133]:


X['Region']=le.fit_transform(X['Region'])


# In[136]:


df['Region']=le.fit_transform(df['Region'])


# In[137]:


df.corr()


# In[147]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import LinearRegression


# In[149]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# In[148]:


l=LinearRegression()


# In[150]:


l.fit(X_train,y_train)


# In[151]:


y_pred=l.predict(X_test)


# In[152]:


score=r2_score(y_test,y_pred)


# In[153]:


score


# In[154]:


a=mean_squared_error(y_test,y_pred)


# In[155]:


a


# In[190]:


from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import roc_auc_score


# In[158]:


r=Ridge()


# In[159]:


r.fit(X_train,y_train)


# In[160]:


y_r=r.predict(X_test)


# In[162]:


rscore=r2_score(y_r,y_test)
print(rscore)


# In[189]:


a=mean_squared_error(y_test,y_r)
a


# In[ ]:





# In[177]:


from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.ensemble import RandomForestRegressor


# In[178]:


model = RandomForestRegressor()


# In[179]:


param_grid={
    'n_estimators':[40,80,120,150],
    'max_depth':[None,5,10],
    'min_samples_split':[2,5,10]
}


# In[182]:


grid=GridSearchCV(estimator=model,param_grid=param_grid,cv=10)
grid.fit(X_train,y_train)


# In[183]:


grid.best_params_


# In[186]:


cross_val_scores = cross_val_score(grid.best_estimator_, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cross_val_scores


# as from the above we can conclude that mean squared error is less therefore effeciency of this model is around 98%

# In[ ]:





# In[ ]:





# In[ ]:




