#!/usr/bin/env python
# coding: utf-8

# # importing all the libraries

# In[36]:


import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style("whitegrid")
import matplotlib.pyplot as plt
import sklearn


# # importing dataset

# In[12]:


data=pd.read_csv('matches1234.csv')


# # analyzing the first 5 rows of the dataset

# In[11]:


data.head()


# # brief summary of the IPL dataset

# In[13]:


data.describe()


# # checking null values present in the dataset

# In[15]:


data.dropna(inplace=True)
data.isnull().sum()


# In[16]:


data["team1"].unique()


# # changing old name of teams

# In[17]:


data['team1']=data['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
data['team2']=data['team2'].str.replace('Delhi Daredevils','Delhi Capitals')
data['winner']=data['winner'].str.replace('Delhi Daredevils','Delhi Capitals')


# In[18]:


data['team1']=data['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
data['team2']=data['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
data['winner']=data['winner'].str.replace('Deccan Chargers','Sunrisers Hyderabad')


# # visualizations

# In[19]:


plt.figure(figsize = (10,6))
sns.countplot(y = 'winner',data = data,order= data['winner'].value_counts().index)
plt.xlabel('Wins')
plt.ylabel('Team')
plt.title('Number of  IPL  matches won by each team')


# In[20]:


plt.figure(figsize = (10,6))
sns.countplot(y = 'venue',data = data,order = data['venue'].value_counts().iloc[:10].index)
plt.xlabel('No of matches',fontsize=12)
plt.ylabel('Venue',fontsize=12)
plt.title('Total Number of matches played in different stadium')


# In[21]:


plt.figure(figsize = (10,6))
sns.countplot(x = "toss_decision", data=data)
plt.xlabel('Toss Decision',fontsize=12)
plt.ylabel('Count',fontsize=12)
plt.title('Toss Decision')


# # dropping needless features

# In[25]:


data.drop([ "season","city", 'umpire1', "venue"], axis=1, inplace=True)


# # converting data into dependent and independent variable

# In[26]:


X = data.drop(["winner"], axis=1)
y = data["winner"]


# # converting categorical values into numerical values

# In[28]:


X = pd.get_dummies(X, ["team1","team2", "toss_winner", "toss_decision"], drop_first = True)


# In[29]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# # split data into test and train set

# In[30]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)


# # model creation and evaluation

# In[31]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=200,min_samples_split=3,
                               max_features = "auto")


# In[32]:


model.fit(x_train, y_train)


# In[33]:


y_pred = model.predict(x_test)


# In[34]:


from sklearn.metrics import accuracy_score
ac = accuracy_score(y_pred, y_test)


# In[35]:


ac

