#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[88]:


from sklearn.model_selection import KFold, cross_val_score, train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from  sklearn.preprocessing  import StandardScaler


# In[35]:


# read the dataset
train = pd.read_csv(r"F:\Dataset\Big Mart Sales\train.csv",encoding='utf-8')
test = pd.read_csv(r"F:\Dataset\Big Mart Sales\test.csv",encoding='utf-8')


# In[32]:


train.head()


# In[28]:


train.isnull().sum()


# In[33]:


test.head()


# In[29]:


test.isnull().sum()


# In[37]:


sns.distplot(train["Item_Outlet_Sales"])


# In[38]:


train.groupby('Outlet_Establishment_Year')['Item_Outlet_Sales'].mean().plot.bar()


# In[39]:


train.groupby('Item_Type')['Item_Outlet_Sales'].sum().plot.bar()


# In[40]:


train.groupby('Outlet_Type')['Item_Outlet_Sales'].mean().plot.bar()
train.pivot_table(values='Item_Outlet_Sales',index='Outlet_Type')


# In[42]:


train_id = train.Item_Identifier
test_id = test.Item_Identifier

y_train = train.Item_Outlet_Sales


# In[44]:


train = train.drop(['Item_Outlet_Sales', "Item_Identifier"], axis = 1)
test = test.drop(["Item_Identifier"], axis = 1)


# In[51]:


combined_data = pd.concat([train, test], ignore_index = True)


# In[52]:


combined_data.sample(5)


# In[53]:


sns.countplot(x = "Outlet_Size", data = combined_data)


# In[54]:


combined_data ["Outlet_Size"] = combined_data["Outlet_Size"].fillna((combined_data["Outlet_Size"].mode()[0] ))

combined_data["Item_Fat_Content"] = combined_data["Item_Fat_Content"].replace({"low fat" :"Low Fat","LF" :"Low Fat", "reg" : "Regular"})

sns.countplot(x="Item_Fat_Content", data= combined_data)


# In[55]:


sns.boxplot(x = "Item_Weight", data = combined_data)


# In[56]:


combined_data["Item_Weight"] = combined_data["Item_Weight"].fillna((combined_data["Item_Weight"].mean()))

combined_data.isnull().sum()


# In[57]:


combined_data.head()


# In[58]:


combined_data = pd.get_dummies(combined_data, columns = ["Item_Fat_Content", "Item_Type", "Outlet_Identifier", "Outlet_Establishment_Year","Outlet_Size", "Outlet_Location_Type", "Outlet_Type" ], drop_first = True)
combined_data.head()


# In[60]:


X_train = combined_data[:len(train)]
X_test = combined_data[len(test):]


# In[64]:





# In[66]:


trainX, testX, trainY, testY = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0) 

slc= StandardScaler()
trainX = slc.fit_transform(trainX)
X_test = slc.transform(X_test)
testX = slc.transform(testX)


# In[67]:


num_folds = 10
seed = 0
scoring = 'neg_mean_squared_error'
kfold = KFold(n_splits = num_folds, random_state = seed)


# In[87]:


model = XGBRegressor(n_estimators = 200 , learning_rate = .5)
score_= cross_val_score(model, trainX, trainY, cv = kfold, scoring = scoring)
model.fit(trainX, trainY)
predictions = model.predict(testX)
model.score(trainX, trainY)


# In[75]:


print(r2_score(testY, predictions))
rmse = np.sqrt(mean_squared_error(testY, predictions))

