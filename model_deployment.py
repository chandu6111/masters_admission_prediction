#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import warnings
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
warnings.filterwarnings("ignore")
import math



# In[2]:


df=pd.read_csv("datasetforProject.csv")
df=np.array(df)


# In[3]:



# In[4]:


# In[5]:


X= df.iloc[:,1:-1].values
y = df.iloc[:,-1].values


# In[6]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# In[7]:


sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)


# In[8]:




# In[9]:


regressor=RandomForestRegressor(n_estimators=100,random_state=0)


# In[10]:


regressor.fit(X_train,y_train)


# In[11]:


y_pred=regressor.predict(final_features)


# In[12]:


print(mean_absolute_error(y_test,y_pred))
m=mean_squared_error(y_test,y_pred)
rm=np.sqrt(m)
print(m)
print(rm)
print(r2_score(y_test,y_pred))


# In[13]:


print(y_pred)


# In[14]:


#import pickle 


# In[15]:
with open("regressor.pkl","wb") as f:
    pickle.dump(regressor,f)

#pickle.dump(regressor,open("regressor.pkl","wb"))
model=pickle.load(open("regressor.pkl","rb"))


# In[ ]:




