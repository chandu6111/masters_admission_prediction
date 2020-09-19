import numpy as np
import pandas as pd
df=pd.read_csv("datasetforProject.csv")
from sklearn.model_selection import GridSearchCV,cross_validate
X= df.iloc[:,1:-1].values
y = df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

from sklearn.ensemble import RandomForestRegressor
reg=RandomForestRegressor(n_estimators=100,random_state=0)
reg.fit(X_train,y_train)
y_pred=reg.predict(X_test)


import pickle 
with open("model.pkl","wb") as f:
    pickle.dump(reg,f)
lr_model=pickle.load(open("model.pkl","rb"))