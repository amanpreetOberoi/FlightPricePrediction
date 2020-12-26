#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


from google.colab import files
uploaded = files.upload()


# In[ ]:


import io
df_train = pd.read_excel(io.BytesIO(uploaded['Data_Train.xlsx']))
# Dataset is now stored in a Pandas Dataframe


# In[ ]:


from google.colab import files
uploaded1 = files.upload()


# In[ ]:


import io
df_test = pd.read_excel(io.BytesIO(uploaded1['Test_set.xlsx']))
# Dataset is now stored in a Pandas Dataframe


# In[ ]:


df_train=pd.read_excel('Data_Train.xlsx')
df_test=pd.read_excel('Test_set.xlsx')


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


df=df_train.append(df_test,sort=False)


# In[ ]:


df.columns


# ### EDA or Feature Engineering

# In[ ]:


### Whenever we have features like date----->we need to split it into day month and yr


df['Journey_Date']=df['Date_of_Journey'].str.split('/').str[0]


# In[ ]:


df['Journey_Month']=df['Date_of_Journey'].str.split('/').str[1]


# In[ ]:


df['Journey_Year']=df['Date_of_Journey'].str.split('/').str[2]


# In[ ]:


df.head()


# In[ ]:


df.dtypes


# In[ ]:


### newly created features are of object type we need to convert this into int....
df['Journey_Date']=df['Journey_Date'].astype(int)
df['Journey_Month']=df['Journey_Month'].astype(int)
df['Journey_Year']=df['Journey_Year'].astype(int)


# In[ ]:


df.dtypes


# In[ ]:


df.drop(['Date_of_Journey'],inplace=True,axis=1)


# In[ ]:


df.head()


# In[ ]:


### Arrival time---In some date and month is also given but we don't want because he have that info in Journey_Date month and Year


df['Arrival_Time']=df['Arrival_Time'].str.split(' ').str[0]


# In[ ]:


df.head()


# In[ ]:



df[df['Total_Stops'].isnull()]


# In[ ]:


### Total stop and route is null in above recor So we will assume that their is only one stop from Delhi to Cochin

df['Total_Stops'].fillna('1 stop',inplace=True)


# In[ ]:


df.isnull().sum()


# In[ ]:


df['Total_Stops']=df['Total_Stops'].replace("non-stop","0 stop")
df['Total_Stops']=df['Total_Stops'].str.split(' ').str[0]


# In[ ]:


df.head()


# In[ ]:


df.dtypes


# In[ ]:


df['Total_Stops']=df['Total_Stops'].astype(int)


# In[ ]:


df.dtypes


# In[ ]:


df['Arrival_Hour']=df['Arrival_Time'].str.split(':').str[0]
df['Arrival_Minute']=df['Arrival_Time'].str.split(':').str[1]


# In[ ]:


df['Departure_Hour']=df['Dep_Time'].str.split(':').str[0]
df['Departure_Minute']=df['Dep_Time'].str.split(':').str[1]


# In[ ]:


df.head()


# In[ ]:


df.dtypes


# In[ ]:


### now convert data type of above newly created features to int
df['Arrival_Hour']=df['Arrival_Hour'].astype(int)
df['Arrival_Minute']=df['Arrival_Minute'].astype(int)
df['Departure_Hour']=df['Departure_Hour'].astype(int)
df['Departure_Minute']=df['Departure_Minute'].astype(int)


# In[ ]:


df.dtypes


# In[ ]:


df.drop(['Arrival_Time'],inplace=True,axis=1)
df.drop(['Dep_Time'],inplace=True,axis=1)


# In[ ]:


df.head()


# In[ ]:


### Information of Route is same as total Stop So we will dropnthis column

df.drop(['Route'],inplace=True,axis=1)


# In[ ]:


# Time taken by plane to reach destination is called Duration
# It is the differnce betwwen Departure Time and Arrival time
# Assigning and converting Duration column into list
duration = list(df["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i]+ " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour


# In[ ]:


df['Duration']=duration


# In[ ]:


DurationHour=df['Duration'].str.split(' ').str[0]
DurationMinute=df['Duration'].str.split(' ').str[1]


# In[ ]:


df['Duration_Hour']=DurationHour.str.split("h").str[0]
df['Duration_Minute']=DurationMinute.str.split("m").str[0]


# In[ ]:


df['Duration_Hour']=df['Duration_Hour'].astype(int)
df['Duration_Minute']=df['Duration_Minute'].astype(int)


# In[ ]:


df.head()


# In[ ]:


df.drop(['Duration'],inplace=True,axis=1)


# In[ ]:


### Now we will convert Categorial Value using labelEncoding...Assigns value according to priority

from sklearn.preprocessing import LabelEncoder

encode=LabelEncoder()

df['Airline']=encode.fit_transform(df['Airline'])
df['Source']=encode.fit_transform(df['Source'])
df['Destination']=encode.fit_transform(df['Destination'])
df['Additional_Info']=encode.fit_transform(df['Additional_Info'])


# In[ ]:


df.head()


# ### Feature Selection

# In[ ]:


from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel


# In[ ]:


df_train=df[:10683]
df_test=df[10683:]


# In[ ]:


X=df_train.drop(['Price'],axis=1)
y=df_train['Price']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# In[ ]:


model=SelectFromModel(Lasso(alpha=0.005,random_state=0))


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


model.get_support() ## to check which features get selected by SelectFromModel


# In[ ]:


selectedFeatures=X_train.columns[model.get_support()]


# In[ ]:


selectedFeatures


# In[ ]:


## now drop all columns are are not selected from X_train and X_test
X_train.drop(['Journey_Year'],axis=1,inplace=True)
X_test.drop(['Journey_Year'],axis=1,inplace=True)


# In[ ]:


X_train


# ### Apply ML model

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10,15,100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4,10]
# Method of selecting samples for training each tree
bootstrap = [True, False]


# In[ ]:


random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()


# In[ ]:


rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 5, random_state=42)


# In[ ]:


rf_random.fit(X_train,y_train)


# In[ ]:


y_pred=rf_random.predict(X_test)


# In[ ]:


import seaborn as sns

sns.distplot(y_test-y_pred)


# In[ ]:


from sklearn.metrics import mean_squared_error
print(rf_random.score(X_train,y_train))
print(rf_random.score(X_test,y_test))


# In[ ]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[73]:


import pickle
file = open('FlightPricePrediction.pkl', 'wb')
pickle.dump(rf_random, file)


# In[79]:



model = open('FlightPricePrediction.pkl','rb')
result = pickle.load(model)


# In[ ]:


y_prediction = result.predict(X_test)


# In[ ]:


metrics.r2_score(y_test, y_prediction)


# In[74]:





# In[76]:





# In[ ]:




