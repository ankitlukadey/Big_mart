#!/usr/bin/env python
# coding: utf-8

# In[264]:



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import warnings # Ignores any warning
warnings.filterwarnings("ignore")
import math


# In[265]:


#loading the data
train=pd.read_csv("C:\\Users\\ankit\\Desktop\\Internship project\\Train_UWu5bXk (2) (1) (1) (1).csv")
test=pd.read_csv("C:\\Users\\ankit\\Desktop\\Internship project\\Test_u94Q5KV (2) (1) (1) (1).csv")


# In[266]:


train.shape


# In[267]:


test.shape


# In[268]:


#concatening the train and test data
data=pd.concat([train,test],ignore_index=True)
data.shape


# In[269]:


data.info()


# In[270]:


data.describe()


# In[271]:


#checking sum of null values in each column
data.isnull().sum()


# In[272]:


#checking for categorical variables
cat_col=[]
for x in data.dtypes.index:
    if data.dtypes[x]=='object':
        cat_col.append(x)
print(cat_col)


# In[273]:


cat_col.remove('Item_Identifier')
cat_col.remove('Outlet_Identifier')
cat_col


# In[274]:


for col in cat_col:
    print(col)
    print(data[col].value_counts())
    print()


# In[275]:



item_weight_mean=data.pivot_table(values='Item_Weight',index="Item_Identifier")
item_weight_mean


# In[276]:


miss_bool=data['Item_Weight'].isnull()
for i,item in enumerate(data['Item_Identifier']):
    if miss_bool[i]:
        if item in item_weight_mean:
            data['Item_Weight'][i]=item_weight_mean.loc[item]['Item_Weight']
        else:
            data['Item_Weight'][i]=np.mean(data['Item_Weight'])


# In[277]:


data['Item_Weight'].isnull().sum()


# In[278]:


outlet_size_mode=data.pivot_table(values='Outlet_Size',columns='Outlet_Type',aggfunc=(lambda x:x.mode()[0]))
outlet_size_mode


# In[279]:


miss_bool=data['Outlet_Size'].isnull()
data.loc[miss_bool,'Outlet_Size']=data.loc[miss_bool,'Outlet_Type'].apply(lambda x:outlet_size_mode[x])


# In[280]:


data['Outlet_Size'].isnull().sum()


# In[281]:


data.loc[:,"Item_Visibility"].replace([0],[data["Item_Visibility"].mean()],inplace=True)


# In[282]:


sum(data["Item_Visibility"]==0)


# In[283]:


#Label encoding
data['Item_Fat_Content']=data['Item_Fat_Content'].replace({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'})


# In[284]:


data['Item_Fat_Content'].value_counts()


# In[285]:


data['New_Item_Type']=data['Item_Identifier'].apply(lambda x:x[:2])
data['New_Item_Type']=data['New_Item_Type'].replace({'FD':'Food','DR':'Drinks','NC':'Non-Consumable'})


# In[286]:


data['New_Item_Type'].value_counts()


# In[287]:


data['Outlet_Year']=2013-data['Outlet_Establishment_Year']


# In[288]:


data['Outlet_Year']


# In[289]:


#plotting
sns.distplot(data['Item_Weight'])


# In[290]:


sns.distplot(data['Item_Visibility'])


# In[291]:


sns.distplot(data['Item_MRP'])


# In[292]:


sns.distplot(data['Item_Outlet_Sales'])


# In[293]:


sns.countplot(data['Item_Fat_Content'])


# In[294]:


plt.figure(figsize=(20,8))
sns.countplot(data['Item_Type'])


# In[295]:


sns.countplot(data['Outlet_Establishment_Year'])


# In[296]:


sns.countplot(data['Outlet_Size'])


# In[297]:


sns.countplot(data['Outlet_Location_Type'])


# In[298]:


plt.figure(figsize=(8,5))
sns.countplot(data['Outlet_Type'])


# In[299]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Outlet']=le.fit_transform(data['Outlet_Identifier'])
cat_col=['Item_Fat_Content','Item_Type','Outlet_Size','Outlet_Location_Type','Outlet_Type','New_Item_Type']
for col in cat_col:
    data[col]=le.fit_transform(data[col])


# In[300]:


data=pd.get_dummies(data,columns=['Item_Fat_Content','Outlet_Size','Outlet_Location_Type','Outlet_Type','New_Item_Type'])


# In[301]:


data.head()


# In[302]:


#Splitting the data in test and train 
data_train=data.iloc[:8523,:]
data_test=data.iloc[8523:,:]


# In[303]:


X=data_test.drop(columns=['Outlet_Establishment_Year','Item_Identifier','Outlet_Identifier','Item_Outlet_Sales'])


# In[304]:


data_train.shape,data_test.shape


# In[305]:


x=data_train.drop(columns=['Outlet_Establishment_Year','Item_Identifier','Outlet_Identifier','Item_Outlet_Sales'])
y=data_train['Item_Outlet_Sales']


# In[306]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[307]:


#Model Tranning
import math
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
def train(model,x_train,y_train):
    model.fit(x_train,y_train)
    pred=model.predict(x_test)
 
    cv_score=cross_val_score(model,x_train,y_train,scoring='neg_mean_squared_error')
    cv_score=np.abs(np.mean(cv_score))
    print("RMSE: ",math.sqrt(mean_squared_error(y_test,pred)))
    print("CV Score: ",cv_score)


# In[308]:


from sklearn.linear_model import LinearRegression,Ridge,Lasso
model=LinearRegression(normalize=True)
train(model,x_train,y_train)
coef=pd.Series(model.coef_,x_train.columns).sort_values()
coef.plot(kind='bar',title='model coefficients')


# In[309]:


model=Lasso()
train(model,x_train,y_train)
coef=pd.Series(model.coef_,x_train.columns).sort_values()
coef.plot(kind='bar',title='model coefficients')


# In[310]:


model=Ridge(normalize=True)
train(model,x_train,y_train)
coef=pd.Series(model.coef_,x_train.columns).sort_values()
coef.plot(kind='bar',title='model coefficients')


# In[311]:


from xgboost import XGBRegressor
regressor=XGBRegressor()
regressor.fit(x_train,y_train)
pred=regressor.predict(x_train)
print("RMSE: ",math.sqrt(mean_squared_error(y_train,pred)))
cv_score=cross_val_score(model,x_train,y_train,scoring='neg_mean_squared_error')
cv_score=np.abs(np.mean(cv_score))
print("CV Score: ",cv_score)


# In[312]:


from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor()
train(model,x_train,y_train)
coef=pd.Series(model.feature_importances_,x_train.columns).sort_values()
coef.plot(kind='bar',title='feature imporatnce')


# In[313]:


from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
train(model,x_train,y_train)
coef=pd.Series(model.feature_importances_,x_train.columns).sort_values()
coef.plot(kind='bar',title='feature imporatnce')


# In[314]:


#Using xgb regressor as it has low RMSE
sub=pd.read_csv("C:\\Users\\ankit\\Desktop\\Internship project\\SampleSubmission_TmnO39y (2) (1) (1) (1) (1).csv")
model=XGBRegressor()
model.fit(x_train,y_train)
pred=model.predict(X)
sub['Item_Outlet_Sales']=pred
sub.head(10)


# In[315]:


#Creating output csv file  
sub.to_csv("My_new_submission.csv")

