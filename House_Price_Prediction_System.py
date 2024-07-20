#!/usr/bin/env python
# coding: utf-8

# # House Price Prediction Systme using Data Analytics Algorithms

# ## Business Problem
# 
# Develop a predictive model that can accurately forecast house prices based on various factors.

# ## Importing Libraries and Loading Dataset

# In[1]:


# Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset

df_raw = pd.read_csv("Bengaluru_House_Data.csv")
df_raw.shape


# In[2]:


df_raw.head()


# In[3]:


df_raw.tail()


# ## Data Preprocessing

# ### Get brief information of Dataset

# In[4]:


# Make a copy of raw data

df = df_raw.copy()


# In[5]:


# Get the information of data

df.info()


# In[6]:


# We have only 3 neumerical features - bath, balcony and price
# 6 categorical features - area type, availability, size, society, and total_srft
# Target Feature =======>>>>>> price >>>>>>


# In[7]:


df.describe()


# In[8]:


# 75% and max valueS shows huge difference (may contain outliers)


# In[9]:


sns.pairplot(df)


# In[10]:


# Bath and Price have slightly linear correlation with some outliers


# In[12]:


# Value count of each feature


def value_count(df):
  for var in df.columns:
    print(df[var].value_counts())
    print("--------------------------------")

value_count(df)


# In[13]:


# Correlation heatmap

num_vars = ["bath", "balcony", "price"]
sns.heatmap(df[num_vars].corr(),cmap="coolwarm", annot=True)


# In[14]:


# Correlation of bath is greater with price than balcony


# ### Data Cleaning

# In[15]:


# Find the missing data available

df.isnull().sum()


# In[16]:


# Percentage of missing value

df.isnull().mean()*100


# In[17]:


# Society has 41.3% missing value (need to drop)


# In[18]:


# Visualize missing value using heatmap to get idea where is the value missing

plt.figure(figsize=(16,9))
sns.heatmap(df.isnull())


# In[19]:


# Drop ----> society feature
# because 41.3% missing value (higher percentage)

df2 = df.drop('society', axis='columns')
df2.shape


# In[20]:


# Fill mean value in -----> balcony feature
# because it contain 4.5% missing value

df2['balcony'] = df2['balcony'].fillna(df2['balcony'].mean())
df2.isnull().sum()


# In[21]:


# drop na value rows from df2
# because the % value missing is very low

df3 = df2.dropna()
df3.shape


# In[22]:


df3.isnull().sum()


# In[23]:


df3.head()


# ### Feature Engineering

# In[24]:


# To show all the columns and rows

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


# #### Converting 'total_sqft' categorical feature to numeric

# In[26]:


df3['total_sqft'].value_counts()


# In[27]:


# Here we observe that 'total_sqft' contain string value in diff format
# float, int like value 1752.12,817 
# range value: 2580 - 2591 
# number and string: 142.84Sq. Meter, 1Grounds

# best strategy is to convert it into number by spliting it


# In[28]:


total_sqft_int = []
for str_val in df3['total_sqft']:
  try:
    total_sqft_int.append(float(str_val)) # if '145.6' like this value in str then conver in float
  except:
    try:
      temp = []
      temp = str_val.split('-')
      total_sqft_int.append((float(temp[0])+float(temp[-1]))/2) # '256 - 625' this str value split and take mean
    except:
      total_sqft_int.append(np.nan) # if value not contain in above format then consider as nan


# In[29]:


# Reset the index of dataframe

df4 = df3.reset_index(drop=True)


# In[30]:


# Join df4 and total_srft_int list

df5 = df4.join(pd.DataFrame({'total_sqft_int':total_sqft_int}))
df5.head()


# In[31]:


df5.tail()


# In[32]:


df5.isnull().sum()


# In[33]:


# Drop na value

df6 = df5.dropna()
df6.shape


# In[34]:


df6.info()


# #### Working on size feature

# In[35]:


# Size feature shows the number of rooms 

df6['size'].value_counts()


# In[36]:


# in  size feature we assume that 
# 2 BHK = 2 Bedroom == 2 RK
# so takes only number and remove suffix text

size_int = []
for str_val in df6['size']:
  temp=[]
  temp = str_val.split(" ")
  try:
    size_int.append(int(temp[0]))
  except:
    size_int.append(np.nan)
    print("ABC = ",str_val)


# In[37]:


df6 = df6.reset_index(drop=True)


# In[38]:


# join df6 and list size_int

df7 = df6.join(pd.DataFrame({'bhk':size_int}))
df7.shape


# In[39]:


df7.tail()


# ### Finding and removing Outliers

# In[40]:


# Function to create histogram, Q-Q plot and boxplot

import scipy.stats as stats

def diagnostic_plots(df, variable):   # function takes a dataframe (df) and
                                      

    plt.figure(figsize=(16, 4))  # define figure size

    # histogram
    
    plt.subplot(1, 3, 1)
    sns.distplot(df[variable], bins=30)
    plt.title('Histogram')

    # Q-Q plot
    
    plt.subplot(1, 3, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)
    plt.ylabel('Variable quantiles')

    # boxplot
    
    plt.subplot(1, 3, 3)
    sns.boxplot(y=df[variable])
    plt.title('Boxplot')

    plt.show()


# In[41]:


num_var = ["bath","balcony","total_sqft_int","bhk","price"]
for var in num_var:
  print("******* {} *******".format(var))
  diagnostic_plots(df7, var)

# Here we observe outlier using histogram,, qq plot and boxplot 


# In[42]:


# Here we consider 1 BHK (requierd min 350 sqft)

df7[df7['total_sqft_int']/df7['bhk'] < 350].head()


# In[43]:


# If 1 BHK total_sqft are < 350 then, remove them

df8 = df7[~(df7['total_sqft_int']/df7['bhk'] < 350)]
df8.shape


# In[44]:


# Create new feature that is price per squre foot 
# It help to find the outliers

df8['price_per_sqft'] = df8['price']*100000 / df8['total_sqft_int']  
df8.head()


# In[45]:


df8.price_per_sqft.describe()


# In[46]:


# Here we can see huge difference between min and max price_per_sqft
# min 267.829813, max 176470.588235


# In[47]:


# Removing outliers using help of 'price per sqrt' taking std and mean per location

def remove_pps_outliers(df):
  df_out = pd.DataFrame()
  for key, subdf in df.groupby('location'):
    m=np.mean(subdf.price_per_sqft)
    st=np.std(subdf.price_per_sqft)
    reduced_df = subdf[(subdf.price_per_sqft>(m-st))&(subdf.price_per_sqft<=(m+st))]
    df_out = pd.concat([df_out, reduced_df], ignore_index = True)
  return df_out

df9 = remove_pps_outliers(df8)
df9.shape


# In[48]:


def plot_scatter_chart(df,location):
  bhk2 = df[(df.location==location) & (df.bhk==2)]
  bhk3 = df[(df.location==location) & (df.bhk==3)]
  plt.figure(figsize=(16,9))
  plt.scatter(bhk2.total_sqft_int, bhk2.price, color='Blue', label='2 BHK', s=50)
  plt.scatter(bhk3.total_sqft_int, bhk3.price, color='Red', label='3 BHK', s=50, marker="+")
  plt.xlabel("Total Square Feet Area")
  plt.ylabel("Price")
  plt.title(location)
  plt.legend()

plot_scatter_chart(df9, "Rajaji Nagar")


# In[49]:


# In above scatterplot we observe that at same location price of 2 BHK house is greater than 3 BHK 
# This indicates there is presence of outliers


# In[50]:


plot_scatter_chart(df9, "Hebbal")


# In[51]:


# In above scatterplot we observe that at same location price of 2 BHK house is greater than 3 BHK 
# This indicates there is presence of outliers


# In[52]:


# Removing BHK outliers

def remove_bhk_outliers(df):
  exclude_indices = np.array([])
  for location, location_df in df.groupby('location'):
    bhk_stats = {}
    for bhk, bhk_df in location_df.groupby('bhk'):
      bhk_stats[bhk]={
          'mean':np.mean(bhk_df.price_per_sqft),
          'std':np.std(bhk_df.price_per_sqft),
          'count':bhk_df.shape[0]}
    for bhk, bhk_df in location_df.groupby('bhk'):
      stats=bhk_stats.get(bhk-1)
      if stats and stats['count']>5:
        exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
  return df.drop(exclude_indices, axis='index')

df10 = remove_bhk_outliers(df9)
df10.shape


# In[53]:


plot_scatter_chart(df10, "Hebbal")


# #### Removing outliers using Bath feature

# In[54]:


df10.bath.unique()


# In[55]:


df10[df10.bath > df10.bhk+2]


# In[56]:


# Here we are considering data having total no. bathroom =  bhk + 1

df11 = df10[df10.bath < df10.bhk+2]
df11.shape


# In[57]:


plt.figure(figsize=(16,9))
for i,var in enumerate(num_var):
  plt.subplot(3,2,i+1)
  sns.boxplot(df11[var])


# In[58]:


df11.head()


# In[59]:


df12 = df11.drop(['area_type', 'availability',"location","size","total_sqft"], axis =1)
df12.head()


# In[60]:


df12.to_csv("clean_data.csv", index=False)


# ### Categorical Features Encoding

# In[61]:


df13 = df11.drop(["size","total_sqft"], axis =1)
df13.head()


# In[62]:


df14 = pd.get_dummies(df13, drop_first=True, columns=['area_type','availability','location'])
df14.shape


# In[63]:


df14.head()


# In[64]:


df14.to_csv('oh_encoded_data.csv', index=False) # test ML model on this data

In ['area_type','availability','location'] contain multiple classes and if we convert them into One Hot Encoding so it increase the size of DF 
So try to use those classes which are *frequently* present
# #### Working on area_type feature

# In[66]:


df13['area_type'].value_counts()


# In[67]:


df15 = df13.copy()

# appy One-Hot encoding on 'area_type' feature

for cat_var in ["Super built-up  Area","Built-up  Area","Plot  Area"]:
  df15["area_type"+cat_var] = np.where(df15['area_type']==cat_var, 1,0)
df15.shape


# In[68]:


df15.head(2)


# #### Working on availability feature

# In[69]:


df15["availability"].value_counts()


# In[70]:


# In availability feature, 5644 house 'Ready to Move" and remaining will be ready on perticuler date
# So we create new feature ""availability_Ready To Move"" and add value 1 if availability is Ready To Move else 0

df15["availability_Ready To Move"] = np.where(df15["availability"]=="Ready To Move",1,0)
df15.shape


# In[71]:


df15.tail()


# #### Working on location feature

# In[72]:


location_value_count = df15['location'].value_counts()
location_value_count


# In[73]:


location_gert_20 = location_value_count[location_value_count>=20].index
location_gert_20


# In[74]:


# location count is greter than 19 then we create column of that feature 
# Then if this location present in location feature then set value 1 else 0 (one hot encoding)

df16 = df15.copy()
for cat_var in location_gert_20:
  df16['location_'+cat_var]=np.where(df16['location']==cat_var, 1,0)
df16.shape


# In[75]:


df16.head()


# ### Drop Categorical features

# In[76]:


df17 = df16.drop(["area_type","availability",'location'], axis =1)
df17.shape


# In[77]:


df17.head()


# In[78]:


df17.to_csv('ohe_data_reduce_cat_class.csv', index=False) 


# In[79]:


df18 = pd.read_csv('ohe_data_reduce_cat_class.csv')
df18.shape


# In[80]:


df18.head()


# ### Spliting dataset in train and test

# In[81]:


X = df18.drop("price", axis=1)
y = df18['price']
print('Shape of X = ', X.shape)
print('Shape of y = ', y.shape)


# In[82]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 51)
print('Shape of X_train = ', X_train.shape)
print('Shape of y_train = ', y_train.shape)
print('Shape of X_test = ', X_test.shape)
print('Shape of y_test = ', y_test.shape)


# ## Feature Scaling

# In[83]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train= sc.transform(X_train)
X_test = sc.transform(X_test)


# ## Model Training

# ### Linear Regression

# In[84]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
lr = LinearRegression()
lr_lasso = Lasso()
lr_ridge = Ridge()


# In[85]:


def rmse(y_test, y_pred):
  return np.sqrt(mean_squared_error(y_test, y_pred))


# In[86]:


lr.fit(X_train, y_train)
lr_score = lr.score(X_test, y_test)
lr_rmse = rmse(y_test, lr.predict(X_test))
lr_score, lr_rmse


# ### Lasso

# In[87]:


lr_lasso.fit(X_train, y_train)
lr_lasso_score=lr_lasso.score(X_test, y_test)
lr_lasso_rmse = rmse(y_test, lr_lasso.predict(X_test))
lr_lasso_score, lr_lasso_rmse


# ### Support Vector Machine

# In[88]:


from sklearn.svm import SVR
svr = SVR()
svr.fit(X_train,y_train)
svr_score=svr.score(X_test,y_test)
svr_rmse = rmse(y_test, svr.predict(X_test))
svr_score, svr_rmse


# ### Random Forest Regression

# In[89]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(X_train,y_train)
rfr_score=rfr.score(X_test,y_test)
rfr_rmse = rmse(y_test, rfr.predict(X_test))
rfr_score, rfr_rmse


# ### XGBoost

# In[90]:


import xgboost
xgb_reg = xgboost.XGBRegressor()
xgb_reg.fit(X_train,y_train)
xgb_reg_score=xgb_reg.score(X_test,y_test) # with 0.8838865742273464
xgb_reg_rmse = rmse(y_test, xgb_reg.predict(X_test))
xgb_reg_score, xgb_reg_rmse


# In[91]:


print(pd.DataFrame([{'Model': 'Linear Regression','Score':lr_score, "RMSE":lr_rmse},
              {'Model': 'Lasso','Score':lr_lasso_score, "RMSE":lr_lasso_rmse},
              {'Model': 'Support Vector Machine','Score':svr_score, "RMSE":svr_rmse},
              {'Model': 'Random Forest','Score':rfr_score, "RMSE":rfr_rmse},
              {'Model': 'XGBoost','Score':xgb_reg_score, "RMSE":xgb_reg_rmse}],
             columns=['Model','Score','RMSE']))


# ## Cross Validation

# In[92]:


from sklearn.model_selection import KFold,cross_val_score
cvs = cross_val_score(xgb_reg, X_train,y_train, cv = 10)
cvs, cvs.mean()


# In[93]:


cvs_rfr = cross_val_score(rfr, X_train,y_train, cv = 10)
cvs_rfr, cvs_rfr.mean()


# In[94]:


from sklearn.model_selection import cross_val_score
cvs_rfr2 = cross_val_score(RandomForestRegressor(), X_train,y_train, cv = 10)
cvs_rfr2, cvs_rfr2.mean()


# ## Hyper Parameter Tuning

# In[95]:


from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBRegressor

# Various parameters to tune

xgb1 = XGBRegressor()
parameters = {'learning_rate': [0.1,0.03, 0.05, 0.07],
              'min_child_weight': [1,3,5],
              'max_depth': [4, 6, 8],
              'gamma':[0,0.1,0.001,0.2],
              'subsample': [0.7,1,1.5],
              'colsample_bytree': [0.7,1,1.5],
              'objective':['reg:linear'],
              'n_estimators': [100,300,500]}

xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 2,
                        n_jobs = -1,
                        verbose=True)

xgb_grid.fit(X_train, y_train)

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)


# In[96]:


xgb_tune = xgb_grid.estimator

xgb_tune.fit(X_train,y_train)
xgb_tune.score(X_test,y_test)


# In[97]:


cvs = cross_val_score(xgb_tune, X_train,y_train, cv = 10)
cvs, cvs.mean()


# In[98]:


from xgboost import XGBRegressor

# Initialize the XGBRegressor with updated parameters
xgb_tune2 = XGBRegressor(
    base_score=0.5,
    booster='gbtree',
    colsample_bylevel=1,
    colsample_bynode=0.6,
    colsample_bytree=1,
    gamma=0,
    importance_type='gain',
    learning_rate=0.25,
    max_delta_step=0,
    max_depth=4,
    min_child_weight=1,
    n_estimators=400,
    n_jobs=1,
    objective='reg:squarederror',  # Updated from 'reg:linear'
    random_state=0,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    subsample=1,
    verbosity=1
)

# Fit the model to the training data
xgb_tune2.fit(X_train, y_train)

# Evaluate the model on the test data
score = xgb_tune2.score(X_test, y_test)
print("Model score:", score)


# In[100]:


from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
parameters = {
    'learning_rate': [0.1, 0.03, 0.05, 0.07],  # Learning rate values
    'min_child_weight': [1, 3, 5],  # Minimum sum of weights of all observations required in a child
    'max_depth': [4, 6, 8],  # Maximum depth of a tree
    'gamma': [0, 0.1, 0.001, 0.2],  # Minimum loss reduction required to make a split
    'subsample': [0.7, 1],  # Fraction of observations to be randomly sampled for each tree
    'colsample_bytree': [0.7, 1],  # Fraction of columns to be randomly sampled for each tree
    'objective': ['reg:squarederror'],  # Loss function to be minimized
    'n_estimators': [100, 300, 500]  # Number of boosting rounds
}

# Initialize the XGBRegressor
xgb_tune2 = XGBRegressor(
    base_score=0.5,
    booster='gbtree',
    colsample_bylevel=1,
    colsample_bynode=0.9,
    colsample_bytree=1,
    gamma=0,
    importance_type='gain',
    learning_rate=0.05,
    max_delta_step=0,
    max_depth=4,
    min_child_weight=5,
    n_estimators=100,
    n_jobs=1,
    objective='reg:squarederror',  # Updated from 'reg:linear'
    random_state=0,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    subsample=1,
    verbosity=1
)

# Fit the model to the training data
xgb_tune2.fit(X_train, y_train)

# Evaluate the model on the test data
score = xgb_tune2.score(X_test, y_test)
print("Model score:", score)

# Perform GridSearchCV to find the best parameters
grid_search = GridSearchCV(estimator=xgb_tune2, param_grid=parameters, cv=5, scoring='r2', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters from GridSearchCV
print("Best parameters found: ", grid_search.best_params_)

# Best score from GridSearchCV
print("Best score found: ", grid_search.best_score_)


# In[101]:


cvs = cross_val_score(xgb_tune2, X_train,y_train, cv = 5)
cvs, cvs.mean()


# In[102]:


np.sqrt(mean_squared_error(y_test, xgb_tune2.predict(X_test)))


# ## Test Model

# In[103]:


list(X.columns)


# In[104]:


# Predict value of hosue by providing feature values 

def predict_house_price(model,bath,balcony,total_sqft_int,bhk,price_per_sqft,area_type,availability,location):

  x =np.zeros(len(X.columns)) # create zero numpy array

  # adding feature's value according to their column index
  x[0]=bath
  x[1]=balcony
  x[2]=total_sqft_int
  x[3]=bhk
  x[4]=price_per_sqft

  if "availability"=="Ready To Move":
    x[8]=1

  if 'area_type'+area_type in X.columns:
    area_type_index = np.where(X.columns=="area_type"+area_type)[0][0]
    x[area_type_index] =1

    print(area_type_index)

  if 'location_'+location in X.columns:
    loc_index = np.where(X.columns=="location_"+location)[0][0]
    x[loc_index] =1

    print(loc_index)

  print(x)

  # feature scaling
  x = sc.transform([x])[0]
  print(x)

  return model.predict([x])[0] # return the predicted value by train XGBoost model


# In[105]:


predict_house_price(model=xgb_tune2, bath=3,balcony=2,total_sqft_int=1672,bhk=3,price_per_sqft=8971.291866,area_type="Plot  Area",availability="Ready To Move",location="Devarabeesana Halli")


# In[106]:


## test sample
# area_type	availability	location	bath	balcony	price	total_sqft_int	bhk	price_per_sqft
# 2	Super built-up Area	Ready To Move	Devarabeesana Halli	3.0	2.0	150.0	1750.0	3	8571.428571

predict_house_price(model=xgb_tune2, bath=3,balcony=2,total_sqft_int=1750,bhk=3,price_per_sqft=8571.428571,area_type="Super built-up",availability="Ready To Move",location="Devarabeesana Halli")


# In[107]:


## test sample
# area_type	availability	location	bath	balcony	price	total_sqft_int	bhk	price_per_sqft
# 1	Built-up Area	Ready To Move	Devarabeesana Halli	3.0	3.0	149.0	1750.0	3	8514.285714
predict_house_price(model=xgb_tune2,bath=3,balcony=3,total_sqft_int=1750,bhk=3,price_per_sqft=8514.285714,area_type="Built-up Area",availability="Ready To Move",location="Devarabeesana Halli")


# ## Save and Load Model

# In[109]:


import joblib

# save model

joblib.dump(xgb_tune2, 'house_price_prediction_model.pkl')
joblib.dump(rfr, 'house_price_prediction_rfr_model.pkl')


# In[110]:


# Load Model
house_price_prediction_model = joblib.load("house_price_prediction_model.pkl")


# In[111]:


# Predict house price
predict_house_price(house_price_prediction_model,bath=3,balcony=3,total_sqft_int=150,bhk=3,price_per_sqft=8514.285714,area_type="Built-up Area",availability="Ready To Move",location="Devarabeesana Halli")


# In[ ]:




