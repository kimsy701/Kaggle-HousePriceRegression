#!/usr/bin/env python
# coding: utf-8

# ## 1. DATA EDA

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# 1.1 Variables
# 1.1.1 Numeric Variable
# 1.1.2 Categoric Variable(including date variables)
# 
# 1.2 Target Variable

# In[3]:


#bring data
df = pd.read_csv("C:/Users/KIM/Desktop/자기개발/kaggle competition/house-prices-advanced-regression-techniques/train.csv", index_col = 0)


# In[4]:


df


# In[5]:


#1.1 Variables
#1.1.1 Numeric Variables
"""
LotFrontage, LotArea, MasVnrArea, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, (6)
TotalBsmtSF, 1stFlrSF, 2ndFlrSF, LowQualFinSF, GrLivArea, BsmtFullBath,(6)
BsmtHalfBath, FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, (6)
Fireplaces, GarageYrBlt, GarageCars, GarageArea, WoodDeckSF, (5)
OpenPorchSF,EnclosedPorch,3SsnPorch,ScreenPorch,PoolArea, (5)
MiscVal(1)
"""
numerical = [
'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 
'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 
'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea', 'MiscVal'  
]
df[numerical].hist(bins=15, figsize=(30, 20), layout=(6, 5))


# In[6]:


#column that only has "0"

zero_column = df[numerical].sum()
zero_column


# In[7]:


#1.1.2 Categorical Variables 
categorical1 = ['MSSubClass','MSZoning','Street','Alley','LotShape','LandContour','Utilities']
categorical2 = ['LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle']
categorical3 = ['OverallQual','OverallCond','YearBuilt','YearRemodAdd','RoofStyle','RoofMatl','Exterior1st']
categorical4 = ['Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond']
categorical5 = ['BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical']
categorical6 = ['KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond']
categorical7 = ['PavedDrive','PoolQC','Fence','MiscFeature','MoSold','YrSold','SaleType','SaleCondition']

fig, ax = plt.subplots(4,2 , figsize=(15, 10))
for var, subplot in zip(categorical1, ax.flatten()):
    sns.boxplot(x=var, y='SalePrice', data=df, ax=subplot)

fig, ax = plt.subplots(4,2 , figsize=(15, 10))
for var, subplot in zip(categorical2, ax.flatten()):
    sns.boxplot(x=var, y='SalePrice', data=df, ax=subplot)
    
fig, ax = plt.subplots(4,2 , figsize=(15, 10))
for var, subplot in zip(categorical3, ax.flatten()):
    sns.boxplot(x=var, y='SalePrice', data=df, ax=subplot)
    
fig, ax = plt.subplots(4,2 , figsize=(15, 10))
for var, subplot in zip(categorical4, ax.flatten()):
    sns.boxplot(x=var, y='SalePrice', data=df, ax=subplot)
    
fig, ax = plt.subplots(4,2 , figsize=(15, 10))
for var, subplot in zip(categorical5, ax.flatten()):
    sns.boxplot(x=var, y='SalePrice', data=df, ax=subplot)

fig, ax = plt.subplots(4,2 , figsize=(15, 10))
for var, subplot in zip(categorical6, ax.flatten()):
    sns.boxplot(x=var, y='SalePrice', data=df, ax=subplot)
    
fig, ax = plt.subplots(4,2 , figsize=(15, 10))
for var, subplot in zip(categorical7, ax.flatten()):
    sns.boxplot(x=var, y='SalePrice', data=df, ax=subplot)


# In[8]:


#1.2 target variable : SalePrice
df['SalePrice'].hist(bins=15, figsize=(8, 5))


# ## 2. Data Preprocessing

# ### 2.1 Fill nans

# In[9]:


#fill nans with something else

#1.0 count nans
print("total nan count: ", df.isna().sum().sum())
print("nans in columns: ", df.isna().sum())

#1.0.0 if a variable has too many nans : delete that variable(column)
#1.0.1 fill nans with avg of each columns : numerical variables
df_numerical = df[numerical].fillna(df[numerical].mean())

#1.0.2 fill nans : categorical??  : fill with most frequent value??
categorical = categorical1 + categorical2 + categorical3 + categorical4 + categorical5 + categorical6 + categorical7 
df_categorical = df[categorical].apply(lambda x: x.fillna(x.value_counts().index[0]))

#1.0.3 y variale
y = df['SalePrice']

df=pd.concat([df_numerical, df_categorical,y], axis = 1)

print("total nan count: ", df.isna().sum().sum())
print("nans in columns: ",df.isna().sum())


# In[10]:


df


# ### 2.2 Drop numerical variables with multicollinearity

# In[11]:


#2.1 multicollinearity checking

#2.1.1 check multicollinearity among "numerical variables": If VIF >10 then that variable has multicollinearity
vif_data = pd.DataFrame()
vif_data["feature"] = df[numerical].columns
vif_data["VIF"] = [variance_inflation_factor(df[numerical].values, i) for i in range(len(df[numerical].columns))]


# In[12]:


vif_data


# In[13]:


#IF VIF >10 then that variable is known to have multicollinearity with other variables
df.drop(['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','TotRmsAbvGrd'], axis=1, inplace=True)


# In[14]:


df


# ### 2.3 Drop categorical variables with multicollinearity, by PCA after one-hot-encoding categorical variables

# In[15]:


fin_y = df['SalePrice']


# In[16]:


#2.3 One hot encoding only categorcal variables 
fin_x = df.loc[:, df.columns!='SalePrice']
fin_x_one_hot = pd.get_dummies(fin_x)
fin_x_one_hot.head(5)


# In[17]:


#standardization(both numerical and categorical varaibles)
scaler = StandardScaler()
fin_x_stand=scaler.fit_transform(fin_x_one_hot)
print(fin_x_stand.shape)


# In[18]:


#PCA (both numerical and categorical) #131개로 85% 설명, 151개로 90%설명

# Loop Function to identify number of principal components that explain at least 85% of the variance
for comp in range(3, fin_x_stand.shape[1]):
    pca = PCA(n_components= comp, random_state=42)
    pca.fit(fin_x_stand)
    comp_check = pca.explained_variance_ratio_
    final_comp = comp
    if comp_check.sum() > 0.85:
        break
        
Final_PCA = PCA(n_components= final_comp,random_state=42)
Final_PCA.fit(fin_x_stand)
cluster_df=Final_PCA.transform(fin_x_stand) #final dataframe after PCA
num_comps = comp_check.shape[0]
print("Using {} components, we can explain {}% of the variability in the original data.".format(final_comp,comp_check.sum()))


# In[19]:


cluster_df.shape


# In[20]:


#to dataframe
fin_x = pd.DataFrame(cluster_df)


# ### 2.3 Remove outliers

# In[21]:


#since we don't have a lot of data here, I won't remove any outliers...Maybe for better performances, I will....


# ### 2.4 Divide data to train/valid/test data

# In[22]:


#concat fin_x and fin_y)
fin_df=pd.concat([fin_x, fin_y], axis = 1)


# In[23]:


train_df, validate_df, test_df = np.split(fin_df.sample(frac=1), [int(.4*len(fin_df)), int(.7*len(fin_df))])


# In[24]:


train_df


# In[25]:


train_df = train_df.dropna(axis=0)
train_x = train_df.drop('SalePrice', axis=1)
train_y = train_df["SalePrice"]

validate_df = validate_df.dropna(axis=0)
valid_x = validate_df.drop('SalePrice', axis=1)
valid_y=validate_df["SalePrice"]

test_df = test_df.dropna(axis=0)
test_x = test_df.drop('SalePrice', axis=1)
test_y = test_df["SalePrice"]


# # 3.1 model1 - random forest

# In[27]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score # 정확도 함수

clf = RandomForestRegressor(n_estimators=20, max_depth=5,random_state=0)
clf.fit(train_x,train_y)

predict1 = clf.predict(test_x)



# In[28]:


#predict1.shape #(439,)
predict1


# In[29]:


test_y


# In[30]:


test_x


# In[31]:


#MSE
from sklearn.metrics import mean_squared_error 
print(mean_squared_error(test_y, predict1)**0.5) #rmse


# # 3.2 model2 - GBM

# In[117]:


from sklearn.ensemble import GradientBoostingClassifier

gb_clf = GradientBoostingClassifier(random_state = 0)
gb_clf.fit(train_x, train_y)
gb_pred = gb_clf.predict(test_x)
print(mean_squared_error(test_y, gb_pred)**0.5) #rmse


# In[118]:


gb_pred


# In[119]:


test_y


# # 3.3 model2 - xgboost

# In[228]:


#!pip install xgboost
import xgboost


# In[232]:


xgb_model = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
xgb_model.fit(train_x,train_y)
xgb_pred = xgb_model.predict(test_x)


# In[242]:


print(mean_squared_error(test_y, xgb_pred)**0.5) #rmse


# # 3.3 model3 - light gbm
# ##### LightGBM is fast and accurate model, but sometimes occurs overfitting problems when data size is small.

# In[251]:


from lightgbm import LGBMClassifier

lgb_clf = LGBMClassifier(num_leaves=31, objective='binary')
lgb_clf.fit(train_x, train_y)
lgbm_pred = lgb_clf.predict(test_x)


# In[252]:


print(mean_squared_error(test_y, lgbm_pred)**0.5) #rmse


# # 3.4 model 4 - ANN(3-layers)

# In[25]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class MultOutRegressor(nn.Module):
    def __init__(self, input_dim, target_dim, hidden_dim=32,seed=1234):
        super().__init__()
        torch.manual_seed(seed)
        self.hidden_dim = hidden_dim
        self.target_dim = target_dim

        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.target_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


# In[26]:


from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class TabularDataSet(Dataset) :
    def __init__(self, X , Y) :
        self._X = np.float32(X)
        self._Y = Y

    def __len__(self,) :
        return len(self._Y)

    def __getitem__(self,idx) :
        return self._X[idx], self._Y[idx]


# In[27]:


from torch import optim
from IPython import display

#convert dataframe to array for DataLoader 
numpy_x = train_x.to_numpy()
numpy_y = train_y.to_numpy()

tabulardataset = TabularDataSet(numpy_x,numpy_y) 
train_dl = DataLoader(tabulardataset) 
model = MultOutRegressor(131 , 1)
optimizer = optim.AdamW(model.parameters(), lr=0.001) #default lr = 0.01
criterion = nn.MSELoss()


# In[33]:


def update(input , target , model, criterion , optimizer,max_norm=5) :
    optimizer.zero_grad()
    output = model(input)
    print('output : ',output)
    loss = criterion(output , target.float())
    print('loss: ', loss)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()
    return loss 

def one_epoch(dataloader , model, criterion , optimizer ) :
    result = torch.FloatTensor([0])
    print('epoch_result',result)
    for idx , (input , target) in enumerate(dataloader) :
        loss = update(input , target , model, criterion , optimizer)
        result = torch.add(result , loss)
    else :
        result /= idx+1
        return result.detach().cpu().numpy()

def visualize(result) :
    display.clear_output(wait=True)
    plt.plot(result)
    plt.show()

def train(n_epochs , dataloader , model, criterion , optimizer , log_interval=10) :
    epoch_loss = []
    for epoch in range(n_epochs) :
        loss = one_epoch(dataloader , model, criterion , optimizer )
        print('train_loss:',loss)
        if epoch > 0 :
            epoch_loss.append(loss)
        if epoch % log_interval == 0 :
            visualize(epoch_loss)
    else :
        return np.min(epoch_loss)


# In[113]:


#train the model
train(500, train_dl ,model, criterion , optimizer,log_interval=50) #default n_epochs = 500 : min_loss = 2532927200.0


# In[132]:


#test the model
ann_pred1 = model(torch.tensor(test_x.values).float())
print(mean_squared_error(test_y, ann_pred1.detach().numpy())**0.5) #rmse


# # 3.5 model 5 - ANN(5-layers)

# In[136]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class MultOutRegressor2(nn.Module):
    def __init__(self, input_dim, target_dim, hidden_dim=32,seed=1234):
        super().__init__()
        torch.manual_seed(seed)
        self.hidden_dim = hidden_dim
        self.target_dim = target_dim

        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc5 = nn.Linear(self.hidden_dim, self.target_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        return x


# In[137]:


from torch import optim
from IPython import display

#convert dataframe to array for DataLoader 
numpy_x = train_x.to_numpy()
numpy_y = train_y.to_numpy()

tabulardataset = TabularDataSet(numpy_x,numpy_y) 
train_dl = DataLoader(tabulardataset) 
model2 = MultOutRegressor2(131 , 1)
optimizer = optim.AdamW(model2.parameters(), lr=0.001) #default lr = 0.01
criterion = nn.MSELoss()


# In[138]:


#train the model
train(500, train_dl ,model2, criterion , optimizer,log_interval=50) #default n_epochs = 500 : min_loss = 481092860


# In[139]:


#test the model
ann_pred2 = model2(torch.tensor(test_x.values).float())
print(mean_squared_error(test_y, ann_pred2.detach().numpy())**0.5) #rmse


# # 4. Tuning model's hyperparameters with K-fold cross validation

# In[32]:


#training data for k-fold cross validation 
train_x_kf = pd.concat([train_x, valid_x], axis = 0)
train_y_kf = pd.concat([train_y, valid_y], axis = 0)


#  ## 4.1 Random Forest - K-fold cross validation

# In[33]:


from sklearn.model_selection import RandomizedSearchCV

#define rf_grid
# Number of trees in Random Forest
rf_n_estimators = [int(x) for x in np.linspace(200, 1000, 5)]
rf_n_estimators.append(1500)
rf_n_estimators.append(2000)

# Maximum number of levels in tree
rf_max_depth = [int(x) for x in np.linspace(5, 55, 11)]
# Add the default as a possible value
rf_max_depth.append(None)

# Number of features to consider at every split
rf_max_features = ['auto', 'sqrt', 'log2']

# Criterion to split on
#rf_criterion = ['mse', 'mae']

# Minimum number of samples required to split a node
rf_min_samples_split = [int(x) for x in np.linspace(2, 10, 9)]

# Minimum decrease in impurity required for split to happen
rf_min_impurity_decrease = [0.0, 0.05, 0.1]

# Method of selecting samples for training each tree
rf_bootstrap = [True, False]

rf_grid = {'n_estimators': rf_n_estimators,
               'max_depth': rf_max_depth,
               'max_features': rf_max_features,
               #'criterion': rf_criterion,
               'min_samples_split': rf_min_samples_split,
               'min_impurity_decrease': rf_min_impurity_decrease,
               'bootstrap': rf_bootstrap}

# Create the model to be tuned
rf_base = RandomForestRegressor()

# Create the random search Random Forest
rf_random = RandomizedSearchCV(estimator = rf_base, param_distributions = rf_grid, 
                               n_iter = 200, cv = 3, verbose = 2, random_state = 42, 
                               n_jobs = -1)

# Fit the random search model
rf_random.fit(train_x_kf, train_y_kf)

# View the best parameters from the random search
rf_random.best_params_


# In[34]:


#now try with random forest with best parameters
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score # 정확도 함수

clf_cv = RandomForestRegressor(n_estimators=600, max_depth=5,random_state=0,min_samples_split=3,min_impurity_decrease=0.05,max_features='log2',bootstrap=True)

clf_cv.fit(train_x,train_y)

predict1 = clf_cv.predict(test_x)


# In[35]:


#MSE
from sklearn.metrics import mean_squared_error 
print(mean_squared_error(test_y, predict1)**0.5) #rmse

