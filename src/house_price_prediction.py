#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 15:06:20 2020

@author: jonyoung
"""

# import the libraries we need
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, DotProduct

# set directories
data_dir = '../data/'
results_dir = '../results/'

# import train and test 
train_data_DF = pd.read_csv(data_dir + 'train.csv')
test_data_DF = pd.read_csv(data_dir + 'test.csv')

# split both train and test into labels (for train only), data and id
# take log of sale price
train_id = train_data_DF['Id'].values
train_sale_price = train_data_DF['SalePrice'].values
train_data_DF = train_data_DF.iloc[:, 1:-1]
test_id = test_data_DF['Id'].values
test_data_DF = test_data_DF.iloc[:, 1:]
log_train_sale_price = np.log(train_sale_price)

# data is a mixture of numerical and categorical variables - these must be treated differently.
# use information from the data description to decide which is 
# hard-code list of numerical as there are fewer!
predictor_vars = train_data_DF.columns.to_list()
numerical_vars = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 
                  'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
                  'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 
                  'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
                  'MiscVal', 'MoSold', 'YrSold']
categorical_vars = [predictor_var for predictor_var in predictor_vars if not predictor_var in numerical_vars]

# split both train and test datasets into numercial and categorical
train_numerical_data_DF = train_data_DF[numerical_vars]
test_numerical_data_DF = test_data_DF[numerical_vars]
train_categorical_data_DF = train_data_DF[categorical_vars]
test_categorical_data_DF = test_data_DF[categorical_vars]

# for categorical data, replace nans & one-hot encode
# MSSubCLass is really categorical but is interpreted as numeric
# convert to string as hacky way to use get_dummies on it
train_categorical_data_DF = train_categorical_data_DF.fillna('None')
train_categorical_data_DF['MSSubClass'] = train_categorical_data_DF['MSSubClass'].astype(str)
train_categorical_data_dummy_DF = pd.get_dummies(train_categorical_data_DF)
test_categorical_data_DF = test_categorical_data_DF.fillna('None')
test_categorical_data_DF['MSSubClass'] = test_categorical_data_DF['MSSubClass'].astype(str)
test_categorical_data_dummy_DF = pd.get_dummies(test_categorical_data_DF)

# train and test have slightly different sets of variables due to different values in categorical vars
# only use common ones so train & test match
common_columns = np.intersect1d(train_categorical_data_dummy_DF.columns, test_categorical_data_dummy_DF.columns)
train_categorical_data_dummy_DF = train_categorical_data_dummy_DF[common_columns]
test_categorical_data_dummy_DF = test_categorical_data_dummy_DF[common_columns]

# process numerical data: replace all NaNs with column mean as a quick and dirty 
# way to deal with missing(??) data. Use train mean for test data
train_numerical_data_DF=train_numerical_data_DF.fillna(train_numerical_data_DF.mean())
test_numerical_data_DF=test_numerical_data_DF.fillna(train_numerical_data_DF.mean())

# extract numerical values for all datasets and horizontally join to make numpy
# array train and test sets
train_data = np.hstack((train_numerical_data_DF.values, train_categorical_data_dummy_DF.values))
test_data = np.hstack((test_numerical_data_DF.values, test_categorical_data_dummy_DF.values))

# get list of column names for model interpretability
final_variables = train_numerical_data_DF.columns.to_list() + train_categorical_data_dummy_DF.columns.to_list()

# variables are at very different scales to each other. Larger ones may artificially
# dominate more informative smaller ones so need to normalise
# would normally use z-score but this makes little sense for dummy variables so scale to 0 to 1 instead
# subtract min then divide by max. use training min and max for test data
train_data_min = np.min(train_data, axis=0)
train_data = train_data - train_data_min
test_data = test_data - train_data_min
train_data_max = np.max(train_data, axis=0)
train_data = train_data / train_data_max
test_data = test_data / train_data_max

# initialise the regressor: regression with elastic net penalty
# use internal CV to fit l1 ratio regression parameter
# leave others as default
# regularisation minimises overfitting
# also encourages sparsity to improve model interpretability
rgr = ElasticNetCV(l1_ratio=[.01, .05, .1, .5, .7, .9, .95, .99, 1])

# initialise alternative regressor: Gaussian process w choice of kernels
kernel = RationalQuadratic()
gpr = GaussianProcessRegressor(kernel=kernel)

# try cross validation on train data only to compare methods
# to hold predictions
n_folds = 10
rmse_per_fold = np.zeros(n_folds,)
CV_preds = np.zeros_like(log_train_sale_price)
kf = KFold(n_splits=n_folds)
for i, (train_index, test_index) in enumerate(kf.split(train_data)) :
    
    print ('fold ' + str(i + 1))
    
    fold_train_data = train_data[train_index, :]
    fold_test_data = train_data[test_index, :]
    fold_train_labels = log_train_sale_price[train_index]
    fold_test_labels = log_train_sale_price[test_index]
    # rgr.fit(fold_train_data, fold_train_labels)
    # preds = rgr.predict(fold_test_data)
    gpr.fit(fold_train_data, fold_train_labels)
    preds = gpr.predict(fold_test_data)
    CV_preds[test_index] = preds
    rmse_per_fold[i] = np.sqrt(mean_squared_error(fold_test_labels, preds))
    CV_preds[test_index] = preds
    
std_RMSE = np.std(rmse_per_fold)
overall_RMSE = np.sqrt(mean_squared_error(log_train_sale_price, CV_preds))
print ('std of RMSE over folds = ' + str(std_RMSE))
print ('overall RMSE = ' + str(overall_RMSE))


# train and make predictions on test data
rgr.fit(train_data, log_train_sale_price)
log_preds = rgr.predict(test_data)

# get regressor weights for model interpretability
# get absolute values as these show importance to model
# but keep sign to show direction of influence on the model
weights = rgr.coef_
abs_weights = np.abs(weights)
sign = np.sign(weights)

# put in DF with variable names
weight_ranks = pd.DataFrame(final_variables)
weight_ranks = pd.concat([weight_ranks, pd.DataFrame(abs_weights)], axis = 1)
weight_ranks = pd.concat([weight_ranks, pd.DataFrame(sign)], axis = 1)
weight_ranks.columns = ['predictor', 'absolute weight', 'weight sign']

# write out final results and weight rankings
# remember predctions are LOG price so take exponential
results = pd.DataFrame(test_id)
results = pd.concat([results, pd.DataFrame(np.exp(log_preds))], axis = 1)
results.columns = ['Id', 'SalePrice']
results.to_csv(results_dir + 'my_submission_results.csv', index=False)
weight_ranks.to_csv(results_dir + 'feature_weights.csv', index=False)






                  