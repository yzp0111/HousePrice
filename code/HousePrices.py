# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import sklearn.metrics as sm
import sklearn.preprocessing as sp
import sklearn.ensemble as se
import sklearn.tree as st
import sklearn.svm as svm


train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')
print(train_data.info())
print(train_data.head())
for row in train_data:
    if sum(train_data[row].isnull()):
        print(row, '缺少', sum(train_data[row].isnull()))
features = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF',
            '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea']
TotalBsmtSF_features = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'GarageCars', 'GarageArea',
                        '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd']
GarageCars_features = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF',
                       '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd']
print(test_data[features].info())


model_TotalBsmtSF = lm.Ridge(80, fit_intercept=True)
model_TotalBsmtSF.fit(
    train_data[TotalBsmtSF_features], train_data[['TotalBsmtSF']])
test_data.loc[test_data.TotalBsmtSF.isnull(), ['TotalBsmtSF']
              ] = model_TotalBsmtSF.predict(test_data.loc[test_data.TotalBsmtSF.isnull(), TotalBsmtSF_features])
model_GarageCars = lm.Ridge(80, fit_intercept=True)
model_GarageCars.fit(
    train_data[GarageCars_features], train_data[['GarageCars']])
test_data.loc[test_data.GarageCars.isnull(), ['GarageCars']
              ] = model_GarageCars.predict(test_data.loc[test_data.GarageCars.isnull(), GarageCars_features])
model_GarageArea = lm.Ridge(80, fit_intercept=True)
model_GarageArea.fit(
    train_data[GarageCars_features], train_data[['GarageCars']])
test_data.loc[test_data.GarageArea.isnull(), ['GarageArea']
              ] = model_GarageArea.predict(test_data.loc[test_data.GarageArea.isnull(), GarageCars_features])
zero_to_one = sp.MinMaxScaler(feature_range=(0, 1))
train_data[features] = zero_to_one.fit_transform(train_data[features])
test_data[features] = zero_to_one.transform(test_data[features])

# 岭回归
# model = lm.Ridge(80	, fit_intercept=True)
# # train_x, test_x, train_y, test_y = ms.train_test_split(
# #     train_data[features], train_data[['SalePrice']], random_state=1)
# # model.fit(train_x, train_y)
# # pred_test_y = model.predict(test_x)
# # print(sm.r2_score(test_y, pred_test_y))
# model.fit(train_data[features], train_data[['SalePrice']])
# pred_test_y = model.predict(test_data[features]).flatten()
# print(pred_test_y)
# result = pd.DataFrame({'Id': test_data.Id, 'SalePrice': pred_test_y})
# result.to_csv('./predict_Ridge.csv', index=False)

# 随机森林回归
# params = [{'max_depth': [i for i in range(6, 15)], 'n_estimators':[
#     i for i in range(60, 300, 20)]}]
# model = ms.GridSearchCV(se.RandomForestRegressor(
#     random_state=4), params, cv=3, verbose=2)
# model.fit(train_data[features], train_data['SalePrice'])
# for param, score in zip(model.cv_results_['params'], model.cv_results_['mean_test_score']):
#     print(param, score)
# print(model.best_params_)
# print(model.best_score_)

model = se.RandomForestRegressor(
    max_depth=12, n_estimators=80, random_state=6)
# train_x, test_x, train_y, test_y = ms.train_test_split(
#     train_data[features], train_data[['SalePrice']], random_state=1)
# model.fit(train_x, train_y)
# pred_test_y = model.predict(test_x)
# print(sm.r2_score(test_y, pred_test_y))
model.fit(train_data[features], train_data[['SalePrice']])
pred_test_y = model.predict(test_data[features]).flatten()
result = pd.DataFrame({'Id': test_data.Id, 'SalePrice': pred_test_y})
result.to_csv('./predict_randomforest.csv', index=False)


# 正向激励
# model = se.AdaBoostRegressor(st.DecisionTreeRegressor(
#     max_depth=6), n_estimators=200, random_state=2)
# train_x, test_x, train_y, test_y = ms.train_test_split(
#     train_data[features], train_data[['SalePrice']], random_state=1)
# model.fit(train_x, train_y)
# pred_test_y = model.predict(test_x)
# print(sm.r2_score(test_y, pred_test_y))
# model.fit(train_data[features], train_data[['SalePrice']])
# pred_test_y = model.predict(test_data[features]).flatten()
# result = pd.DataFrame({'Id': test_data.Id, 'SalePrice': pred_test_y})
# result.to_csv('./predict_AdaBoost.csv', index=False)
