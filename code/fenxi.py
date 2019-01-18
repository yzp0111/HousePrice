# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import pandas as pd
import sklearn.preprocessing as sp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class DigitEncoder():
    # 编码器

    def fit_transform(self, y):
        return y.astype(int)

    def transform(self, y):
        return y.astype(int)

    def inverse_transform(self, y):
        return y.astype(str)

train_data = pd.read_csv('../data/train.csv')
print(train_data.info())
print(train_data.head())
for row in train_data:
    if sum(train_data[row].isnull()):
        print(row, '缺少', sum(train_data[row].isnull()))

# ---------------train------------
# LotFrontage 缺少 259
# Alley 缺少 1369
# MasVnrType 缺少 8
# MasVnrArea 缺少 8
# BsmtQual 缺少 37
# BsmtCond 缺少 37
# BsmtExposure 缺少 38
# BsmtFinType1 缺少 37
# BsmtFinType2 缺少 38
# Electrical 缺少 1
# FireplaceQu 缺少 690
# GarageType 缺少 81
# GarageYrBlt 缺少 81
# GarageFinish 缺少 81
# GarageQual 缺少 81
# GarageCond 缺少 81
# PoolQC 缺少 1453
# Fence 缺少 1179
# MiscFeature 缺少 1406

#---------------test-------------
# MSZoning 缺少 4
# LotFrontage 缺少 227
# Alley 缺少 1352
# Utilities 缺少 2
# Exterior1st 缺少 1
# Exterior2nd 缺少 1
# MasVnrType 缺少 16
# MasVnrArea 缺少 15
# BsmtQual 缺少 44
# BsmtCond 缺少 45
# BsmtExposure 缺少 44
# BsmtFinType1 缺少 42
# BsmtFinSF1 缺少 1
# BsmtFinType2 缺少 42
# BsmtFinSF2 缺少 1
# BsmtUnfSF 缺少 1
# TotalBsmtSF 缺少 1
# BsmtFullBath 缺少 2
# BsmtHalfBath 缺少 2
# KitchenQual 缺少 1
# Functional 缺少 2
# FireplaceQu 缺少 730
# GarageType 缺少 76
# GarageYrBlt 缺少 78
# GarageFinish 缺少 78
# GarageCars 缺少 1
# GarageArea 缺少 1
# GarageQual 缺少 78
# GarageCond 缺少 78
# PoolQC 缺少 1456
# Fence 缺少 1169
# MiscFeature 缺少 1408
# SaleType 缺少 1


# encoders, x = [], []

# for feature in train_data:
#     if type(train_data[feature][0]) == str:
#         encoder = sp.LabelEncoder()
#     else:
#         encoder = DigitEncoder()


# train_data.MSSubClass.value_counts().plot(kind='bar')
# plt.figure()
# plt.scatter(train_data.MSSubClass, train_data.SalePrice)
# plt.show()

# train_data.LotFrontage.value_counts().plot(kind='bar')
# plt.figure()
# plt.scatter(train_data.LotFrontage, train_data.SalePrice)
# plt.show()

# train_data.Alley.value_counts().plot(kind='bar')
# grvl = train_data.SalePrice[train_data.Alley == 'Grvl']
# pave = train_data.SalePrice[train_data.Alley == 'Pave']
# nan = train_data.SalePrice[train_data.Alley.isnull()]
# df = pd.DataFrame({'Grvl': grvl, 'Pave': pave, 'Nan': nan})
# df.plot(kind='kde')
# # plt.scatter(train_data.LotFrontage, train_data.SalePrice)
# plt.show()
# df.plot(kind='bar')
# plt.title('Pclass-Survived', fontsize=20)
# plt.ylabel('num', fontsize=14)
# plt.show()

corrmat = train_data.corr()
print(corrmat.loc['SalePrice', corrmat.SalePrice > 0.5])
# plt.imshow(corrmat, cmap='jet')
# plt.show()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
plt.show()
