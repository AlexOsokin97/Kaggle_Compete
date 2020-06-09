# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 00:40:55 2020

@author: Alexander
"""

import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')

df2 = pd.read_csv('test.csv')

TotalHouseSF = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'] + df['GarageArea'] + df['OpenPorchSF'] + df['EnclosedPorch'] + df['WoodDeckSF'] + df['3SsnPorch'] + df['ScreenPorch'] + df['PoolArea']




updated_df = df[['MSZoning','Street' ,'LandContour','Utilities','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle',
                'OverallQual','OverallCond','YearBuilt','YearRemodAdd','Foundation','BsmtCond','TotalBsmtSF','Heating','HeatingQC','CentralAir',
                'Electrical', '1stFlrSF', '2ndFlrSF', 'KitchenAbvGr','KitchenQual', 'TotRmsAbvGrd', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
                'HalfBath','Functional','Fireplaces','GarageType', 'GarageYrBlt','GarageArea','GarageQual','MiscVal','MoSold',
                'YrSold','SaleType','SaleCondition','SalePrice']]


TotalHouseSF = np.array(TotalHouseSF)


