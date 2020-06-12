# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 00:40:55 2020

@author: Alexander
"""

import pandas as pd
import numpy as np

df = pd.read_csv('new_train.csv')

df_dum = pd.get_dummies(df)

df_dum.drop(labels=['Unnamed: 0','AgeBeforeRemodel', 'MSZoning_RM', 'Street_Pave', 'LandContour_Lvl', 'Utilities_NoSeWa', 'LandSlope_Sev', 'Neighborhood_Veenker', 'Condition1_RRNn', 'Condition2_RRNn', 'BldgType_TwnhsE', 'HouseStyle_SLvl', 'Foundation_Wood', 'BsmtCond_TA', 'Heating_Wall', 'HeatingQC_TA', 'CentralAir_Y', 'Electrical_SBrkr', 'KitchenQual_TA', 'Functional_Typ', 'GarageType_NG', 'GarageQual_TA', 'SaleType_WD', 'SaleCondition_Partial'], axis=1, inplace=True)

X = df_dum.drop(['SalePrice'], axis=1).values
y = df_dum['SalePrice'].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import Normalizer

n = Normalizer()

X_train[:, 0:9] = n.fit_transform(X_train[:,0:9])
X_test[:, 0:9] = n.transform(X_test[:, 0:9])

##########################################################3
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_train, y_train)

reg.score(X_test, y_test)

y_pred = reg.predict(X_test)

from sklearn.metrics import mean_squared_log_error
rmsle = np.sqrt(mean_squared_log_error( y_test, y_pred))
##############################################################
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

rf = RandomForestRegressor(random_state=42)

scores = np.mean(cross_val_score(estimator=rf, X=X_train, y=y_train,
                                 scoring='neg_mean_squared_log_error', cv=3))
print("avg_root_mean_squared_log_error: ", np.sqrt(scores*(-1)))

#feature tuning
params = [{'criterion':('mse','mae'), 'max_features':('sqrt','log2')}]
gs = GridSearchCV(rf, params, scoring='neg_mean_squared_log_error', cv=3)

gs.fit(X_train, y_train)

np.sqrt(gs.best_score_*(-1))
gs.best_params_
gs.best_estimator_

y_pred_rf = gs.best_estimator_.predict(X_test)
rmsle_rf = np.sqrt(mean_squared_log_error( y_test, y_pred_rf))
####################################################################

