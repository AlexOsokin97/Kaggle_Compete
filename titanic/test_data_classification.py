# -*- coding: utf-8 -*-
"""
Created on Mon May  4 23:25:26 2020

@author: Alex
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing our test data
df_test = pd.read_csv("Original_DF's/test.csv")

#preparing our data
df_test.set_index('PassengerId', inplace=True)
df_test.drop('Cabin',axis=1 , inplace=True)
df_test['Embarked'] = df_test['Embarked'].apply(lambda x: 'Queenstown' if x=='Q' else x).apply(lambda x: 'Southampton' if x=='S' else x).apply(lambda x: 'Cherbourg' if x=='C' else x)

#creating the model df
df_model = df_test.drop(['Name', 'Ticket'], axis=1)

#creating dummy variables
df_dum = pd.get_dummies(df_model)

df_dum = df_dum.drop(['Sex_male','Embarked_Southampton'], axis=1)

#creating tasting data for our model
X_test = df_dum.values

#using fancyimpute to fill the missing values
from fancyimpute import KNN
X_test_filled = KNN(k=15).fit_transform(X_test)

#applying standarization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_test_filled[:,[1,4]] = sc.fit_transform(X_test_filled[:,[1,4]])

#loading in our trained SVM model
import pickle
loaded_model = pickle.load(open('gbc_model.sav', 'rb'))

Survived_prediction = loaded_model.predict(X_test_filled)

df_test.insert(0, 'Survived', Survived_prediction)

df = df_test['Survived']

df.value_counts()

df.to_csv('df_TESTED.csv')
