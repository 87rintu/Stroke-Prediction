# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import pickle

df = pd.read_csv('C:/Users/Rintu Pradhan/Desktop/Project/Stroke_prediction/train.csv')

df['smoking_status'].fillna('never smoked', inplace=True)

#Dropping rows having null values
df.dropna(inplace=True)

#Creating dummies of categprical variables
d1 = pd.get_dummies(df[['gender', 'smoking_status']], drop_first=True)
df = pd.concat([df, d1], axis=1)

df.drop(['id', 'ever_married', 'Residence_type','gender', 'work_type', 'smoking_status'], inplace=True, axis=1)

#split into train & test
x = df.drop('stroke', axis=1)
y = df['stroke']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=42)

#Building the model
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(class_weight = 'balanced')
mod_balanced_lr = log_reg.fit(x_train, y_train)
#y_pred_lr_balanced = mod_balanced_lr.predict(x_test)


# Saving model to disk
pickle.dump(mod_balanced_lr, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[1, 0,55, 0,0, 113.45, 27.9, 0,0]]))