import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

dataset = pd.read_csv('Churn.csv')

# Data preprocessing 

X = dataset
X = X.drop(['Churn'], axis =1)
X = X.drop(['customerID'], axis =1)
y = dataset.iloc[:,-1].values  

non_numeric_cols = list(X.select_dtypes(include = ['object']).columns)
le = {}
for col in non_numeric_cols:
    encoder = LabelEncoder()
    X[col] = encoder.fit_transform(X[col])
    le[col] = encoder
    
ley = LabelEncoder()
y =ley.fit_transform(y)    
y = y.astype(np.int64)

# Managing Categorical Variables
cat_cols = X.nunique()[X.nunique() < 6].keys().tolist()
bin_cols   = X.nunique()[X.nunique() == 2].keys().tolist()
#Columns more than 2 values
multi_cols = [i for i in cat_cols if i not in bin_cols]

X = pd.get_dummies(data = X,columns = multi_cols )
X = X.to_numpy()

# Creating Training set and Test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# ANN

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(activation = 'relu', input_dim = 40, units = 20, kernel_initializer = "uniform",  ))
classifier.add(Dense(activation = 'relu', units = 20, kernel_initializer = "uniform",  ))
classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = "uniform" ))
 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 50 , epochs =100 )

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

