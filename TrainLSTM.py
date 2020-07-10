
# coding: utf-8


import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
import yfinance as yf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import numpy as np


today = datetime.today().strftime("%Y-%m-%d")
time_ago = (datetime.today() - relativedelta(years=2)).strftime("%Y-%m-%d")
data = yf.download("btc-usd", start=time_ago, end=today)

best_score=1000
best_period=0

for period in [10,20,30,40,50,60,70]:

    training_set = data[0:-period]
    test_set  = data[len(data)-period:]
    training_set=training_set.iloc[:,0:1].values

    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    X_train = []
    y_train = []


    for i in range(period, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-period: i, 0])
        y_train.append(training_set_scaled[i, 0]) 


    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, newshape = (X_train.shape[0], X_train.shape[1], 1))


    model = Sequential()
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    model.fit(X_train, y_train, epochs = 100, batch_size = 32)


    inputs = data
    inputs = inputs.iloc[:,0:1].values
    inputs = sc.fit_transform(inputs)

    X_test = []
    y_test = []

    for i in range(period, len(inputs)): 
        X_test.append(inputs[i-period: i, 0])
        y_test.append(inputs[i, 0]) 

    X_test,y_test = np.array(X_test), np.array(y_test)

    X_test = np.reshape(X_test, newshape = (X_test.shape[0], X_test.shape[1], 1))

    score  = model.evaluate(X_test, y_test,  batch_size=32)
    print('Test score:', score)
    if score<best_score:
        best_score = score
        best_period = period

print(best_period)
print(best_score)