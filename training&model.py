#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 13:03:07 2023

@author: mac
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler 
import numpy as np
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

plt.style.use('seaborn')

dataset = pd.read_csv("GOOG.csv")
df = pd.DataFrame(dataset)

# Dropping the column of symbol
df.drop(columns =['symbol'] , inplace=True)

# Sorting data acc to date
df = df.sort_values(by='date')

DF = df.copy()

# Convert 'date' column to datetime format
DF['date'] = pd.to_datetime(DF['date'])

# Set 'date' column as index
DF = DF.set_index('date')

DF.drop(columns =['adjClose','adjHigh','adjLow','adjOpen','adjVolume','divCash','splitFactor'] , inplace=True)

# Prepering data for trainning and testing the model

training_set = DF[:'2020'].iloc[:,0:1].values
test_set = DF['2020':].iloc[:,0:1].values

# Using 'close' price for prediction.
DF['close'][:'2020'].plot(figsize=(16,4),legend=True)
DF['close']['2020':].plot(figsize=(16,4),legend=True, color = 'r')
plt.legend(['Training set (Before 2020)','Test set (2020 and beyond)'], fontsize=12)
plt.title('Google stock price', fontsize=16)
plt.show()

# Normalization is very important in all deep learning in general. Normalization makes the properties more consistent.
# Scaling the training set
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

timesteps = 60

X_train = []
y_train = []
for i in range(timesteps,1147):
    X_train.append(training_set_scaled[i-timesteps:i,0])
    y_train.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping X_train for efficient modelling
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

dataset_total = pd.concat((DF['close'][:'2020'], DF['close']['2020':]),axis=0)
inputs = dataset_total[len(dataset_total)-len(test_set) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = sc.transform(inputs)

# Preparing X_test
X_test = []
for i in range(timesteps, len(inputs)):
    X_test.append(inputs[i-timesteps:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

# The LSTM architecture
Model = Sequential()
# First LSTM layer with Dropout regularisation
Model.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train.shape[1],1)))
Model.add(Dropout(0.2))
# Second LSTM layer
Model.add(LSTM(units = 100, return_sequences = True))
Model.add(Dropout(0.2))
# Third LSTM layer
Model.add(LSTM(units = 100, return_sequences = True))
Model.add(Dropout(0.2))
# Fourth LSTM layer
##add 4th lstm layer
#Model.add(layers.LSTM(units = 100))
#Model.add(layers.Dropout(rate = 0.2))

Model.add(layers.LSTM(units = 100, return_sequences = False))
Model.add(layers.Dropout(rate = 0.2))
Model.add(layers.Dense(units = 25))
Model.add(layers.Dense(units = 1))
# The output layer
Model.add(Dense(units = 1))

Model.summary()

# Compiling the model
Model.compile(optimizer= 'adam', loss = 'mean_squared_error', metrics =['accuracy'])

# Epochs and Batch Size
epochs = 15
batch_size = 32

#from keras import callbacks
#earlystopping = callbacks.EarlyStopping(monitor ="val_loss", mode ="min", patience = 2, restore_best_weights = True)
  
# Fitting the model 
history =  Model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)

loss = history.history['loss']

epochs = range(len(loss))

plt.plot(epochs, loss, 'r', label='Training loss')

plt.title('Training loss', size=15, weight='bold')
plt.legend(loc=0)
plt.figure()

plt.show()

predicted_stock_price = Model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Some functions to help out with
def plot_predictions(test,predicted):
    plt.plot(test, color='red',label='Real Google Stock Price')
    plt.plot(predicted, color='blue',label='Predicted Google Stock Price')
    plt.title('Google Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Google Stock Price')
    plt.legend()
    plt.show()

# Visualizing the results for LSTM
plot_predictions(test_set, predicted_stock_price)
