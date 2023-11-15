#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 09:52:58 2023

@author: mac
"""

import pandas as pd
import numpy as np

dataset = pd.read_csv("GOOG.csv")
df = pd.DataFrame(dataset)

df.drop(columns =['symbol', 'adjClose'] , inplace=True)

# Convert 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Set 'date' column as index
df = df.set_index('date')

# Assume 'target_feature' is your target feature, modify it as per your dataset
X_df = df.drop(['close'], axis=1)  # Assuming 'target_feature' is the target feature
y_series = df['close']
features = ['high', 'low', 'open', 'volume',
       'adjHigh', 'adjLow', 'adjOpen', 'adjVolume', 'divCash', 'splitFactor']

# Perform train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_df, y_series, test_size=0.33, random_state=42)

# Apply Random Forest regression algorithm
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=0)
rf.fit(X_train, y_train)

# Print selected features
print(rf.feature_importances_)
print(np.array(features)[rf.feature_importances_ > 0])


