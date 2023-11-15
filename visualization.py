#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 12:51:51 2023

@author: mac
"""

import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn')

dataset = pd.read_csv("GOOG.csv")
df = pd.DataFrame(dataset)

# Sorting data acc to date
df = df.sort_values(by='date')

# Convert 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Set 'date' column as index
df = df.set_index('date')

# Dropping the column of symbol
df.drop(columns =['symbol'] , inplace=True)

copy_df = df.copy()

# Calculate 7-day and 30-day moving averages
copy_df['short_MA'] = df['close'].rolling(window=7).mean()
copy_df['long_MA'] = df['close'].rolling(window=30).mean()

# Create a new figure
plt.figure(figsize=(12, 6))

# Plot the closing prices
plt.plot(df.index, df['close'], label='Closing Price', color='blue')

# Plot the 7-day moving average
plt.plot(copy_df.index, copy_df['short_MA'], label='7-day Moving Average', color='orange')

# Plot the 30-day moving average
plt.plot(copy_df.index, copy_df['long_MA'], label='30-day Moving Average', color='red')

# Add labels and legend
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Google Stock Price and Moving Averages')
plt.legend()

#Volume Bars

plt.figure(figsize=(12, 6))
plt.bar(df.index, df['volume'], color='blue', width=0.5)
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title('Trading Volume')

df.drop(columns =['adjClose','adjHigh','adjLow','adjOpen','adjVolume','divCash','splitFactor'] , inplace=True)

# The number of rows, columns, and the subplot counter are initialized. 
a = 5  # number of rows
b = 1  # number of columns
c = 1  # initialize plot counter

colors = plt.rcParams["axes.prop_cycle"]()
# A figure is initialized 
fig = plt.figure(figsize=(12,15))

for i in range(len(df.columns)):
    color = next(colors)["color"]
    plt.subplot(a, b, c)
    #Plotting the line for each column in a different color
    plt.plot(df[df.columns[i]], color = color)
    # The spines of each plot are made invisible and the figure is adjusted and shown.
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    #Adding a horizontal line for the average of the column
    plt.axhline(df[df.columns[i]].mean(), linestyle='--', lw=2, zorder=1, color='black')
    #Adding Title
    plt.title("Google "+ df.columns[i] + " figures", fontsize=16)
    plt.xlabel('Years')
    #Adding y axis label
    plt.ylabel(df.columns[i])
    #Adding Legend
    plt.legend([df.columns[i]])
    #Plot Counter value is increased by one after each iteration
    c = c + 1

#Layout is tightended up
plt.tight_layout()

#plot is displayed
plt.show()