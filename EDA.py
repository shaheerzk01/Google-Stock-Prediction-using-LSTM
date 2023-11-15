#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 09:48:45 2023

@author: mac
"""


import pandas as pd


dataset = pd.read_csv("GOOG.csv")
df = pd.DataFrame(dataset)

# Sorting data acc to date
df = df.sort_values(by='date')

print("Number of rows and columns\n")
print(df.shape, "\n")

# Dropping the column of symbol
df.drop(columns =['symbol'] , inplace=True)

# Searching for missing values
print("Checking missing values\n")
missing_values = df.isnull().sum()
print(missing_values,"\n")

# Searching for dublicated values by checking the date column
print("Checking the duplicated values\n")
duplicated_values = df["date"].duplicated().sum()
print(duplicated_values,"\n")

# To get a concise summary of a DataFrame.
print("Getting info about the dataframe\n")
print(df.info(), "\n")

# To generate descriptive statistics of a DataFrame.
print("Getting descriptive statistics of dataframe\n")
print(df.describe(), "\n")

# Calculate average daily price change 
daily_price_change = df['close'] - df['open']
average_price_change = daily_price_change.mean()

# Calculate the average closing price
average_close_price = df['close'].mean()

# Calculate the highest closing price
highest_close_price = df['close'].max()

# Calculate the lowest closing price
lowest_close_price = df['close'].min()

# Calculate the total volume traded
total_volume = df['volume'].sum()

# Calculate daily returns on average
daily_return = df['close'].pct_change() * 100
daily_return_average = daily_return.mean()
    
# Print the results
print("Average daily price change: ", average_price_change)
print("Average Close Price: ", average_close_price)
print("Highest Close Price: ", highest_close_price)
print("Lowest Close Price: ", lowest_close_price)
print("Daily return on average: ", daily_return_average)
print("Total Volume: ", total_volume, "\n")

copy_df = df.copy()

# Calculate 7-day and 30-day moving averages
copy_df['short_MA'] = df['close'].rolling(window=7).mean()
copy_df['long_MA'] = df['close'].rolling(window=30).mean()

print("7-days moving average\n")
print(copy_df['short_MA'], "\n")

print("30-days moving average\n")
print(copy_df['long_MA'], "\n")








