import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression

stock= pd.read_csv("sphist.csv")
#print(stock.head())

stock["Date"]= pd.to_datetime(stock["Date"])
after2015= stock["Date"]> datetime(year= 2015, month=4, day=1)

stock= stock.sort_values(by=['Date'], ascending= True)

# Adding new columns that provide information like average and std of the past 'n' days.

# This will incorporate information from multiple prior rows into one, and will make predictions much more accurate.

stock["past_5"]= stock["Close"].rolling(window=5).mean().shift()

stock["past_30"]= stock["Close"].rolling(window=30).mean().shift()

stock["past_365"]= stock["Close"].rolling(window=365).mean().shift()

# print(stock.head(10))

# We can see that using the rolling mean generated a lot of NaN values

# Let's see how many missing values do we have here.

print(stock.isnull().sum())

# There were 5, 30 and 365 missing value respectively.

print(stock.shape)

## Cleaning the data

# If we get rid of all the NaN values we lose about 2% of the total number of rows.

stock_new= stock.dropna()

print("New shape", stock_new.shape)

#Getting rid of irrelevant information
latest_stock= stock_new[stock_new["Date"] >= datetime(year=1951, month=1, day=3)]


train= latest_stock[latest_stock["Date"] < datetime(year=2013, month=1, day=1)]

test= latest_stock[latest_stock["Date"] >=datetime(year=2013, month=1, day=1)]



# Training a Linear Regression Model
def mae(y, yhat):
    diff= np.abs(y-yhat)
    mean= np.mean(diff)
    return mean


# Training based on the day_5 column
model_5= LinearRegression()
model_5.fit(train[['past_5']], train["Close"])
pred_5= model_5.predict(test[['past_5']])
mae_5= mae(test["Close"], pred_5)
print("MAE for model trained on past_5 column : ", mae_5 )

# Training based on the day_30 column
model_30= LinearRegression()
model_30.fit(train[['past_30']], train["Close"])
pred_30= model_30.predict(test[['past_30']])
mae_30= mae(test["Close"], pred_30)
print("MAE for model trained on past_30 column : ", mae_30 )

# Training based on the day_365 column
model_365= LinearRegression()
model_365.fit(train[['past_365']], train["Close"])
pred_365= model_365.predict(test[['past_365']])
mae_365= mae(test["Close"], pred_365)
print("MAE for model trained on past_365 column : ", mae_365 )


# Training based on the day_5 and day_30 column
model_5_30= LinearRegression()
model_5_30.fit(train[['past_365', 'past_365']], train["Close"])
pred_5_30= model_5_30.predict(test[['past_365', 'past_5']])
mae_5_30= mae(test["Close"], pred_5_30)
print("MAE for model trained on day_5 and day_30 columns : ", mae_5_30 )

# Training based on the day_5, day_30 and day_365 columns
model_5_30_365= LinearRegression()
model_5_30_365.fit(train[['past_30', 'past_5', 'past_365']], train["Close"])
pred_5_30_365= model_5_30_365.predict(test[['past_30', 'past_5', 'past_365']])
mae_5_30_365= mae(test["Close"], pred_5_30_365)
print("MAE for model trained on day_5, day_30 and day_365 columns : ", mae_5_30_365 )

# It can be seen from the MAE values that choosing more number of columns certainly helps. But something to consider in a sitauation when there are a lot of features is that choosing the right features is equally important. 

# Future scope includes adding more features like
## average volume over the past five days
## ratio between the average volume for the past five days, and the average volume for the past year
## The standard deviation of the average volume over the past five days.
## Ratio between the lowest price in the past year and the current price
## Taking into consideration the year, month, day of the week and the number of holidays in the prior month.