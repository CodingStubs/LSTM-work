# Set Up libraries
from pandas_datareader import data as pdr
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow.keras import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from fredapi import Fred
from keras.optimizers.legacy import Adam
import yfinance as yf
from keras import backend as K
import tensorflow as tf
import pandas_ta as ta

"""
This project was inspired by the article listed here:
https://www.sciencedirect.com/science/article/pii/S2666827022000378#tbl2
"""

fred = Fred(api_key='3b7e7d31bcc6d28556c82c290eb3572e')

yf.pdr_override()
 
plt.style.use('fivethirtyeight')

# Define the loss function to be used in model training
def loss_function(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    RMSE = K.sqrt(K.mean(K.square(y_pred - y_true)))
    return RMSE
    

# Get Macro-Economic Data from the FRED API and Yahoo Tickers
def get_macro_data():
    df = pd.DataFrame()
    
    VIX = pdr.get_data_yahoo('^VIX') #CBOE Volatility index
    df.insert(loc=0, column='VIX', value=VIX['Adj Close'])
                 
    EFFR = fred.get_series('DFF') # Federal Funds Rate (Interest Rate)
    df.insert(loc=1, column='EFFR', value=EFFR)
    
    UNRATE = fred.get_series('UNRATE') # Unemployment Rate
    df.insert(loc=2, column='UNRATE', value=UNRATE)
    
    UMCSENT = fred.get_series('UMCSENT') # Consumer Sentiment Index
    df.insert(loc=3, column='UMCSENT', value=UMCSENT)
    
    USDX = pdr.get_data_yahoo('DX-Y.NYB') # US Dollar index
    df.insert(loc=4, column='USDX', value=USDX['Adj Close'])

    df = df.ffill() #Forward fill for Monthly Macroeconomic data
    
    return df

# Retrieve data for desired stock
def load_data():
    stock = '^GSPC'
    all_data = []
    
    start = '2006-01-01'
    end = '2022-12-31'
    
    try:
        macro_data = get_macro_data()
        
        df = pdr.get_data_yahoo(stock, start, end)
        df.insert(loc=0, column='Ticker', value=stock)
        
        df.ta.macd(append=True) # MACD_12_26_9  MACDh_12_26_9  MACDs_12_26_9
        df.ta.atr(append=True)
        df.ta.rsi(append=True) # RSI_5
        
        #Custom variables not from article
        df.ta.sma(length=10, append=True) # SMA_10
        df.ta.cci(length=24, append=True) # CCI_24_0.015
        df.ta.mom(append=True) # MOM_10
        df.ta.roc(append=True) # ROC_10
        df.ta.rsi(length=5, append=True) # RSI_5
        df.ta.willr(length=9, append=True) #WILLR_9

        df = pd.concat([df, macro_data], axis=1)
        df = df.dropna()
        df = df[["Ticker", "Adj Close", "MACD_12_26_9", "ATRr_14", "RSI_14", "VIX", "EFFR", "UNRATE", "UMCSENT", "USDX",
                 "SMA_10", "CCI_24_0.015", "MOM_10", "ROC_10", "RSI_5", "WILLR_9"]]
        #df = df[["Ticker", "Adj Close", "MACD_12_26_9", "ATRr_14", "RSI_14", "VIX", "EFFR", "UNRATE", "UMCSENT", "USDX"]]
        
        all_data.append(df)
        
    except Exception as e:
        print(e)
        print(f"Could not gather data on {stock}")
        
    
    return all_data 
        
    
stock_data = load_data()
stock_data = stock_data[0]

all_features = []

#Scale the data
scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

original_index = stock_data.index
original_columns = stock_data.columns
ticker = stock_data.iloc[0, 0]

# Select only the numeric columns for scaling
numeric_data = stock_data.select_dtypes(include=[np.number])

close_prices = numeric_data.iloc[:, 0]
scaled_data = numeric_data.iloc[:, :]

# Scale the numeric data
scaled_data = scaler.fit_transform(scaled_data)

close_prices = np.array(close_prices)
close_prices = close_prices.reshape(-1, 1)
y_scaled = y_scaler.fit_transform(close_prices)

# Create a DataFrame from the scaled data
scaled_df = pd.DataFrame(scaled_data, columns=numeric_data.columns, index=original_index)
scaled_df.rename(columns={'Adj Close': 'Scaled Adj Close'}, inplace=True)
scaled_df.insert(loc=0, column=numeric_data.columns[0], value=y_scaled)

all_features.append(scaled_df)


df = all_features[0]
train_dates = pd.to_datetime(df.index)

#Variables for training
cols = list(df)[:]

#New dataframe with only training data
df_for_training = df[cols].astype(float)

df_for_training_scaled = all_features[0].to_numpy()

trainX = []
trainY = []

# Define the percentages for train, validation, and test sets
train_percent = 0.80  # 80% for training
test_percent = 0.20  # 20% for testing

# Calculate the split points
num_data_points = len(df_for_training_scaled)
train_end = int(num_data_points * train_percent)
test_start = train_end

# Split the data
train_data = df_for_training_scaled[:train_end]
test_data = df_for_training_scaled[test_start:]

# Optionally, you can also split the corresponding dates for future use
train_dates = pd.to_datetime(df.index[:train_end])
test_dates = pd.to_datetime(df.index[test_start:])

n_future = 1   # Number of days we want to look into the future based on the past days.
n_past = 10  # Number of past days we want to use to predict the future.

for i in range(n_past, len(train_data) - n_future +1):
    trainX.append(train_data[i - n_past:i, 1: ])
    trainY.append(train_data[i + n_future - 1:i + n_future, 0])

trainX, trainY = np.array(trainX), np.array(trainY)

# Define hyperparameters
num_neurons = 150
learning_rate = 0.001
batch_size = 8
epochs = 50

# Compile the LSTM Model
model = Sequential()
model.add(LSTM(num_neurons, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_function,
              metrics=[metrics.RootMeanSquaredError()])

model.summary()

# Fit the model
history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size)

"""
Optional lines to save/load model

from keras.models import load_model
model.save('RMSE_model')
model = load_model('model_1')
"""

# Plot the loss during training
plt.plot(history.history['loss'], label='Training loss')
plt.legend()


# Predict using market calender
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

n_days_for_prediction= len(test_data)  #Predict all points in test data

predict_period_dates = pd.date_range(list(test_dates)[-n_days_for_prediction], periods=n_days_for_prediction, freq=us_bd).tolist()

testX = []
testY = []

for i in range(n_past, len(train_data) - n_future +1):
    testX.append(train_data[i - n_past:i, 1: ])
    testY.append(train_data[i + n_future - 1:i + n_future, 0])

testX, testY = np.array(testX), np.array(testY)

#Make prediction
prediction = model.predict(testX)
#Inverse scale Y values
prediction = y_scaler.inverse_transform(prediction)

prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
y_pred_future = (prediction_copies)[:,0]

# Convert timestamp to date
forecast_dates = []
for time_i in predict_period_dates:
    forecast_dates.append(time_i.date())
    
# Create the Dataframe for predictions
df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Adj Close':y_pred_future[-n_days_for_prediction:]})
df_forecast['Date']=pd.to_datetime(df_forecast['Date'])
    
df.index = pd.to_datetime(df.index)
df_forecast.index = pd.to_datetime(df_forecast.index)

# Get original data to compare to
original = scaler.inverse_transform(scaled_data)
original=pd.DataFrame(original, columns=original_columns[1:], index=original_index)
original = original[original.index >= '2010-11-01']

# Plot the data
plt.figure(figsize=(10, 6))
sns.lineplot(data=original, x=original.index, y='Adj Close', label='Original Data')
sns.lineplot(data=df_forecast, x='Date', y='Adj Close', label='Forecasted Data')
plt.xlabel('Date')
plt.ylabel('Adj Close')
plt.title(f'Adj Close Prices. Using Past {n_past} Days')
plt.legend()
plt.show()