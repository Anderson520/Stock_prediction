#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -q yfinance')
get_ipython().system('pip install pandas-datareader')


# In[2]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
get_ipython().run_line_magic('matplotlib', 'inline')

# For reading stock data from yahoo
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr

yf.pdr_override()

# For time stamps
from datetime import datetime


# The tech stocks we'll use for this analysis
tech_list = ['VTI']

# Set up End and Start times for data grab
tech_list = ['VTI']

end = datetime.now()
start = datetime(2010, 1, 1)

for stock in tech_list:
    globals()[stock] = yf.download(stock, start, end)


company_list = [VTI]
company_name = ["Vanguard Total Stock Market Index Fund ETF"]

for company, com_name in zip(company_list, company_name):
    company["company_name"] = com_name

df = pd.concat(company_list, axis=0)
df.tail(10)


# In[3]:


# Summary Stats
VTI.describe()


# In[4]:


# General info
VTI.info()


# In[5]:


plt.figure(figsize=(15, 10))
plt.subplots_adjust(top=1.25, bottom=1.2)

for i, company in enumerate(company_list, 1):
    plt.subplot(2, 2, i)
    company['Adj Close'].plot(linewidth=1.5)  # 設置線條寬度為1.5
    plt.ylabel('Adj Close')
    plt.xlabel(None)
    plt.title(f"Closing Price of {tech_list[i - 1]}")

plt.tight_layout()


# In[6]:


# Now let's plot the total volume of stock being traded each day
plt.figure(figsize=(15, 10))
plt.subplots_adjust(top=1.25, bottom=1.2)

for i, company in enumerate(company_list, 1):
    plt.subplot(2, 2, i)
    company['Volume'].plot(linewidth=1.5)
    plt.ylabel('Volume')
    plt.xlabel(None)
    plt.title(f"Sales Volume for {tech_list[i - 1]}")

plt.tight_layout()


# In[7]:


ma_day = [10, 20, 50]

for ma in ma_day:
    column_name = f"MA for {ma} days"
    company_list[0][column_name] = company_list[0]['Adj Close'].rolling(ma).mean()

plt.figure(figsize=(15, 10))

# 繪製 Adj Close 和移動平均線圖，並調整線條粗細
plt.plot(company_list[0]['Adj Close'], label='Adj Close', linewidth=2)
plt.plot(company_list[0]['MA for 10 days'], label='MA for 10 days', linewidth=2)
plt.plot(company_list[0]['MA for 20 days'], label='MA for 20 days', linewidth=2)
plt.plot(company_list[0]['MA for 50 days'], label='MA for 50 days', linewidth=2)

plt.title('Vanguard Total Stock Market Index Fund ETF')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

plt.tight_layout()
plt.show()


# In[8]:


# 使用 pct_change 找到每日的百分比變化
for company in company_list:
    company['Daily Return'] = company['Adj Close'].pct_change()

plt.figure(figsize=(15, 10))

# 繪製每日回報百分比圖，並設置線條粗細為2
VTI['Daily Return'].plot(legend=True, linestyle='--', marker='o', linewidth=2)
plt.title('Vanguard Total Stock Market Index Fund ETF')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.legend()

plt.tight_layout()
plt.show()


# In[9]:


plt.figure(figsize=(12, 9))

for i, company in enumerate(company_list, 1):
    plt.subplot(2, 2, i)
    company['Daily Return'].hist(bins=50)
    plt.xlabel('Daily Return')
    plt.ylabel('Counts')
    plt.title(f'{company_name[i - 1]}')

plt.tight_layout()


# In[10]:


# Get the stock quote
df = pdr.get_data_yahoo('VTI', start=start, end=datetime.now())
# Show teh data
df


# In[11]:


plt.figure(figsize=(16,6))
plt.title('Close Price History')
plt.plot(df['Close'], linewidth=2)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()


# # data processing 1

# In[28]:


# Create a new dataframe with only the 'Close column
data = df.filter(['Close'])
# Convert the dataframe to a numpy array
dataset = data.values


# In[29]:


dataset


# In[35]:


# Scale the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

print(scaled_data.shape)
scaled_data


def create_dataset(dataset, time_step=1, if_training_dataset = 1):
    dataX, dataY = [], []
    if if_training_dataset == 1:
        for i in range(len(dataset) - 1):
            a = scaled_data[i:(i + time_step), 0]
            dataX.append(a)
            dataY.append(scaled_data[i + time_step, 0])
    else:
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


# In[36]:


# Parameters
time_step = 100
training_size = int(len(scaled_data) * 0.9)
test_size = len(scaled_data) - training_size
train_data, test_data = scaled_data[0:training_size,:], scaled_data[training_size:len(scaled_data),:]


# In[37]:


X_train, y_train = create_dataset(train_data, time_step, 1)
X_test, y_test = create_dataset(test_data, time_step, 0)
print( "X_train shape: ", X_train.shape, "y_train shape: ", y_train.shape, "X_test shape: ", X_test.shape, "y_test shape: ", y_test.shape)


# In[38]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the model
model = Sequential([
    Dense(units=64, activation='relu', input_shape=(time_step,)),
    Dense(units=32, activation='relu'),
    Dense(units=1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Reshape the data for ANN
X_train_ann = X_train.reshape(X_train.shape[0], -1)
X_test_ann = X_test.reshape(X_test.shape[0], -1)

# Train the model
model.fit(X_train_ann, y_train, epochs=100, batch_size=32)

# Evaluate the model
test_loss = model.evaluate(X_test_ann, y_test)
print("Test Loss:", test_loss)


# In[41]:


# Get the models predicted price values
predictions = model.predict(X_test)

predictions = predictions.reshape(-1)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
print(f"Test RMSE: {rmse}")



# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)


# In[42]:


import pandas as pd
import matplotlib.pyplot as plt

# Create DataFrame for visualization
df_pred = pd.DataFrame({'Predictions': predictions, 'Actual': y_test})

# Plot the data
plt.figure(figsize=(16, 6))
plt.plot(df_pred.index, df_pred['Actual'], label='Actual', linewidth=2)
plt.plot(df_pred.index, df_pred['Predictions'], label='Predictions', linewidth=2)
plt.title('Predictions vs Actual')
plt.xlabel('Index', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.legend()
plt.show()

