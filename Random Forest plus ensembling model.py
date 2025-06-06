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

# In[12]:


# Create a new dataframe with only the 'Close column
data = df.filter(['Close'])
# Convert the dataframe to a numpy array
dataset = data.values


# In[13]:


dataset


# In[14]:


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


# In[15]:


# Parameters
time_step = 100
training_size = int(len(scaled_data) * 0.9)
test_size = len(scaled_data) - training_size
train_data, test_data = scaled_data[0:training_size,:], scaled_data[training_size:len(scaled_data),:]


# In[16]:


X_train, y_train = create_dataset(train_data, time_step, 1)
X_test, y_test = create_dataset(test_data, time_step, 0)
print( "X_train shape: ", X_train.shape, "y_train shape: ", y_train.shape, "X_test shape: ", X_test.shape, "y_test shape: ", y_test.shape)


# In[17]:


y_predict = dataset[int(len(scaled_data) * 0.9)-1:-1, :]

rmse = np.sqrt(np.mean(((y_predict - y_test) ** 2)))
rmse


# In[18]:


# Plot the data
train = data[:int(len(scaled_data) * 0.9)]
valid = data[int(len(scaled_data) * 0.9):]
valid['Predictions'] = y_predict
# Visualize the data
plt.figure(figsize=(16,6))
plt.title('Naïve Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'], linewidth=2)
plt.plot(valid[['Close', 'Predictions']], linewidth=2)
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()


# In[19]:


from sklearn.ensemble import RandomForestRegressor

# 設定模型參數
model = RandomForestRegressor(
    n_estimators=100,         # 樹的數量
    max_depth=None,           # 每棵樹的最大深度
    min_samples_split=2,      # 分裂節點所需的最小樣本數
    min_samples_leaf=1,       # 葉子節點所需的最小樣本數
    max_features='auto',      # 每次分裂時考慮的最大特徵數
    bootstrap=True,           # 是否使用自助抽樣法
    random_state=42           # 隨機種子
)
# Train the model
model.fit(X_train, y_train) #沒有batch size epoch


# In[20]:


# Print the length of the training and test sets
print(f"Length of training set: {len(X_train)}")
print(f"Length of test set: {len(X_test)}")

import numpy as np
from sklearn.metrics import mean_squared_error

# Make predictions on the test set
test_predict = model.predict(X_test)

# Reshape predictions to 2D array
test_predict = test_predict.reshape(-1, 1)

# Inverse transform the predictions
test_predict = scaler.inverse_transform(test_predict)

# Reshape predictions to 1D array for evaluation
test_predict = test_predict.reshape(-1)


# Calculate the root mean squared error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, test_predict))
print(f"Test RMSE: {rmse}")

# Make predictions on the training set
train_predict = model.predict(X_train)

# Reshape and inverse transform the training predictions
train_predict = train_predict.reshape(-1, 1)
train_predict = scaler.inverse_transform(train_predict)
train_predict = train_predict.reshape(-1)

# The test predictions have already been handled above, so no need to handle them again


# # Data Preprocessing_type2

# In[22]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 假設你的資料已經讀入名為df的DataFrame中
# 這裡是模擬讀入資料，實際情況中應該是你自己的DataFrame
# df = pd.read_csv('your_data.csv')

# 創建特徵和標籤
df['Open_prev'] = df['Open'].shift(1)
df['High_prev'] = df['High'].shift(1)
df['Low_prev'] = df['Low'].shift(1)
df['Close_prev'] = df['Close'].shift(1)
df['Close_next'] = df['Close'].shift(-1)

# 去除缺失值
df = df.dropna()

# 定義特徵和標籤
features = ['Open_prev', 'High_prev', 'Low_prev', 'Close_prev']
X = df[features]
y = df['Close_next']

# 分割訓練和測試集
train_size = int(len(df) * 0.9)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# MinMaxScaler歸一化處理
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 檢查數據大小
print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')


# In[23]:


y_train

Random Forest
# In[24]:


from sklearn.ensemble import RandomForestRegressor

# 設定模型參數
model = RandomForestRegressor(
    n_estimators=100,         # 樹的數量
    max_depth=None,           # 每棵樹的最大深度
    min_samples_split=2,      # 分裂節點所需的最小樣本數
    min_samples_leaf=1,       # 葉子節點所需的最小樣本數
    max_features='auto',      # 每次分裂時考慮的最大特徵數
    bootstrap=True,           # 是否使用自助抽樣法
    random_state=42           # 隨機種子
)
# Train the model
model.fit(X_train, y_train) #沒有batch size epoch


# In[25]:


# Get the models predicted price values
predictions = model.predict(X_test)

predictions = predictions.reshape(-1)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
print(f"Test RMSE: {rmse}")



# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)


# In[30]:


import pandas as pd
import matplotlib.pyplot as pl
# 先將 valid DataFrame 按照日期排序
valid = valid.sort_index()

# 視覺化數據
plt.figure(figsize=(16,6))
plt.title('Random Forest Model with Type 2 Input')  # 調整標題
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(valid.index, valid['Close'], label='Val', linewidth=2)
plt.plot(valid.index, valid['Predictions'], label='Predictions', linewidth=2)
plt.legend(['Val', 'Predictions'], loc='lower right')
plt.show()


# In[27]:


# Show the valid and predicted prices
valid


# use ensembling

# In[31]:


from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.ensemble import RandomForestRegressor

# 定義第一個模型 - LSTM 模型
def create_lstm_model():
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 創建 LSTM 模型
lstm_model = create_lstm_model()

# 訓練 LSTM 模型
lstm_model.fit(X_train, y_train, batch_size=1, epochs=1)

# 創建隨機森林模型
rf_model = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto', bootstrap=True, random_state=42)

# 訓練隨機森林模型
rf_model.fit(X_train, y_train)

# 使用兩個模型進行預測
predictions_lstm = lstm_model.predict(X_test)
predictions_rf = rf_model.predict(X_test)

# 將兩個模型的預測結果進行平均
ensemble_predictions = (predictions_lstm + predictions_rf) / 2


# In[34]:


import pandas as pd
import matplotlib.pyplot as plt

# 假設 train_predict 是從模型獲得的訓練集預測結果，test_predict 是測試集預測結果
# 例如：train_predict = model.predict(X_train)，test_predict = model.predict(X_test)

# 更新 valid DataFrame，確保它包含 Predictions
valid_index = data.index[len(X_train)+(time_step)+1:len(scaled_data)-1]

# 確保valid_index的長度與ensemble_predictions相同
valid = pd.DataFrame(index=valid_index, columns=['Close', 'Predictions'])

valid['Close'] = data['Close'].values[len(X_train)+(time_step)+1:len(scaled_data)-1]
valid['Predictions'] = ensemble_predictions[:len(valid_index)]  # 將ensemble_predictions調整為與valid_index相同的長度

# 畫圖
plt.figure(figsize=(16,6))
plt.title('ensemble learning')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)

# 畫驗證數據的真實值和預測值
plt.plot(valid['Close'], linewidth=2)
plt.plot(valid['Predictions'], linewidth=2)

plt.legend(['Val', 'Val Predictions'], loc='lower right')
plt.show()


# In[ ]:




