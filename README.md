# Stock Price Prediction: Deep Learning Application
## Project Introduction
This project aims to predict stock prices using deep learning models. Given the vastness of stock market data, the diversity of influencing factors, and the nonlinear and complex patterns of stock price fluctuations, deep learning offers powerful tools to capture these intricate patterns. Our goal is to utilize historical price information (such as opening price, closing price, high price, low price, and trading volume) to predict future closing prices.

## Data and Preprocessing
This project acquires stock market data from Yahoo Finance , and conducts predictions for both the overall market (Vanguard Total Stock Market Index Fund ETF, VTI) and individual stocks (Microsoft Corporation, MSFT). The raw data undergoes Min-Max standardization to ensure efficient model training and stable data distribution. We explored two types of input data: sequential input based on the closing prices of the previous K days , and input based on single-day, multi-dimensional information. Data for training spans from January 1, 2010, to May 1, 2023, with testing data from May 1, 2023, to the present.


## Model Introduction
The following models were implemented and evaluated to predict stock prices:

**Naïve Model**: Serving as a baseline, this model directly uses the previous day's closing price as the prediction for the current day's closing price. Based on the Random Walk Theory, it provides a simple and conservative benchmark for comparison.
**Transformer Model**: This model utilizes its self-attention mechanism to capture long-range dependencies and complex patterns within stock price time series. We experimented with different parameter settings and numbers of layers to optimize its prediction performance.
**LSTM (Long Short-Term Memory) Model**: A type of recurrent neural network particularly well-suited for time series data. We experimented with various LSTM architectures, including directly predicting the closing price , predicting daily stock price change rates , and predicting the difference between the stock price and moving averages over different periods, to explore its performance under various data transformations.
**Random Forest Model**: This is an ensemble learning method that improves prediction accuracy and robustness by constructing multiple decision trees and combining their predictions. It effectively handles outliers by introducing diversity through random sampling and random feature selection.
**DNN (Deep Neural Network)**: As a multi-layered fully connected neural network, DNN is used to learn complex nonlinear relationships between input features and target stock prices.
**Ensembling Model**: This approach combines the prediction results from multiple different models. By integrating the strengths of various models, the ensemble model aims to provide more stable and accurate final predictions, compensating for the limitations of individual models.