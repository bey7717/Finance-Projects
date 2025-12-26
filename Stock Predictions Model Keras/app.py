import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt

model = tf.keras.models.load_model(r'C:\Users\bey77\OneDrive\Desktop\Projects\FinanceProj\Stock Predictions Model Keras\Stock Predictions Model.keras')
# model = tf.keras.models.load_model(r'C:\Users\bey77\PycharmProjects\ML\Stock Predictions Model Keras\Stock Predictions Model.keras')

st.header('Stock Market Predictor')
stock = st.text_input('Enter Stock Ticker', 'MSFT')
start = '2012-01-01'
end = '2020-01-01'

data = yf.download(stock, start, end)
st.subheader('Stock Data')
st.write(data)

# slicing with the data
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data_train)

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs Moving Average for 50 Days')
st.subheader('Price = Green; MA 50 = Red')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA 50 days vs MA 100 days')
st.subheader('Price = Green; MA 50 = Red; MA 100 = Blue')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)


st.subheader('Price vs MA 50 days vs MA 100 days vs MA 200 days')
st.subheader('Price = Green; MA 50 = Red; MA 100 = Blue, MA 200 = Yellow')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(ma_200_days, 'y')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3)

x = []
t = []
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i - 100: i])
    t.append(data_test_scale[i, 0])
x, t = np.array(x), np.array(t)

pred = model.predict(x)
scale = 1/scaler.scale_
pred = pred * scale
t = t * scale


st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8, 6))
plt.plot(pred, 'r', label='Original Price')
plt.plot(t, 'g', label='Predicted Price')
plt.xlabel('Time (in days)')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)


