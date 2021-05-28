
import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import datetime as dt
import pandas_datareader as web
import sys

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

st.title('NIFTY50 App')


st.sidebar.header('Select a sector')


# Web scraping of NIFTY50 data
#
@st.cache
def load_data():
    url = 'https://en.wikipedia.org/wiki/NIFTY_50'
    html = pd.read_html(url, header = 0)
    df = html[1]
    
    return df

df = load_data()
sector = df.groupby('Sector')


# Sidebar - Sector selection
sorted_sector_unique = sorted( df['Sector'].unique() )
selected_sector = 0
selected_sector = st.multiselect('Sector', sorted_sector_unique, sorted_sector_unique)

# Filtering data
df_selected_sector = df[ (df['Sector'].isin(selected_sector)) ]


if df_selected_sector.empty:
    st.write('enter a sector')
    st.stop()
    sys.exit("dataframe is empty")
    
    
st.header('Display Companies in Selected Sector')
st.write('Data Dimension: ' + str(df_selected_sector.shape[0]) + ' rows and ' + str(df_selected_sector.shape[1]) + ' columns.')
st.dataframe(df_selected_sector)



# Download S&P500 data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="SP500.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df_selected_sector), unsafe_allow_html=True)


    
# https://pypi.org/project/yfinance/

data = yf.download(
        tickers = list(df_selected_sector[:5].Symbol),
        period = "ytd",
        interval = "1d",
        group_by = 'ticker',
        auto_adjust = True,
        prepost = True,
        threads = True,
        proxy = None
    )


# Plot Closing Price of Query Symbol
def price_plot(symbol):
  df = pd.DataFrame(data[symbol].Close)
  df['Date'] = df.index
  fig, ax = plt.subplots()
  plt.fill_between(df.Date, df.Close, color='skyblue', alpha=0.3)
  plt.plot(df.Date, df.Close, color='skyblue', alpha=0.8)
  plt.xticks(rotation=90)
  plt.title(symbol, fontweight='bold')
  plt.xlabel('Date', fontweight='bold')
  plt.ylabel('Closing Price', fontweight='bold')
  return st.pyplot(fig)


num_company = st.slider('Number of Companies', 1, 5)

st.header('To see how the stock performed for the last few days - click show plots button')
if st.button('Show Plots'):
    st.header('Stock Closing Price')
    for i in list(df_selected_sector.Symbol)[:num_company]:
        price_plot(i)
user_input1 = st.number_input("enter a epochs value", 5)
user_input2 = st.number_input("enter a batch_size", 64)

st.header('To see how the stock will perform tomorrow - click this button')
if st.button('ML model'):
 st.header('This model can be made even more accurate by increasing epoch to 25. Note that, it will take more time to calculate')
 for H in  list(df_selected_sector[:5].Symbol)[:num_company]:
  print(H)
# Load Data

  company = H

  start = dt.datetime(2015,1,1)
  end = dt.datetime(2020, 3, 1)

  data = web.DataReader(company, 'yahoo' , start, end)

#prepare data

  scaler = MinMaxScaler(feature_range=(0,1))

  scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

  prediction_days = 60

  x_train =[]
  y_train = []
 
 

  for x in range(prediction_days, len(scaled_data)):
      x_train.append(scaled_data[x-prediction_days:x, 0])
      y_train.append(scaled_data[x, 0])
    

  x_train, y_train = np.array(x_train), np.array(y_train)
  x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# Build the Model

  model = Sequential()

  model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
  model.add(Dropout(0.2))
  model.add(LSTM(units=50, return_sequences=True))
  model.add(Dropout(0.2))
  model.add(LSTM(units=50))
  model.add(Dropout(0.2))
  model.add(Dense(units=1))


  model.compile(optimizer='adam' , loss='mean_squared_error')

  model.fit(x_train, y_train, epochs = user_input1 , batch_size = user_input2)



# testing the model


  test_start = dt.datetime(2020,3,1)
  test_end = dt.datetime.now()

  test_data = web.DataReader(company, 'yahoo', test_start, test_end)

  actual_prices = test_data['Close'].values

  total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)


  model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
  model_inputs = model_inputs.reshape(-1, 1)
  model_inputs = scaler.transform(model_inputs)

#make prediction on test data.

  x_test = []

  for x in range(prediction_days, len(model_inputs)):
      x_test.append(model_inputs[x-prediction_days:x, 0])
    
  x_test = np.array(x_test)
  x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

  predicted_prices = model.predict(x_test)
  predicted_prices = scaler.inverse_transform(predicted_prices)
 

#plot the test predictions

  fig1, ax1 = plt.subplots()
  plt.plot(actual_prices, color = "black", label = f"Actual price")
  plt.plot(predicted_prices, color="green", label = f"Prediction price" )
  plt.title(f"{company} share price")
  plt.xlabel('Time')
  plt.ylabel(f'{company} share price')
  plt.legend()
  st.pyplot(fig1)
  fig1 = 0
 
 #predict next day
 
  real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs+1), 0]]
  real_data = np.array(real_data)
  real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1],1))
 
  prediction = model.predict(real_data)
  prediction = scaler.inverse_transform(prediction)
  st.write("Price prediction for tomorrow")
  st.write(prediction)

