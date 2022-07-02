
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from keras.models import load_model 
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# start = '2010-01-01'
# end = '2019-12-31'


st.title('Stock Price Prediction')
user_input = st.text_input('Enter Stock Ticker' , 'AAPL')
st.subheader('Data from last 5 years')

### Data Collection
key="4ca710d54195ce27657bfc15fcb00c2a6dd6b636"
df = pdr.get_data_tiingo(user_input, api_key=key)
st.write(df)

df1=df.reset_index()['close']

# print(df1)

###############################################################

st.subheader('Stock Price Data of past 5 years')
fig = plt.figure(figsize = (12,6))
plt.plot(df1)
st.pyplot(fig)

###############################################################





#    # minmax scaler brings every value to 0 to 1 range or any range
# scaler=MinMaxScaler(feature_range=(0,1))         # formula used is Xscaled = (x - xmin)/(xmax-xmin)
# df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
# training_size=int(len(df1) - 200)
# test_size=len(df1)-training_size
# train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]



# # plt.plot(df1)
# # # convert an array of values into a dataset matrix 
# # # we have a sereis of data and we need to convert that into sereis of input and output like xi -> yi 1 2 3 -> 2.2
# # # for that we take a time step i.e how many prev values our output will be dependent on 
# # # so for example 1 2 3 4 2 3 1 data with time step 3 is
# # #  x         y
# #   #  1 2 3    4
# #   #  2 3 4    2
# #   #  3 4 2    3
# #   #  4 2 3    1
  
# def create_dataset(dataset, time_step=1):
  
# 	dataX, dataY = [], []
# 	for i in range(len(dataset)-time_step-1):
# 		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
# 		dataX.append(a)
# 		dataY.append(dataset[i + time_step, 0])
    
# 	return np.array(dataX), np.array(dataY)
 

# time_step = 100
# X_train, y_train = create_dataset(train_data, time_step)
# X_test, ytest = create_dataset(test_data, time_step)

# X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1) # same dimension data now converted in 3d where 3d is 1
# X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# model=Sequential()
# model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))   # input shape is last 2 dim of x_train
# model.add(LSTM(50,return_sequences=True))                       # we are adding hidden layers in stacked manner
# model.add(LSTM(50))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error',optimizer='adam')



# model.fit(X_train , y_train ,validation_data=(X_test,ytest) ,   epochs = 50 , batch_size = 64 , verbose = 1)








# import tensorflow as tf
# train_predict=model.predict(X_train)
# test_predict=model.predict(X_test)

# train_predict=scaler.inverse_transform(train_predict)
# test_predict=scaler.inverse_transform(test_predict)

# import math
# from sklearn.metrics import mean_squared_error
# math.sqrt(mean_squared_error(y_train,train_predict))


# look_back=100
# trainPredictPlot = np.empty_like(df1)
# trainPredictPlot[:, :] = np.nan
# trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# testPredictPlot = np.empty_like(df1)
# testPredictPlot[:, :] = np.nan
# testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict


# ##################################################################
# st.subheader('Actual Data vs predicted on training and testing')
# fig = plt.figure(figsize = (12,6))
# plt.plot(scaler.inverse_transform(df1))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# st.pyplot(fig)
# plt.show()

# ##################################################################









# last100Days = df1[len(df1) - 100 : ]

# ##################################################################
# st.subheader('Last 100 days of actual data vs predicted data')
# fig = plt.figure(figsize = (12,6))
# plt.plot(scaler.inverse_transform(last100Days))
# plt.plot(test_predict)
# st.pyplot(fig)

# ##################################################################






# # till now we have predicted for ytest and can calcilte profit/loss accordingly
# # now we predict for next 100 days

# x_input=test_data[len(test_data) -100 :].reshape(1,-1)
# temp_input=list(x_input)
# temp_input=temp_input[0].tolist()

# from numpy import array

# lst_output=[]
# n_steps=100
# i=0
# while(i<100):
    
#     if(len(temp_input)>100):
#         x_input=np.array(temp_input[1:])
#         x_input=x_input.reshape(1,-1)
#         x_input = x_input.reshape((1, n_steps, 1))
#         yhat = model.predict(x_input, verbose=0)
#         temp_input.extend(yhat[0].tolist())
#         temp_input=temp_input[1:]
#         lst_output.extend(yhat.tolist())
#         i=i+1
#     else:
#         x_input = x_input.reshape((1, n_steps,1))
#         yhat = model.predict(x_input, verbose=0)
#         temp_input.extend(yhat[0].tolist())
#         lst_output.extend(yhat.tolist())
#         i=i+1
    

# day_new=np.arange(1,101)
# day_pred=np.arange(101,201)

# ########################################################################

# st.subheader('Future 100 days predicton')
# fig = plt.figure(figsize = (12,6))
# plt.plot(day_new,scaler.inverse_transform(df1[len(df1) - 100:]))
# plt.plot(day_pred,scaler.inverse_transform(lst_output))
# st.pyplot(fig)

# ########################################################################




# df3=df1.tolist()
# df3.extend(lst_output)
# df3=scaler.inverse_transform(df3).tolist()




# ########################################################################
# st.subheader('Overall Dataset with 100 days prediction')
# fig = plt.figure(figsize = (12,6))
# plt.plot(df3)
# st.pyplot(fig)
# ########################################################################

