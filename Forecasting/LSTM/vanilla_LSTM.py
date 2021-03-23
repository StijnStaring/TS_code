"""
This file is implementing the vanilla LSTM.
The test set used are the 31 days of December of a single time-serie of the 261 time-series with measurements of the full year 2017.
Missing days are estimated.
Naive basemodels that are used are the mean forecast and the MAPE-minimization.
Evaluation metrics used: MSE, RMSE, NRMSE, MAE
####################################################

calculate the MSE on the total test set and for each day of the week.
Should normalize history and temperature with min max
Rest: weekday, time, holiday use one hot encoder
Stack different LSTM blocks --> hidden nodes serve as inputs for next LSTM's
Use finally a fully connected layer to generate an output
Use different learning rules: adam!!, stochastic gradient descent, Adagrad, Adadelta and RMSProp
Adam has parameters: learning rate, momentum and decay
play with mini batches

Inputs:
1. history ok
2. temp ok
3. day week ok
4. time ok
5. holiday
6. (previous week future lag)
7. previous week error history lag --> how good are you following previous week
just the difference between the values
8. previous weeks load at the same moment in time and the temperatures of these days
"""
import datetime as dt
import pandas as pd # pandas
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import casadi as ca
from Test_basemodel_functions import *

plt.rc('axes', linewidth=2)
plt.rc('axes', labelsize= 16)
plt.rc('axes',titlesize = 18)
plt.rc('legend',fontsize=14)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('figure',figsize=(10,8))

# importing the data
fullYeardata = pd.read_csv("D:\Onedrive\Leuven\Final project\data\Forecasting_writtendata\FullYear.csv",index_col= "date",parse_dates= True)
av_temperature = pd.read_csv("D:\Onedrive\Leuven\Final project\data\weather-avg.csv",index_col='meter_id')
av_temperature = av_temperature.transpose()
av_temperature.index = pd.to_datetime(av_temperature.index)


name = fullYeardata.columns[0]
TS = fullYeardata[name]
temperature = av_temperature[name]


# subsitute the missing values by prev week < prev day < mean of all the values at that time of the day (rarely used)
def substitute_missing_values(TS: pd.Series):
    for date in TS.index:
        if np.isnan(TS[date]):
            prev_week =  date + dt.timedelta(days=-7)
            prev_day = date + dt.timedelta(days=-1)
            if not np.isnan(TS[prev_week]):
                TS[date] = TS[prev_week]

            elif not np.isnan(TS[prev_day]):
                TS[date] = TS[prev_day]

            else:
                temp = TS[TS.index.hour == date.hour]
                data = temp[temp.index.minute == date.minute]
                TS[date] = data.mean()
    print("amount of missing values: %s. \n"%TS.isnull().sum())
    return TS


# normalize the data --> min/max method (when using a single time serie)
temperature_norm = norm(temperature)
TS_norm = norm(TS)
temp = TS_norm[TS_norm.index.month != 12]
# training = temp[temp != 11]
# validation = TS_norm[TS_norm.index.month == 11]
# test = TS_norm[TS_norm.index.month == 12]
# remove from the test set all the days that contain nan values -> only estimate real days
training = TS_norm[0:335]
print(training)
test.dropna(inplace=True)
# substitute the missing values
training = substitute_missing_values(training)


def get_av_temp(time_stamp: pd.Timestamp,temperature: pd.Series)-> float:
    return temperature[temperature.index.dayofyear == time_stamp.dayofyear]

def get_weekday(time_stamp: pd.Timestamp)-> int:
    return time_stamp.weekday()

def get_daytime(time_stamp: pd.Timestamp):
    value = time_stamp.hour*2
    if time_stamp.minute == 30:
        value = value + 1
    return value

def is_holiday(time_stamp: pd.Timestamp, holidays: pd.DatetimeIndex)->bool:
    return any(time_stamp.dayofyear == holidays.dayofyear)


# forecast one time stamp at the time --> for loop to forecast the 48 time stamps
def input_output_LSTM(training, temperature_norm, lag_value: int):
    holidays = EnglandAndWalesHolidayCalendar().holidays(start=pd.Timestamp('2017-01-01'),end=pd.Timestamp('2017-12-31'))
    amount_forecasts = len(training) - lag_value
    amount_features = 1+1+7+48+2
    X = np.zeros((amount_forecasts,amount_features,lag_value))
    y = np.zeros((amount_forecasts,1))

    for sample_id in np.arange(0,amount_forecasts):

        history = training.values[sample_id:lag_value + sample_id]
        y[sample_id] = training.values[lag_value+1]
        X[sample_id,0,:] = history
        all_time_stamps = training.index[sample_id:lag_value + sample_id]
        for lag_value_index in np.arange(0,lag_value): # goes from old to new
            time_stamp = all_time_stamps[lag_value_index]

            temperature_feed = get_av_temp(time_stamp,temperature_norm)
            X[sample_id,1,lag_value_index] = temperature_feed

            weekday = np.zeros(7)
            day = get_weekday(time_stamp) # 0 - 6
            weekday[day] = 1
            X[sample_id,2:9,lag_value_index] = weekday

            time = np.zeros(48)
            time_of_day = get_daytime(time_stamp) # 0 - 47
            time[time_of_day] = 1
            X[sample_id,9:57,lag_value_index] = time

            holiday = np.zeros(2)
            hol = is_holiday(time_stamp,holidays) # [no, yes]
            if hol:
                holiday[1] = 1
            else:
                holiday[0] = 1
            X[sample_id,57:59,lag_value_index] = holiday


    return X,y


from keras.layers import Dense,  LSTM,Embedding
from keras.models import Sequential
from keras import regularizers
from keras.callbacks import EarlyStopping
def build_model(training: pd.Series,validation: pd.Series, temperature_norm: pd.Series, lag_value: int,regularization_parameter = 0.01,batch_size_para: int = 32,verbose_para: int = 2,nb_epoch: int = 150 ):
    X,y = input_output_LSTM(training, temperature_norm, lag_value)
    X_val,y_val = input_output_LSTM(validation, temperature_norm, lag_value)
    regularizers.l2(l=regularization_parameter)
    model = Sequential()
    # no initial state is given --> hidden state are tensors filled with zeros

    model.add(LSTM(units=20,activation='tanh',dropout=0,recurrent_dropout=0,input_shape=()))
    model.add(Dense(units=20,activation='relu',kernel_regularizer='l2',input_shape=(X.shape[2],X.shape[1])))
    # model.add(Dropout(0.10))
    model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
    early_stopping_monitor = EarlyStopping(patience=2,restore_best_weights=True)
    model.fit(x=X,y=y,nb_epoch=nb_epoch,batch_size=batch_size_para,validation_data=(X_val,y_val),callbacks=[early_stopping_monitor],verbose=verbose_para)

    return model


# run the predictor ``warm'' on the trainingset --> before going to do predictions (remember matlab file)
build_model(training,validation,temperature_norm,16,0.01,32,2,1)
