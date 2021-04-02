import datetime as dt
import pandas as pd # pandas
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings as wn
wn.filterwarnings(action='ignore')
from Test_basemodel_functions import EnglandAndWalesHolidayCalendar


# subsitute the missing values by prev week < prev day < mean of all the values at that time of the day (rarely used)
# this function changes the input!
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

def substitute_missing_values_temperature(TS: pd.Series):
    for date in TS.index:
        if np.isnan(TS[date]):
            prev_day = date + dt.timedelta(days=-1)
            next_day = date + dt.timedelta(days=+1)

            if not np.isnan(TS[prev_day]):
                TS[date] = TS[prev_day]

            elif not np.isnan(TS[next_day]):
                TS[date] = TS[next_day]

            else:
                TS[date] = TS.mean()
    print("amount of missing values: %s. \n" % TS.isnull().sum())
    return TS

def norm_forcast(serie: pd.Series):
    values = serie.values
    l = len(values)
    values = values.reshape((l, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(values)
    norm_serie = scaler.transform(values)
    norm_serie = norm_serie.squeeze()
    norm_serie = pd.Series(data=norm_serie,index=serie.index)
    return norm_serie,scaler

# be aware that there are minor calculations rounding errors when transformed back!
def norm_inverse(serie: pd.Series, scaler):
    values = serie.values
    l = len(values)
    values = values.reshape((l, 1))
    norm_serie = scaler.inverse_transform(values)
    norm_serie = norm_serie.squeeze()
    norm_serie = pd.Series(data=norm_serie,index=serie.index)
    return norm_serie



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
    X = np.zeros((amount_forecasts,lag_value,amount_features))
    y = np.zeros((amount_forecasts,1))

    for sample_id in np.arange(0,amount_forecasts):

        history = training.values[sample_id:lag_value + sample_id]
        y[sample_id] = training.values[lag_value + sample_id]
        X[sample_id,:,0] = history
        all_time_stamps = training.index[sample_id:lag_value + sample_id]
        for lag_value_index in np.arange(0,lag_value): # goes from old to new
            time_stamp = all_time_stamps[lag_value_index]

            temperature_feed = get_av_temp(time_stamp,temperature_norm)
            X[sample_id,lag_value_index,1] = temperature_feed

            weekday = np.zeros(7)
            day = get_weekday(time_stamp) # 0 - 6
            weekday[day] = 1
            X[sample_id,lag_value_index,2:9] = weekday

            time = np.zeros(48)
            time_of_day = get_daytime(time_stamp) # 0 - 47
            time[time_of_day] = 1
            X[sample_id,lag_value_index,9:57] = time

            holiday = np.zeros(2)
            hol = is_holiday(time_stamp,holidays) # [no, yes]
            if hol:
                holiday[1] = 1
            else:
                holiday[0] = 1
            X[sample_id,lag_value_index,57:59] = holiday


    return X,y
