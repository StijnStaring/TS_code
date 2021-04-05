import datetime as dt
import pandas as pd # pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings as wn
wn.filterwarnings(action='ignore')
from Test_basemodel_functions import EnglandAndWalesHolidayCalendar
from keras.layers import Dense,  LSTM,Embedding
from keras.models import Sequential, save_model, load_model
from keras import regularizers
from keras.callbacks import EarlyStopping,ModelCheckpoint, History


class forecast_setting:

    def __init__(self, units_LSTM = 20, units_dense = 20, patience = 5, shuffle = False, lag_value = 3, nb_epoch = 1, regularization_parameter = 0.001, batch_size_para = 32, repeat = 10, activation: str = 'tanh', dropout: bool = False, amount_layers = 1):
        self.lag_value = lag_value
        self.nb_epoch = nb_epoch
        self.regularization_parameter = regularization_parameter
        self.batch_size_parameter = batch_size_para
        self.units_LSTM = units_LSTM
        self.units_dense = units_dense
        self.repeat = repeat
        self.evaluation_frame = pd.DataFrame()
        self.activation = activation
        self.patience = patience
        self.shuffle = shuffle


class time_serie:

    def __init__(self, ts: pd.Series, av_temperature: pd.DataFrame, settings: forecast_setting):
        self.name = ts.name
        self.serie = ts
        self.temperature = av_temperature[self.name]
        temperature_norm, scaler_temperature = norm_forcast(self.temperature)
        self.temperature_norm = substitute_missing_values_temperature(temperature_norm)
        TS_norm, scaler_history = norm_forcast(ts)
        temp = TS_norm[TS_norm.index.month == 1]
        temp_true = TS_norm[TS_norm.index.month != 12]
        test_true = TS_norm[TS_norm.index.month == 12]
        # remove from the test set all the days that contain nan values -> only estimate real days
        test_true.dropna(inplace=True)
        # substitute the missing values
        temp = substitute_missing_values(temp)
        training = temp[0:672]
        test = temp[672:912]
        training_true = substitute_missing_values(temp_true)
        TS_norm_full = substitute_missing_values(TS_norm.copy(deep=True))

        self.scaler_temperature = scaler_temperature
        self.scaler_history = scaler_history
        self.TS_norm_full = TS_norm_full
        self.training = training
        self.test = test
        self.training_true = training_true
        self.test_true = test_true

def visualize_loss(history):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    figure_layout(figsize=(10,8),titel="Training and Validation Loss",xlabel="Epochs",ylabel="MSE",grid=False)
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.legend()
    # plt.show()

def generate_training_validation_division(data: np.ndarray, batch_size: int = 32, ratio: float = 0.9):
    # policy is that the oldest data is thrown away.
    # data should be made compatible with the chosen batch size.
    if not len(data.shape) == 3:
        raise Exception("Input data is not of the correct shape: samples x timesteps x features.")
    amount_samples = data.shape[0]
    amount_training = np.around(amount_samples* ratio,0)
    amount_validation = amount_samples - amount_training
    rest_val = amount_validation % batch_size
    if rest_val <= batch_size/2 and (amount_validation - rest_val)//batch_size>=1: # remove validation values
        value = int(amount_validation - rest_val)
        validation = data[-1*value:,:,:]
        less_training = len(data[0:int(amount_samples - value),:,:]) % batch_size
        training = data[int(less_training):int(amount_samples - value), :, :]
        return training, validation

    else: # add training values to the validation set
        value = int(amount_validation + batch_size - rest_val)
        validation = data[-1*value:,:,:]
        less_training = len(data[0:int(amount_samples - value),:,:]) % batch_size
        training = data[int(less_training):int(amount_samples - value), :, :]
        return training, validation

def compile_data(training, validation, temperature_norm, lag_value):
    X, y = input_output_LSTM(training, temperature_norm, lag_value)
    X_val, y_val = input_output_LSTM(validation, temperature_norm, lag_value)

    return (X,y), (X_val, y_val)

# maybe better to use the relu function instead of tanh.
# data is between 0 and one
def build_model_stateless1(setting: forecast_setting, X, y, X_val, y_val, verbose_para: int = 1, save: bool = False):

    history = History()
    regularizers.l2(l=setting.regularization_parameter)
    model = Sequential()
    # no initial state is given --> hidden state are tensors filled with zeros
    # dropout=None,recurrent_dropout=None
    model.add(LSTM(units=setting.units_LSTM,activation= setting.activation,input_shape=(X.shape[1],X.shape[2])))
    model.add(Dense(units=setting.units_dense,activation='relu',kernel_regularizer='l2'))
    # model.add(Dropout(0.10))
    model.add(Dense(units=y.shape[1],activation='relu',kernel_regularizer='l2'))
    model.compile(optimizer='adam',loss='mse')
    early_stopping_monitor = EarlyStopping(patience=setting.patience,restore_best_weights=True)
    # shuffle false --> not shuffle the training data
    model.fit(x=X,y=y,epochs=setting.nb_epoch,shuffle= setting.shuffle, batch_size=setting.batch_size_parameter,validation_data=(X_val,y_val),callbacks=[early_stopping_monitor,history],verbose=verbose_para)
    # save the trained_model
    if save:
        file_path = "model.h5"
        save_model(model,file_path)

    return model,history

# model = load_model(filepath, compile = True)

def figure_layout(figsize=(10,8),titel="",xlabel="",ylabel="",fontsize_titel=18,fontsize_axis=16,fontsize_legend=14,fontsize_ticks=16,grid:bool = False, dpi = 100):

    plt.figure(figsize=figsize, dpi= dpi)
    ax1 = plt.gca()
    plt.rc('legend',fontsize=fontsize_legend)
    plt.title(titel, fontsize=fontsize_titel, fontweight = 'bold')
    plt.grid(grid)
    plt.xlabel(xlabel, fontsize=fontsize_axis)
    plt.ylabel(ylabel, fontsize=fontsize_axis)
    for tick in ax1.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize_ticks)
    #         tick.label1.set_fontweight('bold')
    for tick in ax1.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize_ticks)
    #     tick.label1.set_fontweight('bold')

    return ax1


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
