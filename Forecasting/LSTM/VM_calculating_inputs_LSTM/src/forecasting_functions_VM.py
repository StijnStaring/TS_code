import datetime as dt
import pandas as pd # pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings as wn
wn.filterwarnings(action='ignore')
from Test_basemodel_functions_VM import EnglandAndWalesHolidayCalendar
from keras.layers import Dense,  LSTM, Dropout, Flatten
from keras.models import Sequential, save_model
from keras import regularizers, optimizers
from keras.callbacks import EarlyStopping, History
from Test_basemodel_functions_VM import Switcher


class forecast_setting:

    def __init__(self, units_LSTM = 20, layers_LSTM = 1, units_DENSE = 20, layers_DENSE= 1, patience = 5,
                 shuffle = False, lag_value = 3, nb_epoch = 1, batch_size_para = 32,
                 repeat = 10, activation: str = 'relu', learning_rate: float = 0.001, dropout_LSTM = 0, recurrent_dropout_LSTM = 0, kernel_regularizer_LSTM = None,
                 recurrent_regularizer_LSTM = None, bais_regularizer_LSTM = None, activity_regularizer_LSTM = None, dropout_DENSE = 0,
                 kernel_regularizer_DENSE = None, bais_regularizer_DENSE = None, activity_regularizer_DENSE = None):

        self.units_LSTM = units_LSTM

        self.layers_LSTM = layers_LSTM

        self.dropout_LSTM = dropout_LSTM
        self.recurrent_dropout_LSTM = recurrent_dropout_LSTM
        if kernel_regularizer_LSTM is not None:
            self.kernel_regularizer_LSTM = regularizers.l2(l= kernel_regularizer_LSTM)
        else:
            self.kernel_regularizer_LSTM = None
        if recurrent_regularizer_LSTM is not None:
            self.recurrent_regularizer_LSTM = regularizers.l2(l= recurrent_regularizer_LSTM)
        else:
            self.recurrent_regularizer_LSTM = None
        if bais_regularizer_LSTM is not None:
            self.bais_regularizer_LSTM = regularizers.l2(l= bais_regularizer_LSTM)
        else:
            self.bais_regularizer_LSTM = None
        if activity_regularizer_LSTM is not None:
            self.activity_regularizer_LSTM = regularizers.l2(l= activity_regularizer_LSTM)
        else:
            self.activity_regularizer_LSTM = None

        self.units_DENSE = units_DENSE

        self.layers_DENSE = layers_DENSE

        self.dropout_DENSE = dropout_DENSE

        if kernel_regularizer_DENSE is not None:
            self.kernel_regularizer_DENSE = regularizers.l2(l= kernel_regularizer_DENSE)
        else:
            self.kernel_regularizer_DENSE = None
        if bais_regularizer_DENSE is not None:
            self.bais_regularizer_DENSE = regularizers.l2(l= bais_regularizer_DENSE)
        else:
            self.bais_regularizer_DENSE = None
        if activity_regularizer_DENSE is not None:
            self.activity_regularizer_DENSE = regularizers.l2(l= activity_regularizer_DENSE)
        else:
            self.activity_regularizer_DENSE = None

        self.lag_value = lag_value
        self.nb_epoch = nb_epoch
        self.activation = activation
        self.batch_size_parameter = batch_size_para
        self.learning_rate = learning_rate
        self.patience = patience
        self.shuffle = shuffle
        self.repeat = repeat

    def __str__(self):
        print(vars(self))




class time_serie:

    def __init__(self, ts: pd.Series, av_temperature: pd.DataFrame):
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


def get_all_days_of_year(serie: pd.Series)->set:
    collection = set()
    for date in serie.index:
        collection.add(date.dayofyear)
    return collection

def visualize_loss(history):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    figure_layout(figsize=(10,8),titel="Training and Validation Loss",xlabel="Epochs",ylabel="MSE",grid=False)
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.legend()
    # plt.show()

def generate_training_validation_division(data: np.ndarray, reference: np.array, batch_size: int = 32, ratio: float = 0.9):
    # policy is that the oldest data is thrown away.
    # data should be made compatible with the chosen batch size.
    if not len(data.shape) == 3:
        raise Exception("Input data is not of the correct shape: samples x timesteps x features.")
    if not len(data) == len(reference):
        raise Exception("The reference should have equal length as the observations.")

    amount_samples = data.shape[0]
    amount_training = np.around(amount_samples* ratio,0)
    amount_validation = amount_samples - amount_training
    rest_val = amount_validation % batch_size
    if rest_val <= batch_size/2 and (amount_validation - rest_val)//batch_size>=1: # remove validation values
        value = int(amount_validation - rest_val)
        validation = data[-1*value:,:,:]
        validation_ref = reference[-1*value:,:]
        less_training = len(data[0:int(amount_samples - value),:,:]) % batch_size
        training = data[int(less_training):int(amount_samples - value), :, :]
        training_ref = reference[int(less_training):int(amount_samples - value), :]
        return (training, training_ref), (validation, validation_ref)

    else: # add training values to the validation set
        value = int(amount_validation + batch_size - rest_val)
        validation = data[-1*value:,:,:]
        validation_ref = reference[-1*value:,:]
        less_training = len(data[0:int(amount_samples - value),:,:]) % batch_size
        training = data[int(less_training):int(amount_samples - value), :, :]
        training_ref = reference[int(less_training):int(amount_samples - value), :]
        return (training, training_ref), (validation, validation_ref)



def build_model_stateless1(setting: forecast_setting, X, y, verbose_para: int = 1, save: bool = False):
    """
    The model uses the output of the LSTM layer as input to a dense layer.
    This model doesn't expects a pre-made validation set.
    The validation split is set on 10%.
    """
    history = History()
    model = Sequential()

    # no initial state is given --> hidden state and cell state are tensors filled with zeros

    if setting.layers_LSTM == 1:
        model.add(LSTM(units=setting.units_LSTM, activation=setting.activation, kernel_regularizer= setting.kernel_regularizer_LSTM, recurrent_regularizer= setting.recurrent_regularizer_LSTM, bias_regularizer=setting.bais_regularizer_LSTM, dropout= setting.dropout_LSTM, recurrent_dropout= setting.recurrent_dropout_LSTM, activity_regularizer= setting.activity_regularizer_LSTM,batch_input_shape=(None, X.shape[1], X.shape[2])))  # no need to specify the batch size when stateless
    else:
        model.add(LSTM(units=setting.units_LSTM, activation=setting.activation, return_sequences = True, kernel_regularizer= setting.kernel_regularizer_LSTM, recurrent_regularizer= setting.recurrent_regularizer_LSTM, bias_regularizer=setting.bais_regularizer_LSTM, dropout= setting.dropout_LSTM, recurrent_dropout= setting.recurrent_dropout_LSTM, activity_regularizer= setting.activity_regularizer_LSTM, batch_input_shape=(None, X.shape[1], X.shape[2])))  # no need to specify the batch size when stateless
        for _ in np.arange(1,setting.layers_LSTM-1):
            model.add(LSTM(units=setting.units_LSTM, activation=setting.activation, return_sequences = True, kernel_regularizer= setting.kernel_regularizer_LSTM, recurrent_regularizer= setting.recurrent_regularizer_LSTM, bias_regularizer=setting.bais_regularizer_LSTM, dropout= setting.dropout_LSTM, recurrent_dropout= setting.recurrent_dropout_LSTM, activity_regularizer= setting.activity_regularizer_LSTM))
        model.add(LSTM(units=setting.units_LSTM, activation=setting.activation, kernel_regularizer= setting.kernel_regularizer_LSTM, recurrent_regularizer= setting.recurrent_regularizer_LSTM, bias_regularizer=setting.bais_regularizer_LSTM, dropout= setting.dropout_LSTM, recurrent_dropout= setting.recurrent_dropout_LSTM, activity_regularizer= setting.activity_regularizer_LSTM))

    for _ in range(setting.layers_DENSE):
        model.add(Dropout(setting.dropout_DENSE))
        model.add(Dense(units=setting.units_DENSE,activation='relu', kernel_regularizer=setting.kernel_regularizer_DENSE, bias_regularizer= setting.bais_regularizer_DENSE, activity_regularizer= setting.activity_regularizer_DENSE))
    model.add(Dropout(setting.dropout_DENSE))
    model.add(Dense(units=y.shape[1],activation='relu', kernel_regularizer=setting.kernel_regularizer_DENSE, bias_regularizer= setting.bais_regularizer_DENSE, activity_regularizer= setting.activity_regularizer_DENSE))
    model.compile(optimizer=optimizers.Adam(lr=setting.learning_rate, beta_1=0.9, beta_2=0.999),loss='mse')
    early_stopping_monitor = EarlyStopping(patience=setting.patience,restore_best_weights=True)
    try:
        model.fit(x=X,y=y,epochs=setting.nb_epoch,shuffle= setting.shuffle, batch_size=setting.batch_size_parameter,validation_split=0.10,callbacks=[early_stopping_monitor,history],verbose=verbose_para)
    except:
        print("Training of Model 1 went wrong --> try again")
        model, history = build_model_stateless1(setting, X, y, verbose_para = 1, save = False)
    else:
        print("Model 1 training_finished...")
    # save the trained_model
    if save:
        file_path = "model.h5"
        save_model(model,file_path)

    return model,history


def build_model_stateless2(setting: forecast_setting, X, y, X_val, y_val, verbose_para: int = 1, save: bool = False):
    """
    The model adds a flatten layer after the output of the last LSTM layer which outputs a matrix that is made of the different vectors at each timestep.
    """
    history = History()
    model = Sequential()

    model.add(LSTM(units=setting.units_LSTM, activation=setting.activation, return_sequences = True, kernel_regularizer= setting.kernel_regularizer_LSTM, recurrent_regularizer= setting.recurrent_regularizer_LSTM, bias_regularizer=setting.bais_regularizer_LSTM, dropout= setting.dropout_LSTM, recurrent_dropout= setting.recurrent_dropout_LSTM, activity_regularizer= setting.activity_regularizer_LSTM, batch_input_shape=(None, X.shape[1], X.shape[2])))  # no need to specify the batch size when stateless
    for _ in np.arange(1,setting.layers_LSTM):
        model.add(LSTM(units=setting.units_LSTM, activation=setting.activation, return_sequences = True, kernel_regularizer= setting.kernel_regularizer_LSTM, recurrent_regularizer= setting.recurrent_regularizer_LSTM, bias_regularizer=setting.bais_regularizer_LSTM, dropout= setting.dropout_LSTM, recurrent_dropout= setting.recurrent_dropout_LSTM, activity_regularizer= setting.activity_regularizer_LSTM))

    model.add(Flatten()) # Make from the matrix coming from the the LSTM a single vector

    for _ in range(setting.layers_DENSE):
        model.add(Dropout(setting.dropout_DENSE))
        model.add(
            Dense(units=setting.units_DENSE, activation='relu', kernel_regularizer=setting.kernel_regularizer_DENSE,
                  bias_regularizer=setting.bais_regularizer_DENSE,
                  activity_regularizer=setting.activity_regularizer_DENSE))
    model.add(Dropout(setting.dropout_DENSE))
    model.add(Dense(units=y.shape[1], activation='relu', kernel_regularizer=setting.kernel_regularizer_DENSE,
                    bias_regularizer=setting.bais_regularizer_DENSE,
                    activity_regularizer=setting.activity_regularizer_DENSE))
    model.compile(optimizer=optimizers.Adam(lr=setting.learning_rate, beta_1=0.9, beta_2=0.999), loss='mse')
    early_stopping_monitor = EarlyStopping(patience=setting.patience, restore_best_weights=True)
    model.fit(x=X, y=y, epochs=setting.nb_epoch, shuffle=setting.shuffle, batch_size=setting.batch_size_parameter,
              validation_data=(X_val, y_val), callbacks=[early_stopping_monitor, history], verbose=verbose_para)
    # save the trained_model

    if save:
        file_path = "model.h5"
        save_model(model, file_path)

    return model, history

def build_model_stateful1(setting: forecast_setting, X, y, X_val, y_val, verbose_para: int = 1, save: bool = False, reset_after_epoch: bool = True):

    history = History()
    model = None
    weights = None
    loss = []
    val_loss = []
    tracking = [np.inf]
    for i in range(2):
        if i == 0:
            batch_size = setting.batch_size_parameter
        else:
            batch_size = 1
        model = Sequential()

        if setting.layers_LSTM == 1:
            model.add(LSTM(units=setting.units_LSTM, activation=setting.activation, stateful= True, kernel_regularizer= setting.kernel_regularizer_LSTM, recurrent_regularizer= setting.recurrent_regularizer_LSTM, bias_regularizer=setting.bais_regularizer_LSTM, dropout= setting.dropout_LSTM, recurrent_dropout= setting.recurrent_dropout_LSTM, activity_regularizer= setting.activity_regularizer_LSTM, batch_input_shape=(batch_size, X.shape[1], X.shape[2])))  # no need to specify the batch size when stateless
        else:
            model.add(LSTM(units=setting.units_LSTM, activation=setting.activation, stateful= True, return_sequences=True, kernel_regularizer= setting.kernel_regularizer_LSTM, recurrent_regularizer= setting.recurrent_regularizer_LSTM, bias_regularizer=setting.bais_regularizer_LSTM, dropout= setting.dropout_LSTM, recurrent_dropout= setting.recurrent_dropout_LSTM, activity_regularizer= setting.activity_regularizer_LSTM, batch_input_shape=(batch_size, X.shape[1], X.shape[2])))
            for _ in np.arange(1, setting.layers_LSTM - 1):

                model.add(LSTM(units=setting.units_LSTM, activation=setting.activation, stateful= True, return_sequences=True, kernel_regularizer= setting.kernel_regularizer_LSTM, recurrent_regularizer= setting.recurrent_regularizer_LSTM, bias_regularizer=setting.bais_regularizer_LSTM, dropout= setting.dropout_LSTM, recurrent_dropout= setting.recurrent_dropout_LSTM, activity_regularizer= setting.activity_regularizer_LSTM))
            model.add(LSTM(units=setting.units_LSTM, activation=setting.activation, stateful= True, kernel_regularizer= setting.kernel_regularizer_LSTM, recurrent_regularizer= setting.recurrent_regularizer_LSTM, bias_regularizer=setting.bais_regularizer_LSTM, dropout= setting.dropout_LSTM, recurrent_dropout= setting.recurrent_dropout_LSTM, activity_regularizer= setting.activity_regularizer_LSTM))

        for _ in range(setting.layers_DENSE):
            model.add(Dropout(setting.dropout_DENSE))
            model.add(Dense(units=setting.units_DENSE, activation='relu', kernel_regularizer=setting.kernel_regularizer_DENSE, bias_regularizer= setting.bais_regularizer_DENSE, activity_regularizer= setting.activity_regularizer_DENSE))
        model.add(Dropout(setting.dropout_DENSE))
        model.add(Dense(units=y.shape[1], activation='relu', kernel_regularizer= setting.kernel_regularizer_DENSE, bias_regularizer= setting.bais_regularizer_DENSE, activity_regularizer= setting.activity_regularizer_DENSE))
        model.compile(optimizer=optimizers.Adam(lr=setting.learning_rate, beta_1=0.9, beta_2=0.999),loss='mse')
        early_stopping_monitor = EarlyStopping(patience=setting.patience,restore_best_weights=True)
        if i == 0:
            for k in range(setting.nb_epoch):
                model.fit(x=X,y=y,epochs=1,shuffle= setting.shuffle, batch_size=setting.batch_size_parameter,validation_data=(X_val,y_val),callbacks=[early_stopping_monitor,history],verbose=verbose_para)
                epoch_count = k+1
                print("Epoch number: %s/%s."%(epoch_count,setting.nb_epoch))
                loss.append(history.history['loss'][0])
                current_val_lost = history.history['val_loss'][0]
                val_loss.append(current_val_lost)

                if current_val_lost < tracking[0]:
                    weights = model.get_weights()
                    tracking = [current_val_lost]
                else:
                    tracking.append(current_val_lost)
                    if len(tracking) == setting.patience + 1:
                        break
                if reset_after_epoch:
                    model.reset_states()
        else:
            # Now the batch size is changed to one
            model.set_weights(weights)
    history.history['loss'] = loss
    history.history['val_loss'] = val_loss

    # save the trained_model
    if save:
        file_path = "model.h5"
        save_model(model,file_path)

    return model,history


# model = load_model(filepath, compile = True)

def daily_prediction(model: Sequential, TS_norm_full: pd.Series, temperature_norm: pd.Series, lag_value: int, daily_time_stamps: pd.DatetimeIndex):
    if len(daily_time_stamps) != 48:
            raise Exception("The test data doesn't equal 48 steps.")
    TS_copy = TS_norm_full.copy(deep= True)
    history_predictions = pd.Series(index= daily_time_stamps)
    history_reference = pd.Series(index= daily_time_stamps)
    for i in np.arange(0,len(daily_time_stamps)):
        time_stamp = daily_time_stamps[i]
        index_time_stamp = TS_copy.index.get_loc(time_stamp) # want to predict the time_stamp
        start = index_time_stamp - lag_value
        end = index_time_stamp
        history = TS_copy[start:end+1]
        prediction_input, reference = input_output_LSTM(history, temperature_norm, lag_value) # also the value to predict should be given
        y_hat = model.predict(prediction_input, batch_size= 1)
        # integrate the prediction in the TS_copy to be used in the next iterate
        TS_copy[time_stamp] = y_hat
        history_predictions[time_stamp] = y_hat
        history_reference[time_stamp] = reference

    if history_predictions.isnull().sum() + history_reference.isnull().sum() != 0:
        raise Exception("Not all the values of the day are correct predicted!")
    return history_predictions, history_reference

def test_set_prediction(model: Sequential, setting: forecast_setting, serie: time_serie, test_set:pd.Series, X_train, X_val, real_values: bool = True, seeding: bool = False):
    # assumption that the test set has no gaps in the dates is not valid
    model.reset_states()
    if seeding: # seeding only usefull when have a stateful model
        total_training = np.vstack((X_train,X_val))
        for i in np.arange(0,len(total_training)):
            row = total_training[i:i+1,:,:]
            model.predict(row,batch_size=1)

    collection = list(get_all_days_of_year(test_set))
    day_int = collection[0]
    daily_time_stamps = test_set[test_set.index.dayofyear == day_int].index
    (history_predictions, history_reference) = daily_prediction(model, serie.TS_norm_full, serie.temperature_norm, setting.lag_value, daily_time_stamps)
    all_predictions = history_predictions
    all_references = history_reference

    if len(collection) > 1:
        for day_int in collection[1:]:
            daily_time_stamps = test_set[test_set.index.dayofyear == day_int].index
            (history_predictions, history_reference) = daily_prediction(model, serie.TS_norm_full, serie.temperature_norm, setting.lag_value, daily_time_stamps)
            all_predictions = all_predictions.append(history_predictions)
            all_references = all_references.append(history_reference)

    if real_values:
        all_predictions = norm_inverse(all_predictions,serie.scaler_history)
        all_references = norm_inverse(all_references,serie.scaler_history)

    return all_predictions, all_references


def get_performance(all_predictions: pd.Series, all_references: pd.Series, metric: str = 'NRMSE'):
    return Switcher(metric, all_predictions, all_references)

plt.rc('axes', linewidth=2)
def figure_layout(figsize=(10,8),titel="",xlabel="",ylabel="",fontsize_titel=22,fontsize_axis=22,fontsize_legend=22,fontsize_ticks=20,grid:bool = False, dpi = 100):

    plt.figure(figsize=figsize, dpi= dpi)
    ax1 = plt.gca()
    plt.rc('legend',fontsize=fontsize_legend)
    plt.title(titel, fontsize=fontsize_titel, fontweight = 'bold')
    plt.grid(grid)
    plt.xlabel(xlabel, fontsize=fontsize_axis)
    plt.ylabel(ylabel, fontsize=fontsize_axis)
    for tick in ax1.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize_ticks)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize_ticks)

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
    # print("amount of missing values: %s. \n"%TS.isnull().sum())
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
    # print("amount of missing values: %s. \n" % TS.isnull().sum())
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

def show_forecast(all_predictions, all_references, ID: str, day_int: str, save: bool):
    axis = figure_layout(figsize=(10,8),titel="",xlabel="date",ylabel="kWh", dpi= 300)
    labels = ["True Future", "Model Prediction"]

    all_references.plot(ax=axis, lw= 3.0, grid= True)
    all_predictions.plot(ax=axis, lw= 3.0, grid= True)
    axis.legend(labels)

    if save:
        path = "Vanilla_LSTM_figures/"
        fname = path + "ID" + ID + "_Day" + day_int + ".png"
        plt.savefig(fname, dpi=300, facecolor='w', edgecolor='w', orientation='portrait', format=None,
                    transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)


def show_all_forecasts(all_predictions, all_references,ID: str, save: bool = True):
    collection = get_all_days_of_year(all_predictions)
    for day_int in collection:
        predictions = all_predictions[all_predictions.index.dayofyear == day_int]
        references = all_references[all_references.index.dayofyear == day_int]
        show_forecast(predictions, references, ID, str(day_int), save)
    plt.show()

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def create_validation_stateless(kwargs,ratio = 0.9):
    new_dict = {}
    for (key,value) in kwargs.items():
        X = value[0]
        y = value[1]
        split_value = int(ratio*len(value[0]))
        X_train = X[0:split_value,:,:]
        y_train = y[0:split_value,:]
        X_val = X[split_value:, :, :]
        y_val = y[split_value:, :]
        new_dict[key] = (X_train,y_train,X_val,y_val)
    return new_dict

