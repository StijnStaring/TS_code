
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


from forecasting_functions import *

# importing the data
Stijn = True
fullYearpath = None
av_temperaturepath = None
if Stijn:
    fullYearpath = "D:\Onedrive\Leuven\Final project\data\Forecasting_writtendata\FullYear.csv"
    av_temperaturepath = "D:\Onedrive\Leuven\Final project\data\weather-avg.csv"
else:
    raise Exception("Put some paths here.")

fullYeardata = pd.read_csv(fullYearpath,index_col= "date",parse_dates= True)
av_temperature = pd.read_csv(av_temperaturepath,index_col='meter_id')
av_temperature = av_temperature.transpose()
av_temperature.index = pd.to_datetime(av_temperature.index)


time_serie1 = time_serie(fullYeardata[fullYeardata.columns[1]], av_temperature)
setting1 = forecast_setting(units_LSTM = 5, layers_LSTM = 2, units_DENSE = 5, layers_DENSE= 1, patience = 3, shuffle = False, lag_value = 3, nb_epoch = 100, batch_size_para = 16, repeat = 10, activation = 'relu', learning_rate = 0.001, dropout_LSTM = 0.0, recurrent_dropout_LSTM = 0.0, kernel_regularizer_LSTM = None, recurrent_regularizer_LSTM = None, bais_regularizer_LSTM = None, activity_regularizer_LSTM = None, dropout_DENSE = 0, kernel_regularizer_DENSE = None, bais_regularizer_DENSE = None, activity_regularizer_DENSE = None)
# setting1 = forecast_setting(units_LSTM = 10, layers_LSTM=1, units_DENSE = 5, layers_DENSE= 1, patience = 5, shuffle = False, lag_value = 3, nb_epoch = 150, batch_size_para = 32, repeat = 10, activation = 'tanh', kernel_regularizer_DENSE= 0.01)



X,y = input_output_LSTM(time_serie1.training, time_serie1.temperature_norm, setting1.lag_value)
print("Shape X original: %s."%(X.shape,))
((X,y),(X_val,y_val)) = generate_training_validation_division(X,y,1,0.91018)

print(X.shape)
print(y.shape)
print(X_val.shape)
print(y_val.shape)

axis1 = figure_layout(titel="Division of serie", xlabel="sample", ylabel="kWh")
axis1.plot(np.arange(1,len(y)+1),y.squeeze())
axis1.plot(np.arange(len(y)+1, len(y)+len(y_val)+1),y_val.squeeze())
axis1.plot(np.arange(len(y)+len(y_val)+1,len(y)+len(y_val)+1+len(time_serie1.test)), time_serie1.test)

# path = "Vanilla_LSTM_figures/"
# fname = path+"division_data" + ".png"
# plt.savefig(fname, dpi=300, facecolor='w', edgecolor='w', orientation='portrait', format=None,
#             transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)


trained_model, history = build_model_stateless2(setting1, X,y, X_val, y_val)
# trained_model, history = build_model_stateless1(setting1, X,y, X_val, y_val, reset_after_epoch= True)



trained_model.summary()



all_predictions, all_references = test_set_prediction(trained_model, setting1, time_serie1, time_serie1.test, X, X_val, True, False)

