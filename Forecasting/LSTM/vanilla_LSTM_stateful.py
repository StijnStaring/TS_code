
from forecasting_functions import *
from os import makedirs, path

which_model = "model1_sl"
Stijn = True

if Stijn:
    path_history = "D:\AI_time_series_repos\TS_code\Forecasting\\basemodel\data\DF_three_series.csv"
    path_temperature = "D:\AI_time_series_repos\TS_code\Forecasting\\basemodel\data\DF_three_temp_series.csv"
    path_to_array = "D:\AI_time_series_repos\TS_code\Forecasting\LSTM\VM_calculating_inputs_LSTM\output_arrays_stateful"

else:
    path_history = ""
    path_temperature = ""
    path_to_array = ""

fullYeardata = pd.read_csv(path_history,index_col= "date",parse_dates= True)
names = fullYeardata.columns
av_temperature = pd.read_csv(path_temperature,index_col="meter_id",parse_dates=True)


if which_model == "model1_sl":
    makedirs("./Stateful_investigation/outputs", exist_ok=True)
    output_path = "./Stateful_investigation/outputs"
    path_txt_file = './Stateful_investigation/outputs/output_file.txt'
else:
    output_path = ""
    path_txt_file = ""

print("CSV files are loaded...")



names = fullYeardata.columns
name = names[0]
ts = time_serie(fullYeardata[name], av_temperature)
setting1 = forecast_setting(lag_value= 1, shuffle= False, nb_epoch= 2, batch_size_para= 1)
# load the LSTM input matrices
path_X_train = path.join(path_to_array, "X_" + name + "_" + str(setting1.lag_value) + ".npy")
X_train = np.load(path_X_train)
path_y_train = path.join(path_to_array, "y_" + name + "_" + str(setting1.lag_value) + ".npy")
y_train = np.load(path_y_train)
# take the last 10 days of November as validation for the parameter search
X_train = X_train[:-480]
y_train = y_train[:-480]
ts.test_true = ts.training_true[-480:]
# X_train,y_train = unison_shuffled_copies(X_train,y_train) # no val anymore
print("inputs found...\r\n")



# should set the shuffling off

trained_model, history = build_model_stateful1(setting1, X_train, y_train)
