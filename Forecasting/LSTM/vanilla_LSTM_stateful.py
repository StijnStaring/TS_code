from keras.models import load_model
from forecasting_functions import *
from os import makedirs, path

which_model = "model3_sf"
Stijn = True

if Stijn:
    path_history = "D:\AI_time_series_repos\TS_code\Forecasting\\basemodel\data\DF_three_series.csv"
    path_temperature = "D:\AI_time_series_repos\TS_code\Forecasting\\basemodel\data\DF_three_temp_series.csv"
    path_to_array = "D:\AI_time_series_repos\TS_code\Forecasting\LSTM\VM_calculating_inputs_LSTM\output_arrays"

else:
    path_history = ""
    path_temperature = ""
    path_to_array = ""

fullYeardata = pd.read_csv(path_history,index_col= "date",parse_dates= True)
names = fullYeardata.columns
av_temperature = pd.read_csv(path_temperature,index_col="meter_id",parse_dates=True)


if which_model == "model3_sf":
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
setting1 = forecast_setting(lag_value= 1, shuffle= False, nb_epoch= 1, batch_size_para= 1)
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

trained_model, history = build_model_stateful1(setting1, X_train, y_train, save= True)

model = load_model("model.h5")

if which_model == "model3_sf":
    path_X_train_full = path.join(path_to_array, "X_" + name + "_" + str(setting1.lag_value) + "_full.npy")
    X_train_full = np.load(path_X_train_full)
all_predictions, all_references = test_set_prediction(model, setting1, ts, ts.test_true, X_train_full, True, True)

print("Model 3 prediction finished...")
outputs_model = dict()
for method in ["MSE", "RMSE", "NRMSE", "MAE", "MAPE"]:
    output: float = Switcher(method, all_predictions, all_references)
    outputs_model[method] = output


#
# from keras.models import load_model
# from forecasting_functions import *
# from os import makedirs, path
#
# which_model = "model4_sf"
# Stijn = True
#
# if Stijn:
#     path_history = "D:\AI_time_series_repos\TS_code\Forecasting\\basemodel\data\DF_three_series.csv"
#     path_temperature = "D:\AI_time_series_repos\TS_code\Forecasting\\basemodel\data\DF_three_temp_series.csv"
#     path_to_array = "D:\AI_time_series_repos\TS_code\Forecasting\LSTM\VM_calculating_inputs_LSTM\output_arrays_sf48"
#
# else:
#     path_history = ""
#     path_temperature = ""
#     path_to_array = ""
#
# fullYeardata = pd.read_csv(path_history,index_col= "date",parse_dates= True)
# names = fullYeardata.columns
# av_temperature = pd.read_csv(path_temperature,index_col="meter_id",parse_dates=True)
#
#
# if which_model == "model4_sf":
#     makedirs("./Stateful_investigation/outputs", exist_ok=True)
#     output_path = "./Stateful_investigation/outputs"
#     path_txt_file = './Stateful_investigation/outputs/output_file.txt'
# else:
#     output_path = ""
#     path_txt_file = ""
#
# print("CSV files are loaded...")
#
# names = fullYeardata.columns
# name = names[0]
# ts = time_serie(fullYeardata[name], av_temperature)
# ts.test_true = ts.training_true[-480:]
# setting1 = forecast_setting(lag_value= 48, shuffle= False, nb_epoch= 1, batch_size_para= 1)
# # load the LSTM input matrices
# collection_trained_models = []
# collection_histories = []
#
# # for model_number in range(48):
# #     print("model %s"%model_number)
# #     path_X_train = path.join(path_to_array, "X_" + name + "_" + str(setting1.lag_value) + "_" + str(model_number) + "_.npy")
# #     X_train = np.load(path_X_train)
# #     path_y_train = path.join(path_to_array, "y_" + name + "_" + str(setting1.lag_value) + "_" + str(model_number) + "_.npy")
# #     y_train = np.load(path_y_train)
# #     # take the last 10 days of November as validation for the parameter search
# #     X_train = X_train[:-10]
# #     y_train = y_train[:-10]
# #
# #     print("inputs found...\r\n")
# #
# #     # trained_model, history = build_model_stateful1(setting1, X_train, y_train, save= True)
# #     # save_model(trained_model, path.join(output_path,"model"+str(model_number)+".h5"))
# #     trained_model = load_model(path.join(output_path,"model"+str(model_number)+".h5"))
# #     collection_trained_models.append(trained_model)
# #     # collection_histories.append(history)
# model = load_model(path.join(output_path,"model"+str(0)+".h5"))
# for i in range(48):
#     collection_trained_models.append(model)
# all_predictions, all_references = test_set_prediction_sf(path_to_array, collection_trained_models, setting1, ts, ts.test_true, True)
# # model = load_model("model.h5")
# # if which_model == "model3_sf":
# #     path_X_train_full = path.join(path_to_array, "X_" + name + "_" + str(setting1.lag_value) + "_full.npy")
# #     X_train_full = np.load(path_X_train_full)
# # all_predictions, all_references = test_set_prediction(model, setting1, ts, ts.test_true, X_train_full, True, True)
#
