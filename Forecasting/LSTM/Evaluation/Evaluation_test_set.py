import sys
sys.path.append("D:\AI_time_series_repos\TS_code\Forecasting\LSTM")
sys.path.append("D:\AI_time_series_repos\TS_code\Forecasting\\basemodel")
from forecasting_functions import *
from time import time
from os import makedirs, path, listdir
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from keras.models import load_model


paths_model1 = ["D:\AI_time_series_repos\TS_code\Forecasting\LSTM\Evaluation\Selected_models\Model1\serie1",
"D:\AI_time_series_repos\TS_code\Forecasting\LSTM\Evaluation\Selected_models\Model1\serie2",
"D:\AI_time_series_repos\TS_code\Forecasting\LSTM\Evaluation\Selected_models\Model1\serie3"
]

paths_model2 = ["D:\AI_time_series_repos\TS_code\Forecasting\LSTM\Evaluation\Selected_models\Model2\serie1",
"D:\AI_time_series_repos\TS_code\Forecasting\LSTM\Evaluation\Selected_models\Model2\serie2",
"D:\AI_time_series_repos\TS_code\Forecasting\LSTM\Evaluation\Selected_models\Model2\serie3"
]

path_history = "D:\AI_time_series_repos\TS_code\Forecasting\\basemodel\data\DF_three_series.csv"
path_temperature = "D:\AI_time_series_repos\TS_code\Forecasting\\basemodel\data\DF_three_temp_series.csv"
path_to_array = "D:\AI_time_series_repos\TS_code\Forecasting\LSTM\VM_calculating_inputs_LSTM\output_arrays"

output_path = "D:\AI_time_series_repos\TS_code\Forecasting\LSTM\Evaluation"

name_folder = str(time())
makedirs(path.join(output_path,name_folder), exist_ok=True)
output_path = name_folder

considered_paths = [paths_model1, paths_model2]
fullYeardata = pd.read_csv(path_history, index_col="date", parse_dates=True)
names = fullYeardata.columns
av_temperature = pd.read_csv(path_temperature, index_col="meter_id", parse_dates=True)

print("Paths are expected to be in order.")
DF = pd.DataFrame()
for paths in considered_paths:
    for index in range(len(paths)):
        assert len(paths) == 3
        name = names[index]
        ts = time_serie(fullYeardata[name], av_temperature)
        path_to_folder = paths[index]
        trained_model = None
        Logbook = None
        for filename in listdir(path_to_folder):
            if filename.endswith(".h5"):
                print(path.join(path_to_folder, filename))
                trained_model = load_model(path.join(path_to_folder, filename))

            elif filename[0:7] ==  "logbook":
                Logbook = pd.read_csv(path.join(path_to_folder, filename), index_col=0)

        lag_value = Logbook.loc[["lag_value"],[0]]

        if "Model3" in path_to_folder:
            path_X_train_full = path.join(path_to_array, "X_" + name + "_" + str(lag_value) + "_full.npy")
            X_train_full = np.load(path_X_train_full)
            all_predictions, all_references = test_set_prediction(trained_model, lag_value, ts, ts.test_true, X_train_full,True, True)
            DF["M3_r_S" + str(index)] = all_references
            DF["M3_p_S" + str(index)] = all_predictions
        else:
            all_predictions, all_references = test_set_prediction(trained_model, lag_value, ts, ts.test_true, None, True, False)

            if "Model1" in path_to_folder:
                DF["M1_r_S" + str(index)] = all_references
                DF["M1_p_S" + str(index)] = all_predictions

            elif "Model2" in path_to_folder:
                DF["M2_r_S" + str(index)] = all_references
                DF["M2_p_S" + str(index)] = all_predictions





    # Evaluate the different methods:
    vertical_stack.dropna(inplace=True,axis=0)
    real_values = vertical_stack['real_values']
    for method in ["MSE","RMSE","NRMSE","MAE"]:
        outputs = dict()
        for col_name2 in vertical_stack.columns[:-1]: # expects the real values at the end of the DataFrame
            forecast = vertical_stack[col_name2]
            output:float = Switcher(method,forecast,real_values)
            outputs[col_name2] = output

        titels = list(outputs.keys())
        values = list(outputs.values())
        # plt.figure(method)
        axis = figure_layout(figsize=(18, 12), titel=method, grid=False)
        # axis.grid(axis='x')
        rects = axis.bar(titels, values, color='maroon', width=0.4)
        autolabel(rects, axis)

        fname = save_to + "Serie"+str(col_name) + "_" + method + "_basemodel.png"
        plt.savefig(fname, dpi=300, facecolor='w', edgecolor='w', orientation='portrait', format=None,
                    transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)

print(100*"#")
print("All the series are trained and evaluated!")