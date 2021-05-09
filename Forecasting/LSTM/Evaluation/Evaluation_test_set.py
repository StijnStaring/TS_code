import sys
sys.path.append("D:\AI_time_series_repos\TS_code\Forecasting\LSTM")
sys.path.append("D:\AI_time_series_repos\TS_code\Forecasting\\basemodel")
sys.path.append("D:\AI_time_series_repos\TS_code\Forecasting\\basemodel\local_pc\\Test_basemodel_functions.py")
from forecasting_functions import *
from time import time
from os import makedirs, path, listdir
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from keras.models import load_model
from Test_basemodel_functions import Switcher, autolabel


# paths_model1 = ["D:\AI_time_series_repos\TS_code\Forecasting\LSTM\Evaluation\Selected_models\Model1\serie1",
# "D:\AI_time_series_repos\TS_code\Forecasting\LSTM\Evaluation\Selected_models\Model1\serie2",
# "D:\AI_time_series_repos\TS_code\Forecasting\LSTM\Evaluation\Selected_models\Model1\serie3"
# ]

paths_model1 = ["D:\AI_time_series_repos\TS_code\Forecasting\LSTM\Evaluation\Selected_models\Model1\serie1"
]

# paths_model2 = ["D:\AI_time_series_repos\TS_code\Forecasting\LSTM\Evaluation\Selected_models\Model2\serie1",
# "D:\AI_time_series_repos\TS_code\Forecasting\LSTM\Evaluation\Selected_models\Model2\serie2",
# "D:\AI_time_series_repos\TS_code\Forecasting\LSTM\Evaluation\Selected_models\Model2\serie3"
# ]

path_history = "D:\AI_time_series_repos\TS_code\Forecasting\\basemodel\data\DF_three_series.csv"
path_temperature = "D:\AI_time_series_repos\TS_code\Forecasting\\basemodel\data\DF_three_temp_series.csv"
path_to_array = "D:\AI_time_series_repos\TS_code\Forecasting\LSTM\VM_calculating_inputs_LSTM\output_arrays"

output_path = "D:\AI_time_series_repos\TS_code\Forecasting\LSTM\Evaluation"

name_folder = str(time())
makedirs(path.join(output_path,name_folder), exist_ok=True)
output_path = path.join(output_path,name_folder)

# considered_paths = [paths_model1, paths_model2]
considered_paths = [paths_model1]
fullYeardata = pd.read_csv(path_history, index_col="date", parse_dates=True)
names = fullYeardata.columns
av_temperature = pd.read_csv(path_temperature, index_col="meter_id", parse_dates=True)


print("Warning: Paths are expected to be in order --> serie1, serie2, serie3.")
DF = pd.DataFrame()
MAE_outputs_s1 = dict() # for the bar plot
MAE_outputs_s2 = dict()
MAE_outputs_s3 = dict()

MAPE_outputs_s1 = dict() # for the bar plot
MAPE_outputs_s2 = dict()
MAPE_outputs_s3 = dict()

MAE_plot_s1 = dict()
MAE_plot_s2 = dict()
MAE_plot_s3 = dict()

MAPE_plot_s1 = dict()
MAPE_plot_s2 = dict()
MAPE_plot_s3 = dict()

def assign_to_dict(index: int, model: str):
    # makes use of the global variables --> dicts
    if index == 1:
        indices = all_references[all_references.isnull()].index
        MAE_outputs_s1[model] = Switcher("MAE", all_predictions.drop(axis=0, labels=indices), all_references.drop(axis=0, labels=indices))
        MAPE_outputs_s1[model] = Switcher("MAPE", all_predictions.drop(axis=0, labels=indices), all_references.drop(axis=0, labels=indices))
        MAE_plot_s1[model] = calculate_daily_mean("MAE", all_predictions, all_references)
        MAPE_plot_s1[model] = calculate_daily_mean("MAPE", all_predictions, all_references)

    elif index == 2:
        indices = all_references[all_references.isnull()].index
        MAE_outputs_s2[model] = Switcher("MAE", all_predictions.drop(axis=0, labels=indices), all_references.drop(axis=0, labels=indices))
        MAPE_outputs_s2[model] = Switcher("MAPE", all_predictions.drop(axis=0, labels=indices), all_references.drop(axis=0, labels=indices))
        MAE_plot_s2[model] = calculate_daily_mean("MAE", all_predictions, all_references)
        MAPE_plot_s2[model] = calculate_daily_mean("MAPE", all_predictions, all_references)

    elif index == 3:
        indices = all_references[all_references.isnull()].index
        MAE_outputs_s3[model] = Switcher("MAE", all_predictions.drop(axis=0, labels=indices), all_references.drop(axis=0, labels=indices))
        MAPE_outputs_s3[model] = Switcher("MAPE", all_predictions.drop(axis=0, labels=indices), all_references.drop(axis=0, labels=indices))
        MAE_plot_s3[model] = calculate_daily_mean("MAE", all_predictions, all_references)
        MAPE_plot_s3[model] = calculate_daily_mean("MAPE", all_predictions, all_references)

def create_barplot_from_dict(outputs:dict, titel: str, path: str,ylabel = "Error [kWh]"):
    titels = list(outputs.keys())
    values = list(outputs.values())
    axis = figure_layout(figsize=(18, 12), titel=titel, grid=False, ylabel= ylabel)
    rects = axis.bar(titels, values, color='maroon', width=0.4)
    autolabel(rects, axis)
    if path != "":
        plt.savefig(path, dpi=300, facecolor='w', edgecolor='w', orientation='portrait', format=None,
                    transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)

def create_lineplot_from_dict_with_series(outputs:dict, titel: str, path: str):
    axis = figure_layout(figsize=(10, 8), titel=titel, xlabel="date", ylabel="kWh", dpi=300)
    labels = []
    counter = 0
    markers = ["o","^", "*", "h", "D","x","s",","]
    for item in outputs.items():
        (name, serie) = item
        labels.append(name)
        serie.plot(ax=axis, lw=3.0, grid=True, fontsize=26, marker=markers[counter])
        axis.legend(labels, loc="upper right")
        axis.grid(b=True, axis= "both")
        counter += 1

    if path != "":
        plt.savefig(path, dpi=300, facecolor='w', edgecolor='w', orientation='portrait', format=None,
                    transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)

basemodel_serie1_path = "D:\AI_time_series_repos\TS_code\Forecasting\\basemodel\outputs\ID0x78a812ecd87a4b945e0d262aec41e0eb2b59fe1e.csv"
basemodel_serie1 = pd.read_csv(basemodel_serie1_path, index_col = 0, parse_dates= True)
all_references = basemodel_serie1["real_values"]
all_predictions = basemodel_serie1["mean_forecast"]
assign_to_dict(index = 1, model="BL_mean")
all_predictions = basemodel_serie1["MAPE_forecast"]
assign_to_dict(index = 1, model="BL_MAPE")

basemodel_serie2_path = "D:\AI_time_series_repos\TS_code\Forecasting\\basemodel\outputs\ID0x1e84e4d5cf1f463147f3e4d566167597423d7769.csv"
basemodel_serie2 = pd.read_csv(basemodel_serie2_path, index_col = 0, parse_dates= True)
all_references = basemodel_serie2["real_values"]
all_predictions = basemodel_serie2["mean_forecast"]
assign_to_dict(index = 2, model="BL_mean")
all_predictions = basemodel_serie2["MAPE_forecast"]
assign_to_dict(index = 2, model="BL_MAPE")

basemodel_serie3_path = "D:\AI_time_series_repos\TS_code\Forecasting\\basemodel\outputs\ID0xc3b2f61a72e188cfd44483fce1bc11d6a628766d.csv"
basemodel_serie3 = pd.read_csv(basemodel_serie3_path, index_col = 0, parse_dates= True)
all_references = basemodel_serie3["real_values"]
all_predictions = basemodel_serie3["mean_forecast"]
assign_to_dict(index = 3, model="BL_mean")
all_predictions = basemodel_serie3["MAPE_forecast"]
assign_to_dict(index = 3, model="BL_MAPE")


for paths in considered_paths:
    for index in np.arange(1,len(paths)+1):
        # assert len(paths) == 3
        name = names[index-1]
        ts = time_serie(fullYeardata[name], av_temperature)
        # indices = ts.test_true[ts.test_true.isnull()].index
        # if len(indices) != 0:
        #     ts.test_true = ts.test_true.drop(axis=0,labels=indices,inplace=True)

        days_in_testset = get_all_days_of_year(ts.test_true)
        path_to_folder = paths[index-1]

        for filename in listdir(path_to_folder):
            if filename.endswith(".h5"):
                print(path.join(path_to_folder, filename))
                trained_model = load_model(path.join(path_to_folder, filename))

            elif filename[0:7] ==  "logbook":
                print(path.join(path_to_folder, filename))
                Logbook = pd.read_csv(path.join(path_to_folder, filename), index_col=0)

        lag_value = int(Logbook.loc["lag_value",str(1)])

        if "Model3" in path_to_folder:
            path_X_train_full = path.join(path_to_array, "X_" + name + "_" + str(lag_value) + "_full.npy")
            X_train_full = np.load(path_X_train_full)

            all_references = ts.test_true #this all references contains nan values
            all_predictions, _ = test_set_prediction(trained_model, lag_value, ts, ts.test_true, X_train_full,True, True)

            DF["M3_r_S" + str(index)] = all_references
            DF["M3_p_S" + str(index)] = all_predictions
            assign_to_dict(index,"M3")

        else:
            all_references = ts.test_true  # this all references contains nan values
            all_predictions, _ = test_set_prediction(trained_model, lag_value, ts, ts.test_true, None, True, False)

            if "Model1" in path_to_folder:
                DF["M1_r_S" + str(index)] = all_references # here references doesn't has any nan values anymore because in daily prediction uses substituted reference signal
                DF["M1_p_S" + str(index)] = all_predictions
                assign_to_dict(index, "M1")

            elif "Model2" in path_to_folder:
                DF["M2_r_S" + str(index)] = all_references
                DF["M2_p_S" + str(index)] = all_predictions
                assign_to_dict(index, "M2")

# Create all the plots
create_barplot_from_dict(MAE_outputs_s1, "MAE serie 1", path.join(output_path,"MAE_1.png"))
create_barplot_from_dict(MAE_outputs_s2, "MAE serie 2", path.join(output_path,"MAE_2.png"))
create_barplot_from_dict(MAE_outputs_s3, "MAE serie 3", path.join(output_path,"MAE_3.png"))

create_barplot_from_dict(MAPE_outputs_s1, "MAPE serie 1", path.join(output_path,"MAPE_1.png"))
create_barplot_from_dict(MAPE_outputs_s2, "MAPE serie 2",path.join(output_path,"MAPE_2.png"))
create_barplot_from_dict(MAPE_outputs_s3, "MAPE serie 3",path.join(output_path,"MAPE_3.png"))

create_lineplot_from_dict_with_series(MAE_plot_s1, "MAE serie 1", path.join(output_path,"MAE_1_line.png"))
create_lineplot_from_dict_with_series(MAE_plot_s2, "MAE serie 2", path.join(output_path,"MAE_2_line.png"))
create_lineplot_from_dict_with_series(MAE_plot_s3, "MAE serie 3", path.join(output_path,"MAE_3_line.png"))

create_lineplot_from_dict_with_series(MAPE_plot_s1, "MAPE serie 1", path.join(output_path,"MAPE_1_line.png"))
create_lineplot_from_dict_with_series(MAPE_plot_s2, "MAPE serie 2", path.join(output_path,"MAPE_2_line.png"))
create_lineplot_from_dict_with_series(MAPE_plot_s3, "MAPE serie 3", path.join(output_path,"MAPE_3_line.png"))

print(100*"#")
print("All the series are evaluated!")