from forecasting_functions import *
from multiprocessing import Pool, cpu_count
import time
from functools import reduce
import os

# parameters that are tried
# first stage --> regularization off
class ParameterSearch:
    def __init__(self):
        self.list_units_LSTM = [20,50,100]
        self.list_layers_LSTM = [1,2,3]
        self.list_dropout_LSTM = [0]
        self.list_recurrent_dropout_LSTM = [0]
        self.list_kernel_regularization_LSTM = [None]
        self.list_recurrent_regularization_LSTM = [None]
        self.list_bais_regularization_LSTM = [None]
        self.list_activity_regularization_LSTM = [None]

        self.list_units_DENSE = [75]
        self.list_layers_DENSE = [1]
        self.list_dropout_DENSE = [0]
        self.list_kernel_regularization_DENSE = [None]
        self.list_bais_regularization_DENSE = [None]
        self.list_activity_regularization_DENSE = [None]

        self.list_lag_value = [48,96,336]
        self.list_nb_epoch = [1000]
        self.list_activation = ['relu']
        self.list_batch_size_parameter = [1,32,64]
        self.list_learning_rate = [10**-3,5*10**-3,10**-2,10**-1]
        self.list_patience = [5]
        self.list_shuffle = ['False']
        self.list_repeat = [3]


def run_once_input_LSTM(kwargs):
    print("Running LSTM input...")
    # Now are still trying with toy trainingset
    return input_output_LSTM(kwargs["ts"].training, kwargs["ts"].temperature_norm, kwargs["lag_value"])

if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    os.makedirs("./outputs", exist_ok=True)
    output_folder_path = os.path.join(dirname, 'outputs')

    Stijn = True
    path_history:str
    path_temperature:str

    if Stijn:
        path_history = "D:\AI_time_series_repos\TS_code\Forecasting\\basemodel\data\DF_three_series.csv"
        path_temperature = "D:\AI_time_series_repos\TS_code\Forecasting\\basemodel\data\DF_three_temp_series.csv"

    else:
        path_history = ""
        path_temperature = ""

    fullYeardata = pd.read_csv(path_history,index_col= "date",parse_dates= True)
    names = fullYeardata.columns
    av_temperature = pd.read_csv(path_temperature,index_col="meter_id",parse_dates=True)

    print("Running this file on a PC with %s cores..." % (cpu_count()))
    statistics = pd.DataFrame()
    LogBookIndex = list(vars(forecast_setting()).keys())
    LogBook = pd.DataFrame(index=LogBookIndex)

    results_file_name = os.path.join(output_folder_path,str(time.time()) + ".txt")
    results_file = open(results_file_name, "w")
    results_file.write("Running the similation on a PC with %s cores...\r\n" % (cpu_count()))
    results_file.close()
    chosen_parameters = ParameterSearch()
    amount_of_possibilities: int = reduce((lambda x, y: x * y), [len(x) for x in list(vars(chosen_parameters).values())])
    current_possibility_index: int = 1
    print("Found %s sets of parameters." % amount_of_possibilities)


    for name in names:
        results_file = open(results_file_name, "a")
        results_file.write("#" * 50 + "\r\n")
        results_file.write("The calculation of the LSTM inputs is started for serie: %s.\r\n"%name)
        results_file.write("-"*50 + "\r\n")
        ts = time_serie(fullYeardata[name], av_temperature)
        p = Pool(processes=cpu_count())
        results = p.map(run_once_input_LSTM, [{"ts":ts, "lag_value": lag_value} for lag_value in chosen_parameters.list_lag_value])
        X_shapes = [tup[0].shape for tup in results]
        y_shapes = [tup[1].shape for tup in results]
        results_file = open(results_file_name, "a")
        results_file.write("X shapes: %s.\r\n"%X_shapes)
        results_file.write("y shapes: %s.\r\n"%y_shapes)
        results_file.close()

        print("Completed Serie: %s..." % name)
    print("Finished simulation - data written to txt file.")