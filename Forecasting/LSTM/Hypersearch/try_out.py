from forecasting_functions import *
from multiprocessing import Pool, cpu_count
from time import time
from functools import reduce
from os.path import join, dirname
from os import makedirs
from sys import exit
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False
from keras.backend import clear_session, reset_uids


# parameters that are tried
# first stage --> regularization off
class ParameterSearch:
    def __init__(self):
        self.list_units_LSTM = [20]
        self.list_layers_LSTM = [1]
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
        # this has to be changed!
        # self.list_lag_value = [48, 96, 336]  # four is chosen becaus have four cores
        self.list_lag_value = [48]
        self.list_nb_epoch = [1000]
        self.list_activation = ['relu']
        self.list_batch_size_parameter = [1]
        self.list_learning_rate = [10 ** -2]
        self.list_patience = [5]
        self.list_shuffle = ['True']
        self.list_repeat = [8]  # four is chosen because have four cores


def run_parameter_setting(kwargs):
    all_predictions: pd.Series
    all_references: pd.Series
    print("Model 1 running...")
    # clear_session()
    trained_model1, history1 = build_model_stateless1(kwargs["setting"], kwargs["X"], kwargs["y"],
                                                      kwargs["verbose_para"], kwargs["save"])
    print("Model 1 training finished...")
    all_predictions, all_references = test_set_prediction(trained_model1, kwargs["setting"], kwargs["ts"],
                                                          kwargs["ts"].test_true, kwargs["X"], None, True, False)
    # clear_session()
    print("Model 1 prediction finished...")
    outputs_model1 = dict()
    for method in ["MSE", "RMSE", "NRMSE", "MAE", "MAPE"]:
        output: float = Switcher(method, all_predictions, all_references)
        outputs_model1[method] = output

    return history1, outputs_model1


if __name__ == "__main__":
    dirname = dirname(__file__)
    makedirs("./outputs", exist_ok=True)
    output_folder_path = join(dirname, 'outputs')
    Stijn = True
    path_history: str
    path_temperature: str

    if Stijn:
        path_history = "D:\AI_time_series_repos\TS_code\Forecasting\\basemodel\data\DF_three_series.csv"
        path_temperature = "D:\AI_time_series_repos\TS_code\Forecasting\\basemodel\data\DF_three_temp_series.csv"
        path_npy = "D:\AI_time_series_repos\TS_code\Forecasting\LSTM\VM_calculating_inputs_LSTM\output_arrays"

    else:
        path_history = ""
        path_temperature = ""
        path_npy = ""

    fullYeardata = pd.read_csv(path_history, index_col="date", parse_dates=True)
    # change this [0]!
    print(fullYeardata.columns)
    names = fullYeardata.columns
    av_temperature = pd.read_csv(path_temperature, index_col="meter_id", parse_dates=True)

    print("Running this file on a PC with %s cores..." % (cpu_count()))
    results_file_name = join(output_folder_path, str(time()) + ".txt")
    chosen_parameters = ParameterSearch()
    amount_of_possibilities: int = reduce((lambda x, y: x * y),
                                          [len(x) for x in list(vars(chosen_parameters).values())])

    with open(results_file_name, "w") as results_file:
        results_file.write("Running the similation on a PC with %s cores...\r\n" % (cpu_count()))
        print("Found %s sets of parameters." % amount_of_possibilities)

    which_model = ["model1_sl"]
    start_time_program = time()
    for name in names:
        error = pd.DataFrame()
        logBookIndex = list(vars(forecast_setting()).keys())
        logBook = pd.DataFrame(index=logBookIndex)
        training_result = pd.DataFrame(index=["#epoch", "#training loss final", "#validation loss final"])

        # load the LSTM input matrices
        start_time = time()
        ts = time_serie(fullYeardata[name], av_temperature)
        end_time = time()

        print("the time needed is: %s"%(end_time-start_time))