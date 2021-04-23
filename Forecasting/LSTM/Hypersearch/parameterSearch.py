from forecasting_functions import *
from multiprocessing import Pool, cpu_count
import time
from functools import reduce
import os
import warnings as wn
wn.filterwarnings(action='ignore')
from keras.backend import clear_session
# from tensorflow import get_default_graph
from tensorflow.compat.v1 import get_default_graph
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

        self.list_lag_value = [48] # four is chosen becaus have four cores
        self.list_nb_epoch = [1000]
        self.list_activation = ['relu']
        self.list_batch_size_parameter = [1]
        self.list_learning_rate = [10**-3,10**-2]
        self.list_patience = [5]
        self.list_shuffle = ['False']
        self.list_repeat = [4] #four is chosen because have four cores

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


def run_once_input_LSTM(kwargs):
    print("Running LSTM input...")
    # Now are still trying with toy trainingset
    return input_output_LSTM(kwargs["ts"].training, kwargs["ts"].temperature_norm, kwargs["lag_value"])

def run_parameter_setting(kwargs):
    all_predictions: pd.Series
    all_references: pd.Series
    print("Model 1 running...")
    trained_model1, history1, graph1 = build_model_stateless1(kwargs["setting"], kwargs["X"], kwargs["y"], kwargs["X_val"], kwargs["y_val"], kwargs["verbose_para"], kwargs["save"])
    # remember that you have to set the test to the real test values!
    print("Model 1 training_finished...")
    # clear_session()
    with graph1.as_default():
        all_predictions, all_references = test_set_prediction(trained_model1, kwargs["setting"], kwargs["ts"], kwargs["ts"].test, kwargs["X"], kwargs["X_val"], True, False)
    # clear_session()
    outputs_model1 = dict()
    for method in ["MSE", "RMSE", "NRMSE", "MAE", "MAPE"]:
        output: float = Switcher(method, all_predictions, all_references)
        outputs_model1[method] = output

    # print("Model 2 running...")
    # trained_model2, history2, graph2 = build_model_stateless2(kwargs["setting"], kwargs["X"], kwargs["y"], kwargs["X_val"], kwargs["y_val"], kwargs["verbose_para"], kwargs["save"])
    # print("Model 2 training_finished...")
    # # clear_session()
    # with graph2.as_default():
    #     all_predictions, all_references = test_set_prediction(trained_model2, kwargs["setting"], kwargs["ts"], kwargs["ts"].test, kwargs["X"], kwargs["X_val"], True, False)
    # # clear_session()
    # outputs_model2 = dict()
    # for method in ["MSE", "RMSE", "NRMSE", "MAE", "MAPE"]:
    #     output: float = Switcher(method, all_predictions, all_references)
    #     outputs_model2[method] = output

    # return (history1,outputs_model1),(history2,outputs_model2)
    return history1, outputs_model1

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
    results_file_name = os.path.join(output_folder_path,str(time.time()) + ".txt")
    results_file = open(results_file_name, "w")
    results_file.write("Running the similation on a PC with %s cores...\r\n" % (cpu_count()))
    results_file.close()
    chosen_parameters = ParameterSearch()
    amount_of_possibilities: int = reduce((lambda x, y: x * y), [len(x) for x in list(vars(chosen_parameters).values())])
    print("Found %s sets of parameters." % amount_of_possibilities)


    for name in names:
        error = pd.DataFrame()
        logBookIndex = list(vars(forecast_setting()).keys())
        logBook = pd.DataFrame(index=logBookIndex)
        training_result = pd.DataFrame(index=["#epoch", "#training loss final", "#validation loss final"])
        # calculate the LSTM input matrices
        results_file = open(results_file_name, "a")
        results_file.write("#" * 50 + "\r\n")
        results_file.write("The calculation of the LSTM inputs is started for serie: %s.\r\n"%name)
        results_file.write("-"*50 + "\r\n")
        ts = time_serie(fullYeardata[name], av_temperature)
        p = Pool(processes=cpu_count())
        results = p.map(run_once_input_LSTM, [{"ts":ts, "lag_value": lag_value} for lag_value in chosen_parameters.list_lag_value])
        lag_dict = {chosen_parameters.list_lag_value[i]: results[i] for i in range(len(chosen_parameters.list_lag_value))}
        lag_dict = create_validation_stateless(lag_dict,0.9)
        print_dict = {key: (value[0].shape, value[1].shape, value[2].shape, value[3].shape) for (key, value) in lag_dict.items()}
        results_file = open(results_file_name, "a")
        results_file.write("print dict: %s.\r\n"%print_dict)
        results_file.write("The parameter search has started for serie %s.\r\n"%name)
        print(50*"-" + "\r\n")
        results_file.close()

        # doing parameter search
        setting_identification = 1
        dropout_LSTM = chosen_parameters.list_dropout_LSTM[0]
        recurrent_dropout_LSTM = chosen_parameters.list_recurrent_dropout_LSTM[0]
        kernel_regularization_LSTM = chosen_parameters.list_kernel_regularization_LSTM[0]
        recurrent_regularization_LSTM  = chosen_parameters.list_recurrent_regularization_LSTM[0]
        bais_regularization_LSTM = chosen_parameters.list_bais_regularization_LSTM[0]
        activity_regularization_LSTM = chosen_parameters.list_activity_regularization_LSTM[0]

        dropout_DENSE = chosen_parameters.list_dropout_DENSE[0]
        kernel_regularization_DENSE = chosen_parameters.list_kernel_regularization_DENSE[0]
        bais_regularization_DENSE = chosen_parameters.list_bais_regularization_DENSE[0]
        activity_regularization_DENSE = chosen_parameters.list_activity_regularization_DENSE[0]

        nb_epoch = chosen_parameters.list_nb_epoch[0]
        activation = chosen_parameters.list_activation[0]
        patience = chosen_parameters.list_patience[0]
        shuffle = chosen_parameters.list_shuffle[0]

        for units_LSTM in chosen_parameters.list_units_LSTM:
            for layers_LSTM in chosen_parameters.list_layers_LSTM:
                for units_DENSE in chosen_parameters.list_units_DENSE:
                    for layers_DENSE in chosen_parameters.list_layers_DENSE:
                        for lag_value in chosen_parameters.list_lag_value:
                            for batch_size_parameter in chosen_parameters.list_batch_size_parameter:
                                for learning_rate in chosen_parameters.list_learning_rate:

                                    repeat = chosen_parameters.list_repeat[0]
                                    runner = forecast_setting(units_LSTM = units_LSTM, layers_LSTM = layers_LSTM, units_DENSE = units_DENSE, layers_DENSE= layers_DENSE, patience = patience,
                                    shuffle = shuffle, lag_value = lag_value, nb_epoch = nb_epoch, batch_size_para = batch_size_parameter,
                                    repeat = repeat, activation = activation, learning_rate = learning_rate, dropout_LSTM = dropout_LSTM, recurrent_dropout_LSTM = recurrent_dropout_LSTM, kernel_regularizer_LSTM = kernel_regularization_LSTM,
                                    recurrent_regularizer_LSTM = recurrent_regularization_LSTM, bais_regularizer_LSTM = bais_regularization_LSTM, activity_regularizer_LSTM = activity_regularization_LSTM, dropout_DENSE = dropout_DENSE,
                                    kernel_regularizer_DENSE = kernel_regularization_DENSE, bais_regularizer_DENSE = bais_regularization_DENSE, activity_regularizer_DENSE = activity_regularization_DENSE)
                                    logBook[str(setting_identification)] = [str(x) for x in list(vars(runner).values())]
                                    (X_train,y_train,X_val,y_val) = lag_dict.get(lag_value)
                                    p = Pool(processes=cpu_count())
                                    training_results = p.map(run_parameter_setting, [{"setting": runner,"ts":ts, "X": X_train, "y": y_train, "X_val":X_val, "y_val":y_val, "verbose_para":0, "save": False} for iteration in range(repeat)])
                                    which_model = ["model1_sl", "model2_sl"]
                                    results_file = open(results_file_name,"a")

                                    for ind in range(len(which_model)):
                                        results_file.write("This is %s \r\n"%which_model[ind])
                                        results_file.write(20*"*" + "\r\n")
                                        collected_histories = [x[ind][0] for x in training_results]

                                        for r in np.arange(1,repeat + 1): # patience is the number of epochs without improvement
                                            history = collected_histories[r-1]
                                            results_file.write("Run: %s \r\n"%r)
                                            results_file.write("loss: %s \r\n" % history.history["loss"])
                                            results_file.write("val_loss: %s \r\n" % history.history["val_loss"])
                                            training_result[str(setting_identification)+"_"+which_model[ind]+"_"+"_run_"+str(r)] = [len(history.history["loss"]) - patience, history.history["loss"][-1-patience], history.history["val_loss"][-1-patience]]

                                        collected_outputs = [x[ind][1] for x in training_results]

                                        for method in ["MSE","RMSE","NRMSE","MAE","MAPE"]:
                                            collection_errors = [output_dict[method] for output_dict in collected_outputs]
                                            error[str(setting_identification)+"_"+which_model[ind]+"_"+method] = collection_errors



                                    print("Working on Serie: %s"%name)
                                    print("Setting %s/%s is completed"%(setting_identification, amount_of_possibilities))

                                    results_file.write("Working on Serie: %s"%name)
                                    results_file.write("Setting %s/%s is completed"%(setting_identification, amount_of_possibilities))
                                    results_file.close()
                                    setting_identification += 1

        error_csv_path = os.path.join(output_folder_path,"error_" + name + ".csv")
        error.to_csv(error_csv_path)
        logBook_csv_path = os.path.join(output_folder_path, "logbook_" + name + ".csv")
        logBook.to_csv(logBook_csv_path)
        training_result_csv_path = os.path.join(output_folder_path, "training_result_" + name + ".csv")
        training_result.to_csv(training_result_csv_path)
        print("Completed Serie: %s..." % name)
    results_file.close()
    print("Finished simulation - data written to txt file.")
    # save the model with the best performance in a h5 file.