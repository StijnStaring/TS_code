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

        self.list_lag_value = [48,96,336,672] # four is chosen becaus have four cores
        self.list_nb_epoch = [1000]
        self.list_activation = ['relu']
        self.list_batch_size_parameter = [1,32,64]
        self.list_learning_rate = [10**-3,10**-2,10**-1]
        self.list_patience = [5]
        self.list_shuffle = ['True']
        self.list_repeat = [4] #four is chosen because have four cores

def run_once_input_LSTM(kwargs):
    print("Running LSTM input...")
    # Now are still trying with toy trainingset
    return input_output_LSTM(kwargs["ts"].training_true, kwargs["ts"].temperature_norm, kwargs["lag_value"])

def run_parameter_setting(kwargs):
    all_predictions: pd.Series
    all_references: pd.Series
    print("Model 1 running...")
    # clear_session()
    trained_model1, history1 = build_model_stateless1(kwargs["setting"], kwargs["X"], kwargs["y"], kwargs["verbose_para"], kwargs["save"])
    # remember that you have to set the test to the real test values!
    all_predictions, all_references = test_set_prediction(trained_model1, kwargs["setting"], kwargs["ts"], kwargs["ts"].test_true, kwargs["X"], None, True, False)
    # clear_session()
    print("Model 1 prediction finished...")
    outputs_model1 = dict()
    for method in ["MSE", "RMSE", "NRMSE", "MAE", "MAPE"]:
        output: float = Switcher(method, all_predictions, all_references)
        outputs_model1[method] = output

    # print("Model 2 running...")
    # trained_model2, history2, graph2 = build_model_stateless2(kwargs["setting"], kwargs["X"], kwargs["y"], kwargs["X_val"], kwargs["y_val"], kwargs["verbose_para"], kwargs["save"])
    # print("Model 2 training_finished...")
    # # clear_session()
    #
    # all_predictions, all_references = test_set_prediction(trained_model2, kwargs["setting"], kwargs["ts"], kwargs["ts"].test, kwargs["X"], kwargs["X_val"], True, False)
    # # clear_session()
    # outputs_model2 = dict()
    # for method in ["MSE", "RMSE", "NRMSE", "MAE", "MAPE"]:
    #     output: float = Switcher(method, all_predictions, all_references)
    #     outputs_model2[method] = output

    # return (history1,outputs_model1),(history2,outputs_model2)
    return history1, outputs_model1



if __name__ == "__main__":
    dirname = dirname(__file__)
    makedirs("./outputs", exist_ok=True)
    output_folder_path = join(dirname, 'outputs')
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
    results_file_name = join(output_folder_path,str(time()) + ".txt")
    results_file = open(results_file_name, "w")
    results_file.write("Running the similation on a PC with %s cores...\r\n" % (cpu_count()))
    results_file.close()
    chosen_parameters = ParameterSearch()
    amount_of_possibilities: int = reduce((lambda x, y: x * y), [len(x) for x in list(vars(chosen_parameters).values())])
    print("Found %s sets of parameters." % amount_of_possibilities)

    which_model = ["model1_sl"]
    start_time_program = time()
    for name in names:
        start_inputs = time()
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

        #The data that is going in, is shuffled! This is because when shuffle is true in the fit function the validation set
        #is still taken as the last samples.
        lag_dict = {key: unison_shuffled_copies(value[0], value[1]) for (key, value) in lag_dict.items()}
        print_dict = {key: (value[0].shape, value[1].shape) for (key, value) in lag_dict.items()}
        # print_dict = {key: (value[0].shape, value[1].shape, value[2].shape, value[3].shape) for (key, value) in lag_dict.items()}
        results_file = open(results_file_name, "a")
        results_file.write("print dict: %s.\r\n"%print_dict)
        results_file.write("The parameter search has started for serie %s.\r\n"%name)
        print(50*"-" + "\r\n")
        results_file.close()
        finished_inputs = time()

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
                                    start_time = time()
                                    repeat = chosen_parameters.list_repeat[0]
                                    runner = forecast_setting(units_LSTM = units_LSTM, layers_LSTM = layers_LSTM, units_DENSE = units_DENSE, layers_DENSE= layers_DENSE, patience = patience,
                                    shuffle = shuffle, lag_value = lag_value, nb_epoch = nb_epoch, batch_size_para = batch_size_parameter,
                                    repeat = repeat, activation = activation, learning_rate = learning_rate, dropout_LSTM = dropout_LSTM, recurrent_dropout_LSTM = recurrent_dropout_LSTM, kernel_regularizer_LSTM = kernel_regularization_LSTM,
                                    recurrent_regularizer_LSTM = recurrent_regularization_LSTM, bais_regularizer_LSTM = bais_regularization_LSTM, activity_regularizer_LSTM = activity_regularization_LSTM, dropout_DENSE = dropout_DENSE,
                                    kernel_regularizer_DENSE = kernel_regularization_DENSE, bais_regularizer_DENSE = bais_regularization_DENSE, activity_regularizer_DENSE = activity_regularization_DENSE)
                                    logBook[str(setting_identification)] = [str(x) for x in list(vars(runner).values())]
                                    # (X_train,y_train,X_val,y_val) = lag_dict.get(lag_value)
                                    (X_train, y_train) = lag_dict.get(lag_value)
                                    p = Pool(processes=cpu_count())
                                    training_results = p.map(run_parameter_setting, [{"setting": runner,"ts":ts, "X": X_train, "y": y_train, "verbose_para":0, "save": False} for iteration in range(repeat)])
                                    clear_session()
                                    reset_uids()

                                    results_file = open(results_file_name,"a")

                                    for ind in range(len(which_model)):
                                        results_file.write(20 * "*" + "\r\n")
                                        results_file.write("This is %s \r\n"%which_model[ind])
                                        results_file.write("This is setting identifier %s \r\n" % setting_identification)
                                        results_file.write(20*"*" + "\r\n")
                                        # collected_histories = [x[ind][0] for x in training_results]
                                        collected_histories = [x[0] for x in training_results]

                                        for r in np.arange(1,repeat + 1): # patience is the number of epochs without improvement
                                            history = collected_histories[r-1]
                                            results_file.write("Run: %s \r\n"%r)
                                            results_file.write("loss: %s \r\n" % history.history["loss"])
                                            results_file.write("val_loss: %s \r\n" % history.history["val_loss"])
                                            training_result[str(setting_identification)+"_run_"+str(r)] = [len(history.history["loss"]) - patience, history.history["loss"][-1-patience], history.history["val_loss"][-1-patience]]

                                        # collected_outputs = [x[ind][1] for x in training_results]
                                        collected_outputs = [x[1] for x in training_results]

                                        for method in ["MSE","RMSE","NRMSE","MAE","MAPE"]:
                                            collection_errors = [output_dict[method] for output_dict in collected_outputs]
                                            error[str(setting_identification)+"_"+method] = collection_errors


                                    print("Working on Serie: %s"%name)
                                    print("Setting %s/%s is completed"%(setting_identification, amount_of_possibilities))
                                    end_time = time()
                                    duration = end_time-start_time
                                    predicted_finish = amount_of_possibilities*3*duration + (finished_inputs-start_inputs)*3 + start_time_program
                                    print("The elapsed time to calculate one parameter is: %s"%duration)
                                    print("The expected remaining running time is: %s minutes."%((predicted_finish - time())/60))

                                    exit()

                                    results_file.write("Working on Serie: %s\r\n"%name)
                                    results_file.write("Setting %s/%s is completed\r\n"%(setting_identification, amount_of_possibilities))
                                    results_file.close()
                                    setting_identification += 1

        error_csv_path = join(output_folder_path,"error_" +which_model[0] + "_" +  name + ".csv")
        error.to_csv(error_csv_path)
        logBook_csv_path = join(output_folder_path, "logbook_" + which_model[0] + "_"+ name + ".csv")
        logBook.to_csv(logBook_csv_path)
        training_result_csv_path = join(output_folder_path, "training_result_" + which_model[0] + "_" + name + ".csv")
        training_result.to_csv(training_result_csv_path)
        print("Completed Serie: %s..." % name)
    results_file.close()
    print("Finished simulation - data written to txt file.")
