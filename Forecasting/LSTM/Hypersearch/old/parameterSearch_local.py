from forecasting_functions import *
from multiprocessing import Pool, cpu_count
from time import time
from functools import reduce
from os import makedirs, path
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from keras.backend import clear_session, reset_uids

# parameters that are tried
# first stage --> regularization off
class ParameterSearch:
    def __init__(self):
        self.list_units_LSTM = [20, 50]
        self.list_layers_LSTM = [1, 3]
        self.list_dropout_LSTM = [0]
        self.list_recurrent_dropout_LSTM = [0]
        self.list_kernel_regularization_LSTM = [None]
        self.list_recurrent_regularization_LSTM = [None]
        self.list_bais_regularization_LSTM = [None]
        self.list_activity_regularization_LSTM = [None]

        self.list_units_DENSE = [50]
        self.list_layers_DENSE = [1]
        self.list_dropout_DENSE = [0]
        self.list_kernel_regularization_DENSE = [None]
        self.list_bais_regularization_DENSE = [None]
        self.list_activity_regularization_DENSE = [None]
        self.list_lag_value = [48, 96]
        self.list_nb_epoch = [1]
        self.list_activation = ['relu']
        self.list_batch_size_parameter = [48]
        self.list_learning_rate = [10 ** -3, 10 ** -2, 10 ** -1]
        self.list_patience = [0]
        self.list_shuffle = ['True']  # shuffling is set to True
        self.list_repeat = [3]  # four is chosen because have four cores

        assert self.list_patience[0] < self.list_nb_epoch[0]

def run_parameter_setting(kwargs):
    print("Model 1 running...")
    trained_model1, history1 = build_model_stateless1(kwargs["setting"], kwargs["X"], kwargs["y"], kwargs["verbose_para"], kwargs["save"])

    all_predictions, all_references = test_set_prediction(trained_model1, kwargs["setting"], kwargs["ts"], kwargs["ts"].test_true, kwargs["X"], None, True, False)
    print("Model 1 prediction finished...")
    outputs_model1 = dict()
    for method in ["MSE", "RMSE", "NRMSE", "MAE", "MAPE"]:
        output: float = Switcher(method, all_predictions, all_references)
        outputs_model1[method] = output

    return history1, outputs_model1


if __name__ == "__main__":

    Stijn = True

    if Stijn:
        path_history = "/data/DF_three_series.csv"
        path_temperature = "/data/DF_three_temp_series.csv"
        path_npy = "/VM_calculating_inputs_LSTM/output_arrays"

    else:
        path_history = ""
        path_temperature = ""
        path_npy = ""

    fullYeardata = pd.read_csv(path_history,index_col= "date",parse_dates= True)
    names = fullYeardata.columns
    av_temperature = pd.read_csv(path_temperature,index_col="meter_id",parse_dates=True)

    print("Running this file on a PC with %s cores..." % (cpu_count()))

    makedirs("../local/outputs", exist_ok=True)
    output_path = "../local/outputs"
    path_txt_file = '../local/outputs/output_file.txt'
    path_to_array = "/VM_calculating_inputs_LSTM/output_arrays"

    print("CSV files are loaded...")

# start of file
    print("Running this file on a PC with %s cores..." % (cpu_count()))
    with open(path_txt_file,"w") as txt_file:
        txt_file.write("Running this file on a PC with %s cores...\n" % (cpu_count()))

    chosen_parameters = ParameterSearch()
    amount_of_possibilities: int = reduce((lambda x, y: x * y), [len(x) for x in list(vars(chosen_parameters).values())])

    print("Found %s sets of parameters." % amount_of_possibilities)
    with open(path_txt_file,"a") as txt_file:
        txt_file.write("Found %s sets of parameters.\n" % amount_of_possibilities)

    which_model = ["model1_sl"]
    start_time_program = time()
    multithreading = False
    counter = 1
    #change this!!
    ##################
    for name in names:
        error = pd.DataFrame()
        logBookIndex = list(vars(forecast_setting()).keys())
        logBook = pd.DataFrame(index=logBookIndex)
        training_result = pd.DataFrame(index=["#epoch", "#training loss final"])

        ts = time_serie(fullYeardata[name], av_temperature)
        setting_identification = 1
        for lag_value in chosen_parameters.list_lag_value:
            print("lag_value: %s \r\n" % lag_value)

            # load the LSTM input matrices
            path_X_train = path.join(path_to_array, "X_" + name + "_" + str(lag_value) + ".npy")
            X_train = np.load(path_X_train)
            path_y_train = path.join(path_to_array, "y_" + name + "_" + str(lag_value) + ".npy")
            y_train = np.load(path_y_train)
            # take the last 10 days of November as validation for the parameter search
            X_train = X_train[:-480]
            y_train = y_train[:-480]
            ts.test_true = ts.training_true[-480:]
            # X_train,y_train = unison_shuffled_copies(X_train,y_train) # no val anymore
            print("inputs found...\r\n")

            print("The shape of X_train: %s"%(X_train.shape,))
            print("The shape of y_train: %s" % (y_train.shape,))

            print("The parameter search has started for serie %s.\r\n" % name)
            print(50 * "-" + "\r\n")

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
                                    print("start iteration. \r\n")

                                    if multithreading:
                                        p = Pool(processes=cpu_count())
                                        training_results = p.map(run_parameter_setting, [{"setting": runner, "ts": ts, "X": X_train, "y": y_train, "verbose_para": 1,"save": False} for iteration in range(repeat)])
                                        clear_session()
                                        reset_uids()
                                        collected_histories = [x[0] for x in training_results]
                                        collected_outputs = [x[1] for x in training_results]

                                    else:
                                        collected_histories = []
                                        collected_outputs = []
                                        for iteration in range(repeat):
                                            history,outputs_model = run_parameter_setting({"setting": runner,"ts":ts, "X": X_train, "y": y_train, "verbose_para":1, "save": False})
                                            collected_histories.append(history)
                                            collected_outputs.append(outputs_model)
                                            clear_session()
                                            reset_uids()
                                    with open(path_txt_file,"a") as txt_file:
                                        print(20 * "*" + "\r\n")
                                        txt_file.write(20 * "*" + "\r\n")
                                        print("This is model: %s. \r\n"%which_model[0])
                                        txt_file.write("This is model: %s. \r\n" % which_model[0])
                                        print("This is setting identifier %s \r\n" % setting_identification)
                                        txt_file.write("This is setting identifier %s \r\n" % setting_identification)
                                        print(20*"*" + "\r\n")
                                        txt_file.write(20 * "*" + "\r\n")

                                        for r in np.arange(1,repeat + 1): # patience is the number of epochs without improvement
                                            history = collected_histories[r-1]
                                            print("Run: %s \r\n"%r)
                                            txt_file.write("Run: %s \r\n" % r)
                                            print("loss: %s \r\n" % history.history["loss"])
                                            txt_file.write("loss: %s \r\n" % history.history["loss"])
                                            # print("val_loss: %s \r\n" % history.history["val_loss"])
                                            # txt_file.write("val_loss: %s \r\n" % history.history["val_loss"])
                                            # training_result[str(setting_identification)+"_run_"+str(r)] = [len(history.history["loss"]) - patience, history.history["loss"][-1-patience], history.history["val_loss"][-1-patience]]
                                            training_result[str(setting_identification) + "_run_" + str(r)] = [
                                                len(history.history["loss"]) - patience,
                                                history.history["loss"][-1 - patience]]


                                        for method in ["MSE","RMSE","NRMSE","MAE","MAPE"]:
                                            collection_errors = [output_dict[method] for output_dict in collected_outputs]
                                            error[str(setting_identification)+"_"+method] = collection_errors
                                        end_time = time()

                                        print("Working on Serie: %s."%name)
                                        txt_file.write("Working on Serie: %s.\r\n" % name)
                                        print("Setting %s/%s is completed"%(setting_identification, amount_of_possibilities))
                                        txt_file.write("Setting %s/%s is completed\r\n" % (setting_identification, amount_of_possibilities))
                                        duration = end_time-start_time
                                        predicted_finish = (amount_of_possibilities*len(names)- counter)*duration
                                        print("The elapsed time to calculate one parameter is: %s minutes \r\n"%(duration/60))
                                        txt_file.write("The elapsed time to calculate one parameter is: %s minutes\n" % (duration/60))
                                        print("The expected remaining running time is: %s minutes.\r\n"%(predicted_finish/60))
                                    setting_identification += 1
                                    counter += 1

        error_csv_path = path.join(output_path,"error_" +which_model[0] + "_" +  name + ".csv")
        error.to_csv(error_csv_path)
        logBook_csv_path = path.join(output_path, "logbook_" + which_model[0] + "_"+ name + ".csv")
        logBook.to_csv(logBook_csv_path)
        training_result_csv_path = path.join(output_path, "training_result_" + which_model[0] + "_" + name + ".csv")
        training_result.to_csv(training_result_csv_path)
        end_time_program = time()
        with open(path_txt_file, "a") as txt_file:
            txt_file.write(100 * "#" + "\n")
            txt_file.write("Completed Serie: %s...\n" % name)
            txt_file.write(100*"#"+"\n")
            txt_file.write("The program ran for %s minutes." % ((end_time_program - start_time_program) / 60))
        print("Completed Serie: %s..." % name)
    print("Finished simulation...")
