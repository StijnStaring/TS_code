from forecasting_functions import *
from multiprocessing import Pool, cpu_count
from time import time
from functools import reduce
from os import makedirs, path
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from keras.backend import clear_session, reset_uids

# first stage --> regularization off
class ParameterSearch:
    def __init__(self):
        self.list_units_LSTM = [20,50]
        self.list_layers_LSTM = [1,3]
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
        self.list_lag_value = [48]
        self.list_nb_epoch = [1]
        self.list_activation = ['relu'] # found that an activation function of relu gives bad results
        # self.list_batch_size_parameter = [48]
        self.list_batch_size_parameter = [1]
        # self.list_learning_rate = [10**-3, 10 **-4]
        self.list_learning_rate = [10 ** -4, 10 ** -3, 10**-2] # found that 10**-1 gave instable results
        self.list_patience = [0]
        self.list_shuffle = [False]  # shuffling is set to True only when stateless
        self.list_repeat = [3]  # four is chosen because have four cores

        assert self.list_patience[0] < self.list_nb_epoch[0]

def run_parameter_setting1(kwargs):
    print("Model 1 running...")
    trained_model, history = build_model_stateless1(kwargs["setting"], kwargs["X"], kwargs["y"], kwargs["verbose_para"], kwargs["save"])

    all_predictions, all_references = test_set_prediction(trained_model, kwargs["setting"], kwargs["ts"], kwargs["ts"].test_true, None, True, False)
    # show_all_forecasts(all_predictions, all_references, "nameless", False)
    print("Model 1 prediction finished...")
    outputs_model = dict()
    for method in ["MSE", "RMSE", "NRMSE", "MAE", "MAPE"]:
        output: float = Switcher(method, all_predictions, all_references)
        outputs_model[method] = output

    return history, outputs_model

def run_parameter_setting2(kwargs):
    print("Model 2 running...")
    trained_model, history = build_model_stateless2(kwargs["setting"], kwargs["X"], kwargs["y"], kwargs["verbose_para"], kwargs["save"])
    all_predictions, all_references = test_set_prediction(trained_model, kwargs["setting"], kwargs["ts"], kwargs["ts"].test_true, None, True, False)

    print("Model 2 prediction finished...")
    outputs_model = dict()
    for method in ["MSE", "RMSE", "NRMSE", "MAE", "MAPE"]:
        output: float = Switcher(method, all_predictions, all_references)
        outputs_model[method] = output

    return history, outputs_model


def run_parameter_setting3(kwargs):
    print("Model 3 running...")
    trained_model, history = build_model_stateful1(kwargs["setting"], kwargs["X"], kwargs["y"], kwargs["verbose_para"], kwargs["save"])
    all_predictions, all_references = test_set_prediction(trained_model, kwargs["setting"], kwargs["ts"], kwargs["ts"].test_true, kwargs["X_train_full"], True, True)
    print("Model 3 prediction finished...")
    outputs_model = dict()
    for method in ["MSE", "RMSE", "NRMSE", "MAE", "MAPE"]:
        output: float = Switcher(method, all_predictions, all_references)
        outputs_model[method] = output

    return history, outputs_model


def run_parameter_setting4(kwargs):

    collection_trained_models = []
    for model_number in range(48):
        print("model %s"%model_number)
        path_X_train = path.join(kwargs["path_to_array"], "X_" + kwargs["ts"].name + "_" + str(kwargs["setting"].lag_value) + "_" + str(model_number) + "_.npy")
        X_train = np.load(path_X_train)
        path_y_train = path.join(kwargs["path_to_array"], "y_" + kwargs["ts"].name + "_" + str(kwargs["setting"].lag_value) + "_" + str(model_number) + "_.npy")
        y_train = np.load(path_y_train)
        # take the last 10 days of November as validation for the parameter search
        X_train = X_train[:-10]
        y_train = y_train[:-10]

        print("inputs found...\r\n")

        print("model 4 running...")
        trained_model, history = build_model_stateful1(kwargs["setting"], X_train, y_train)
        collection_trained_models.append(trained_model)


    all_predictions, all_references = test_set_prediction_sf(kwargs["path_to_array"], collection_trained_models, kwargs["setting"], kwargs["ts"],
                                                                 kwargs["ts"].test_true, True)
    print("Model 4 prediction finished.")

    for method in ["MSE", "RMSE", "NRMSE", "MAE", "MAPE"]:
        output: float = Switcher(method, all_predictions, all_references)
        outputs_model[method] = output

    return outputs_model



if __name__ == "__main__":
    which_model = "model4_sf"
    Stijn = True

    if Stijn:
        path_history = "D:\AI_time_series_repos\TS_code\Forecasting\\basemodel\data\DF_three_series.csv"
        path_temperature = "D:\AI_time_series_repos\TS_code\Forecasting\\basemodel\data\DF_three_temp_series.csv"
        path_to_array = "D:\AI_time_series_repos\TS_code\Forecasting\LSTM\VM_calculating_inputs_LSTM\output_arrays_sf48"

    else:
        path_history = ""
        path_temperature = ""
        path_to_array = ""

    fullYeardata = pd.read_csv(path_history,index_col= "date",parse_dates= True)
    names = fullYeardata.columns
    av_temperature = pd.read_csv(path_temperature,index_col="meter_id",parse_dates=True)


    if which_model == "model1_sl":
        makedirs("local/outputs", exist_ok=True)
        output_path = "local/outputs"
        path_txt_file = 'local/outputs/output_file.txt'
    elif which_model == "model2_sl":
        makedirs("local_stateless2/outputs", exist_ok=True)
        output_path = "local_stateless2/outputs"
        path_txt_file = 'local_stateless2/outputs/output_file.txt'

    elif which_model == "model3_sf":
        makedirs("local_stateful3/outputs", exist_ok= True)
        output_path = "local_stateful3/outputs"
        path_txt_file = 'local_stateful3/outputs/output_file.txt'

    elif which_model == "model4_sf":
        makedirs("local_stateful4/outputs", exist_ok=True)
        output_path = "local_stateful4/outputs"
        path_txt_file = 'local_stateful4/outputs/output_file.txt'
    else:
        output_path = ""
        path_txt_file = ""


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


    start_time_program = time()
    multithreading = False
    counter = 1

    for name in names:
        error = pd.DataFrame()
        logBookIndex = list(vars(forecast_setting()).keys())
        logBook = pd.DataFrame(index=logBookIndex)
        training_result = pd.DataFrame(index=["#epoch", "#training loss final"])

        ts = time_serie(fullYeardata[name], av_temperature)
        setting_identification = 1
        for lag_value in chosen_parameters.list_lag_value:

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
                                    history = None
                                    outputs_model = None

                                    if multithreading:
                                        training_results = None
                                        p = Pool(processes=cpu_count())
                                        if which_model == "model1_sl":
                                            training_results = p.map(run_parameter_setting1, [{"setting": runner, "ts": ts, "X": X_train, "y": y_train, "verbose_para": 1,"save": False} for iteration in range(repeat)])
                                        elif which_model == "model2_sl":
                                            training_results = p.map(run_parameter_setting2, [{"setting": runner, "ts": ts, "X": X_train, "y": y_train, "verbose_para": 1,"save": False} for iteration in range(repeat)])

                                        elif which_model == "model3_sf":
                                            training_results = p.map(run_parameter_setting3, [{"setting": runner, "ts": ts, "X": X_train, "y": y_train,"verbose_para": 1, "save": False, "X_train_full": X_train_full} for iteration in range(repeat)])

                                        clear_session()
                                        reset_uids()
                                        collected_histories = [x[0] for x in training_results]
                                        collected_outputs = [x[1] for x in training_results]

                                    else:
                                        collected_histories = []
                                        collected_outputs = []

                                        for iteration in range(repeat):
                                            if which_model == "model1_sl":
                                                history,outputs_model = run_parameter_setting1({"setting": runner,"ts":ts, "X": X_train, "y": y_train, "verbose_para":1, "save": False})
                                            elif which_model == "model2_sl":
                                                history, outputs_model = run_parameter_setting2(
                                                    {"setting": runner, "ts": ts, "X": X_train, "y": y_train,
                                                     "verbose_para": 1, "save": False})
                                            elif which_model == "model3_sf":
                                                history, outputs_model = run_parameter_setting3(
                                                    {"setting": runner, "ts": ts, "X": X_train, "y": y_train,
                                                     "verbose_para": 1, "save": False, "X_train_full": X_train_full})

                                            elif which_model == "model4_sf":
                                                assert runner.lag_value == 48
                                                outputs_model = run_parameter_setting4(
                                                    {"setting": runner, "ts": ts, "path_to_array":path_to_array})

                                            # collected_histories.append(history)
                                            collected_outputs.append(outputs_model)
                                            clear_session()
                                            reset_uids()
                                    with open(path_txt_file,"a") as txt_file:
                                        print(20 * "*" + "\r\n")
                                        txt_file.write(20 * "*" + "\r\n")
                                        print("This is model: %s. \r\n"%which_model)
                                        txt_file.write("This is model: %s. \r\n" % which_model)
                                        print("This is setting identifier %s \r\n" % setting_identification)
                                        txt_file.write("This is setting identifier %s \r\n" % setting_identification)
                                        print(20*"*" + "\r\n")
                                        txt_file.write(20 * "*" + "\r\n")

                                        # for r in np.arange(1,repeat + 1): # patience is the number of epochs without improvement
                                        #     history = collected_histories[r-1]
                                        #     print("Run: %s \r\n"%r)
                                        #     txt_file.write("Run: %s \r\n" % r)
                                        #     print("loss: %s \r\n" % history.history["loss"])
                                        #     txt_file.write("loss: %s \r\n" % history.history["loss"])
                                        #     # print("val_loss: %s \r\n" % history.history["val_loss"])
                                        #     # txt_file.write("val_loss: %s \r\n" % history.history["val_loss"])
                                        #     # training_result[str(setting_identification)+"_run_"+str(r)] = [len(history.history["loss"]) - patience, history.history["loss"][-1-patience], history.history["val_loss"][-1-patience]]
                                        #     training_result[str(setting_identification) + "_run_" + str(r)] = [
                                        #         len(history.history["loss"]) - patience,
                                        #         history.history["loss"][-1 - patience]]


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

        error_csv_path = path.join(output_path,"error_" +which_model + "_" +  name + ".csv")
        error.to_csv(error_csv_path)
        logBook_csv_path = path.join(output_path, "logbook_" + which_model + "_"+ name + ".csv")
        logBook.to_csv(logBook_csv_path)
        training_result_csv_path = path.join(output_path, "training_result_" + which_model + "_" + name + ".csv")
        training_result.to_csv(training_result_csv_path)
        end_time_program = time()
        with open(path_txt_file, "a") as txt_file:
            txt_file.write(100 * "#" + "\n")
            txt_file.write("Completed Serie: %s...\n" % name)
            txt_file.write(100*"#"+"\n")
            txt_file.write("The program ran for %s minutes." % ((end_time_program - start_time_program) / 60))
        print("Completed Serie: %s..." % name)
    print("Finished simulation...")
