from forecasting_functions_VM import *
from time import time
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from azureml.core import Run
import argparse
from os import path, listdir, makedirs
"""
Using a promo GPU in the azure cloud for calculating the inputs to the LSTM and saving them for later use.
"""
run = Run.get_context()

def run_once_input_LSTM(kwargs):
    print("Running LSTM input...")
    return input_output_LSTM(kwargs["ts"].training_true, kwargs["ts"].temperature_norm, kwargs["lag_value"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        help='Path to the training data'
    )
    args = parser.parse_args()
    print("===== DATA =====")
    print("DATA PATH: " + args.data_path)
    print("LIST FILES IN DATA PATH...")
    print(listdir(args.data_path))
    print("================")

    Stijn = True
    path_history:str
    path_temperature:str
    if Stijn:
        path_history = path.join(args.data_path, "DF_three_series.csv")
        path_temperature = path.join(args.data_path, "DF_three_temp_series.csv")
    else:
        path_history = "Path is not yet indicated."
        path_temperature = "Path is not yet indicated."

    fullYeardata = pd.read_csv(path_history,index_col= "date",parse_dates= True)
    names = fullYeardata.columns
    av_temperature = pd.read_csv(path_temperature,index_col="meter_id",parse_dates=True)

    print("This is the data...")
    print(fullYeardata.head())
    print(av_temperature.head())

    makedirs("./outputs", exist_ok=True)


    # results file
    with open('./outputs/output_file.txt', 'w') as file:
        file.write("This file contains the results of the calculated data.\n")
        file.write("Running is started...\n")

    start_time_program = time()
    # list_lag_value = [48, 96, 336, 672]
    list_lag_value = [336]
    for name in names:
        with open('./outputs/output_file.txt', 'a') as file:
            file.write("Staring new serie: %s.\n"%name)
            file.write(100*"*"+"\n")
        ts = time_serie(fullYeardata[name], av_temperature)
        for lag_value in list_lag_value:
            with open('./outputs/output_file.txt', 'a') as file:
                file.write(100*"-"+"\n")
                file.write("lag_value: %s \r\n"%lag_value)
            start_function_call = time()
            X_train,y_train = run_once_input_LSTM({"ts":ts, "lag_value": lag_value})
            end_function_call = time()
            calc_time = (end_function_call - start_function_call)
            with open('./outputs/output_file.txt', 'a') as file:
                file.write("The function call for Serie: %s and lag value %s took %s seconds --> %s minutes\n."%(name, lag_value,calc_time,calc_time/60))
                file.write("The dimensions of X_train: %s.\n"%(X_train.shape,))
                file.write("The dimensions of y_train: %s.\n" % (y_train.shape,))

            # saving the matrices
            np.save("./outputs/X_" + name+"_"+str(lag_value)+".npy", X_train)
            np.save("./outputs/y_" + name + "_" + str(lag_value) + ".npy", y_train)

    end_time_program = time()
    time_program = end_time_program - start_time_program
    with open('./outputs/output_file.txt', 'a') as file:
        file.write("\nThe program ran for %s minutes."%(time_program/60))

