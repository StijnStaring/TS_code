"""
This file is testing the different base models.
The test set used are the 31 days of December of the 261 time-series with measurements of the full year 2017.
Forecasts are only included in the evaluation metrics when all the different models output a forecast
different than nan.

Evaluation metrics used: MSE, RMSE, NRMSE, MAE, (MAPE is discarted because real values are sometimes zero)

model 1. find most similar consumption in trainingset based on id house, which day, which time, which temperature
model 2. previous day
model 3. previous week
model 4. Mean forecast based on time of day and type of day
model 5. Empirical mape minimization

"""
from Test_basemodel_functions_VM import *
from forecasting_functions_VM import show_all_forecasts
from azureml.core import Run, Workspace
import argparse
from os import path, listdir, makedirs

run = Run.get_context()

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
    av_temperature = pd.read_csv(path_temperature,index_col="meter_id",parse_dates=True)

    print("Running is started...")
    counter = 0
    total_count = len(fullYeardata.columns)
    for col_name in fullYeardata.columns:
        vertical_stack = pd.DataFrame()
        TS = fullYeardata[col_name]
        TS_temperature = av_temperature[col_name]
        TS_december = TS[TS.index.month == 12]

        forecast1 = find_most_similar_day(TS_december.index,TS,TS_temperature,time_steps =48)
        forecast2 = base_model_week_before(TS_december.index,TS,amount_days=1)
        forecast3 = base_model_week_before(TS_december.index, TS, amount_days=7)
        forecast4 = mean_forecast(TS_december.index,TS)
        forecast5 = MAPE_estimator(TS_december.index,TS)
        real_values = TS_december

        run = show_all_forecasts(forecast1,real_values,col_name,run,"similar")
        run = show_all_forecasts(forecast2, real_values, col_name, run,"1ago")
        run = show_all_forecasts(forecast3, real_values, col_name, run, "7ago")
        run = show_all_forecasts(forecast4, real_values, col_name, run,"mean")
        run = show_all_forecasts(forecast5, real_values, col_name, run,"MAPE")

        # uploading the csv files
        data = {forecast1.name: forecast1,
                forecast2.name: forecast2,
                forecast3.name: forecast3,
                forecast4.name: forecast4,
                forecast5.name: forecast5,
                "real_values": real_values}

        CSV = pd.concat(data,axis=1)

        makedirs("./csv_files", exist_ok=True)
        CSV.to_csv("./csv_files/ID" + str(col_name) + ".csv")
        run.upload_file("./csv_files/ID" + str(col_name) + ".csv", "./csv_files/ID" + str(col_name) + ".csv")

        forecast1 = forecast1.reset_index(drop=True)
        forecast2 = forecast2.reset_index(drop=True)
        forecast3 = forecast3.reset_index(drop=True)
        forecast4 = forecast4.reset_index(drop=True)
        forecast5 = forecast5.reset_index(drop=True)
        real_values = TS_december.reset_index(drop=True)

        df = pd.DataFrame()
        df[forecast1.name] = forecast1
        df[forecast2.name] = forecast2
        df[forecast3.name] = forecast3
        df[forecast4.name] = forecast4
        df[forecast5.name] = forecast5
        df['real_values'] = real_values


        vertical_stack = pd.concat([vertical_stack, df], axis=0, ignore_index=True)

        # Evaluate the different methods:
        vertical_stack.dropna(inplace=True,axis=0)
        real_values = vertical_stack['real_values']
        print("The amount of predicted values: %s." % len(real_values))
        print("The amount of predicted days: %s."%(len(real_values)/48))

        for method in ["MSE","RMSE","NRMSE","MAE"]:
            outputs = dict()
            for col_name2 in vertical_stack.columns[:-1]: # expects the real values at the end of the DataFrame
                forecast = vertical_stack[col_name2]
                output:float = Switcher(method,forecast,real_values)
                outputs[col_name2] = output

            titels = list(outputs.keys())
            values = list(outputs.values())
            axis,fig = figure_layout_VM(figsize=(18, 12), titel=method, grid=False)
            rects = axis.bar(titels, values, color='maroon', width=0.4)
            autolabel(rects, axis)
            makedirs("./barplots",exist_ok=True)
            run.log_image(name="./barplots/Serie"+str(col_name) + "_" + method + "_basemodel.png",plot=fig, description="barplot performance")

        counter += 1
        print("%s of %s time-series are trained.\n" % (counter, total_count))

    print(100*"#")
    print("All the series are trained and evaluated!")