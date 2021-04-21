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
from azureml.core import Run
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
    normalize_outputs = True
    path_history:str
    path_temperature:str
    if Stijn:
        path_history = path.join(args.data_path, "FullYear.csv")
        path_temperature = path.join(args.data_path, "weather-avg.csv")
    else:
        path_history = "Path is not yet indicated."
        path_temperature = "Path is not yet indicated."

    fullYeardata = pd.read_csv(path_history,index_col= "date",parse_dates= True)
    av_temperature = pd.read_csv(path_temperature,index_col="meter_id")
    av_temperature = av_temperature.transpose()
    av_temperature.index = pd.to_datetime(av_temperature.index)

    print("This is the data...")
    print(fullYeardata.head())
    print(av_temperature.head())

    print("Running is started...")
    counter = 0
    total_count = len(fullYeardata.columns)

    final_dict = {"MSE": np.array([0,0,0,0,0]),
                  "RMSE": np.array([0,0,0,0,0]),
                  "NRMSE": np.array([0,0,0,0,0]),
                  "MAE": np.array([0,0,0,0,0])}
    titels = []
    for col_name in fullYeardata.columns:
        TS = fullYeardata[col_name]
        TS_temperature = av_temperature[col_name]
        TS_december = TS[TS.index.month == 12]

        forecast1 = find_most_similar_day(TS_december.index,TS,TS_temperature,time_steps =48)
        forecast2 = base_model_week_before(TS_december.index,TS,amount_days=1)
        forecast3 = base_model_week_before(TS_december.index, TS, amount_days=7)
        forecast4 = mean_forecast(TS_december.index,TS)
        forecast5 = MAPE_estimator(TS_december.index,TS)
        real_values = TS_december

        df = pd.DataFrame()
        df[forecast1.name] = forecast1
        df[forecast2.name] = forecast2
        df[forecast3.name] = forecast3
        df[forecast4.name] = forecast4
        df[forecast5.name] = forecast5
        df['real_values'] = real_values

        df.dropna(inplace=True, axis=0)
        if len(df['real_values'])>0:
            # Evaluate the different methods:

            print("The amount of predicted values: %s." % len(df['real_values']))
            print("The amount of predicted days: %s."%(len(df['real_values'])/48))

            for method in ["MSE","RMSE","NRMSE","MAE"]:
                outputs = dict()
                for col_name2 in df.columns[:-1]: # expects the real values at the end of the DataFrame
                    forecast = df[col_name2]
                    real_values_input = df['real_values']
                    output:float = Switcher(method,forecast,real_values_input)
                    outputs[col_name2] = output

                titels = list(outputs.keys())

                if normalize_outputs:
                    max_value = max(outputs.values())
                    outputs = {key: outputs[key] / max_value
                           for key in outputs.keys()}
                output_list = [value for value in outputs.values()]
                final_dict[method] = final_dict[method] + np.array(output_list)
            counter += 1
        print("%s of %s time-series are trained.\n" % (counter, total_count))
    print("final dict: %s"%final_dict)

    final_dict = {key: final_dict[key] / counter
               for key in final_dict.keys()}
    for (method,values) in final_dict.items():
        axis,fig = figure_layout_VM(figsize=(18, 12), titel=method, grid=False)
        rects = axis.bar(titels, list(values), color='maroon', width=0.4)
        autolabel(rects, axis)
        makedirs("./barplots",exist_ok=True)
        run.log_image(name="./barplots/mean" + "_" + method + "_basemodel.png",plot=fig, description="barplot performance")

    print(100*"#")
    print("All the series are trained and evaluated!")