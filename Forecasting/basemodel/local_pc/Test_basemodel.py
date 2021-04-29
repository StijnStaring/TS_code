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
from Test_basemodel_functions import *

Stijn = True
path_history:str
path_temperature:str
save_to:str
if Stijn:
    path_history = "../data/DF_three_series.csv"
    path_temperature = "../data/DF_three_temp_series.csv"
    save_to = "D:\Onedrive\Leuven\Final project\Results\Forecasting\Base_model_figures"
else:
    path_history = ""
    path_temperature = ""
    save_to = ""
fullYeardata = pd.read_csv(path_history,index_col= "date",parse_dates= True)
av_temperature = pd.read_csv(path_temperature,index_col="meter_id",parse_dates=True)


vertical_stack = pd.DataFrame()
counter = 0
total_count = len(fullYeardata.columns)
for col_name in fullYeardata.columns:
    TS = fullYeardata[col_name]
    TS_temperature = av_temperature[col_name]
    TS_december = TS[TS.index.month == 12]

    forecast1 = find_most_similar_day(TS_december.index,TS,TS_temperature,time_steps =48)
    forecast2 = base_model_week_before(TS_december.index,TS,amount_days=1)
    forecast3 = base_model_week_before(TS_december.index, TS, amount_days=7)
    forecast4 = mean_forecast(TS_december.index,TS)
    forecast5 = MAPE_estimator(TS_december.index,TS)

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


    vertical_stack = pd.concat([vertical_stack,df], axis=0, ignore_index=True)
    counter += 1
    print("%s of %s time-series are trained.\n"%(counter,total_count))


    # Evaluate the different methods:
    vertical_stack.dropna(inplace=True,axis=0)
    real_values = vertical_stack['real_values']
    for method in ["MSE","RMSE","NRMSE","MAE"]:
        outputs = dict()
        for col_name2 in vertical_stack.columns[:-1]: # expects the real values at the end of the DataFrame
            forecast = vertical_stack[col_name2]
            output:float = Switcher(method,forecast,real_values)
            outputs[col_name2] = output

        titels = list(outputs.keys())
        values = list(outputs.values())
        # plt.figure(method)
        axis = figure_layout(figsize=(18, 12), titel=method, grid=False)
        # axis.grid(axis='x')
        rects = axis.bar(titels, values, color='maroon', width=0.4)
        autolabel(rects, axis)

        fname = save_to + "Serie"+str(col_name) + "_" + method + "_basemodel.png"
        plt.savefig(fname, dpi=300, facecolor='w', edgecolor='w', orientation='portrait', format=None,
                    transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)

print(100*"#")
print("All the series are trained and evaluated!")


