"""
This file is testing the different base models.
The test set used are the 31 days of December of the 261 time-series with measurements of the full year 2017.
Forecasts are only included in the evaluation metrics when all the different models output a forecast
different then nan.

Evaluation metrics used: MSE, RMSE, NRMSE, MAE, MAPE

model 1. find most similar consumption in trainingset based on id house, which day, which time, which temperature
model 2. previous day
model 3. previous week
model 4. Mean forecast based on time of day and type of day
model 5. Empirical mape minimization

"""

import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Test_basemodel_functions import *

plt.rc('axes', linewidth=2)
plt.rc('axes', labelsize= 16)
plt.rc('axes',titlesize = 18)
plt.rc('legend',fontsize=14)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('figure',figsize=(10,8))

fullYeardata = pd.read_csv("D:\Onedrive\Leuven\Final project\data\Forecasting_writtendata\FullYear.csv",index_col= "date",parse_dates= True)
av_temperature = pd.read_csv("D:\Onedrive\Leuven\Final project\data\weather-avg.csv",index_col='meter_id')
av_temperature = av_temperature.transpose()
av_temperature.index = pd.to_datetime(av_temperature.index)

vertical_stack = pd.DataFrame()

for col_name in fullYeardata.columns:
    TS = fullYeardata[col_name]
    TS_temperature = av_temperature[col_name]
    TS_december = TS[TS.index.month == 12]

    forecast1 = find_most_similar_day(TS_december.index,TS,TS_temperature)
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


# Evaluate the different methods:
vertical_stack.dropna(inplace=True)
real_values = vertical_stack['real_values']
for method in ["MSE","RMSE","NRMSE","MAE","MAPE"]:
    outputs = dict()
    for col_name in vertical_stack.columns[:-1]: # expects the real values at the end of the DataFrame
        forecast = vertical_stack[col_name]
        output:float = Switcher(method,forecast,real_values)
        outputs[col_name] = output

    titels = list(outputs.keys())
    values = list(outputs.values())
    plt.figure(method)
    axis = figure_layout(figsize=(18, 12), titel=method, grid=False)
    # axis.grid(axis='x')
    rects = axis.bar(titels, values, color='maroon', width=0.4)
    autolabel(rects, axis)

    fname = "basemodel_figures/" + method + "_basemodel.png"
    plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w', orientation='portrait', format=None,
                transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)

plt.close("all")