"""
Overview of the different basemodels.
"""
import datetime as dt
import pandas as pd # pandas
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, DateOffset, EasterMonday,
    GoodFriday, Holiday, MO,
    next_monday, next_monday_or_tuesday)
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import casadi as ca


plt.rc('axes', linewidth=2)
plt.rc('axes', labelsize= 16)
plt.rc('axes',titlesize = 18)
plt.rc('legend',fontsize=14)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('figure',figsize=(10,8))

def figure_layout(figsize=(10,8),titel="",xlabel="",ylabel="",fontsize_titel=18,fontsize_axis=16,fontsize_legend=14,fontsize_ticks=16,grid:bool = False):

    plt.figure(figsize=figsize)
    ax1 = plt.gca()
    plt.rc('legend',fontsize=fontsize_legend)
    plt.title(titel, fontsize=fontsize_titel, fontweight = 'bold')
    plt.grid(grid)
    plt.xlabel(xlabel, fontsize=fontsize_axis)
    plt.ylabel(ylabel, fontsize=fontsize_axis)
    for tick in ax1.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize_ticks)
    #         tick.label1.set_fontweight('bold')
    for tick in ax1.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize_ticks)
    #     tick.label1.set_fontweight('bold')

    return ax1

def figure_layout_VM(figsize=(10,8),titel="",xlabel="",ylabel="",fontsize_titel=18,fontsize_axis=16,fontsize_legend=14,fontsize_ticks=16,grid:bool = False,
                     dpi = 300):

    plt.figure(figsize=figsize,dpi=dpi)
    fig = plt.gcf()
    ax1 = plt.gca()
    plt.rc('legend',fontsize=fontsize_legend)
    plt.title(titel, fontsize=fontsize_titel, fontweight = 'bold')
    plt.grid(grid)
    plt.xlabel(xlabel, fontsize=fontsize_axis)
    plt.ylabel(ylabel, fontsize=fontsize_axis)
    for tick in ax1.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize_ticks)
    #         tick.label1.set_fontweight('bold')
    for tick in ax1.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize_ticks)
    #     tick.label1.set_fontweight('bold')

    return ax1,fig

# The model will not always produce an output --> can be that the previous day is not known
def base_model_week_before(test_dates: pd.DatetimeIndex,serie: pd.Series,amount_days:int = 7):
    base_forecast = pd.Series(index= test_dates,name= 'base_forecast: ' + str(amount_days) + ' day(s)')
    for d in test_dates:
        get_date = d + dt.timedelta(days=-amount_days)
        forecast = serie.loc[get_date]
        base_forecast[d] = forecast

    return base_forecast

# importing the holidays

class EnglandAndWalesHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('New Years Day', month=1, day=1, observance=next_monday),
        GoodFriday,
        EasterMonday,
        Holiday('Early May bank holiday',
                month=5, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday('Spring bank holiday',
                month=5, day=31, offset=DateOffset(weekday=MO(-1))),
        Holiday('Summer bank holiday',
                month=8, day=31, offset=DateOffset(weekday=MO(-1))),
        Holiday('Christmas Day', month=12, day=25, observance=next_monday),
        Holiday('Boxing Day',
                month=12, day=26, observance=next_monday_or_tuesday)
    ]

def norm(serie: pd.Series):
    values = serie.values
    l = len(values)
    values = values.reshape((l, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(values)
    norm_serie = scaler.transform(values)
    norm_serie = norm_serie.squeeze()
    norm_serie = pd.Series(data=norm_serie,index=serie.index)
    return norm_serie

def get_closest_day(training_days: pd.DatetimeIndex,temperature: pd.Series,daily_temp:float)-> pd.Timestamp:
    day = abs(temperature[training_days] - daily_temp).idxmin()
    return day

def find_most_similar_day(test_dates: pd.DatetimeIndex,serie: pd.Series,temperature: pd.Series,time_steps:int):
    # The assumption is made that the test_dates are one continuous sequence of days
    daily_test_dates = pd.date_range(start=test_dates[0],end=test_dates[-1],freq='D')
    base_forecast = pd.Series(index= test_dates,name= 'closest_day_forecast')
    temperature = norm(temperature)

    for day in daily_test_dates:
        training_days = pd.date_range(start=pd.Timestamp('2017-01-01'),end=day + dt.timedelta(-1),freq='D')
        holidays = EnglandAndWalesHolidayCalendar().holidays(start=pd.Timestamp('2017-01-01'),end=day + dt.timedelta(-1))
        training_days = training_days.symmetric_difference(holidays)
        day_temp = temperature[temperature.index.dayofyear == day.dayofyear][0]

        # if have no information about the temperature of the day --> can't forecast the day
        if np.isnan(day_temp) :
            forecast = np.ones(time_steps)
            forecast[:] = np.nan
            base_forecast[base_forecast.index.dayofyear == day.dayofyear] = forecast
            continue

        if day in holidays:
            # look for holiday and Sunday (to get more days) --> we know that similarity is highest with Sunday
            sundays = training_days[training_days.weekday == 6]
            training_days = sundays.union(holidays)
            cl_day = get_closest_day(training_days,temperature,day_temp)
            forecast = serie[serie.index.dayofyear == cl_day.dayofyear].values
            base_forecast[base_forecast.index.dayofyear == day.dayofyear] = forecast

        else:
            training_days = training_days[training_days.weekday == day.weekday()]
            cl_day = get_closest_day(training_days,temperature,day_temp)
            forecast = serie[serie.index.dayofyear == cl_day.dayofyear].values
            base_forecast[base_forecast.index.dayofyear == day.dayofyear] = forecast

    return base_forecast

def mean_forecast(test_dates: pd.DatetimeIndex,serie: pd.Series):
    # The assumption is made that the test_dates are one continuous sequence of days
    daily_test_dates = pd.date_range(start=test_dates[0],end=test_dates[-1],freq='D')
    base_forecast = pd.Series(index= test_dates,name= "mean_forecast")

    for day in daily_test_dates:
        training_days = pd.date_range(start=pd.Timestamp('2017-01-01'),end=day + dt.timedelta(-1),freq='D')
        holidays = EnglandAndWalesHolidayCalendar().holidays(start=pd.Timestamp('2017-01-01'),end=day + dt.timedelta(-1))
        training_days = training_days.symmetric_difference(holidays)
        df = pd.DataFrame()

        if day in holidays:
            #mean holiday
            for holiday in holidays:
                selected_day = serie[serie.index.dayofyear == holiday.dayofyear].values
                df[str(holiday)] = selected_day
            df_mean = df.mean(axis=1).values.squeeze()
            base_forecast[base_forecast.index.dayofyear == day.dayofyear] = df_mean

        else:
            #mean weekday
            training_days = training_days[training_days.weekday == day.weekday()]
            for training_day in training_days:
                selected_day = serie[serie.index.dayofyear == training_day.dayofyear].values
                df[str(training_day)] = selected_day
            df_mean = df.mean(axis=1).values.squeeze()
            base_forecast[base_forecast.index.dayofyear == day.dayofyear] = df_mean

    return base_forecast

def histogram(training_serie: pd.Series):
    loads = training_serie.values
    # ax = figure_layout(figsize=(10,8),titel="Discretized distribution",xlabel="Load [kWh]",ylabel="Probability [-]")
    # ax.grid(axis='y', alpha=0.75)
    # ax.grid(axis='x', alpha=0.75)
    # plt.clf()
    loads = loads[~np.isnan(loads)]
    n, bins = np.histogram(a=loads, bins='fd',density=False)

    return n,bins,loads

from os import devnull
from sys import __stdout__,stdout
# Disable
def blockPrint():
    stdout = open(devnull, 'w')

# Restore
def enablePrint():
    stdout = __stdout__

# need to select the time of the day and day type
def MAPE_estimator(test_dates: pd.DatetimeIndex,serie: pd.Series):
    # The assumption is made that the test_dates are one continuous sequence of days
    daily_test_dates = pd.date_range(start=test_dates[0],end=test_dates[-1],freq='D')
    base_forecast = pd.Series(index= test_dates,name= 'MAPE_forecast')

    for day in daily_test_dates:
        training_days = pd.date_range(start=pd.Timestamp('2017-01-01'),end=day + dt.timedelta(-1),freq='D')
        holidays = EnglandAndWalesHolidayCalendar().holidays(start=pd.Timestamp('2017-01-01'),end=day + dt.timedelta(-1))
        training_days = training_days.symmetric_difference(holidays)

        if day in holidays:
            sundays = training_days[training_days.weekday == 6]
            training_days = sundays.union(holidays)
            forecast = MAPE_optimization(training_days,serie) # numpy array
            base_forecast[base_forecast.index.dayofyear == day.dayofyear] = forecast

        else:
            training_days = training_days[training_days.weekday == day.weekday()]
            forecast = MAPE_optimization(training_days,serie) # numpy array
            base_forecast[base_forecast.index.dayofyear == day.dayofyear] = forecast

    return base_forecast

def MAPE_optimization(training_days:pd.DatetimeIndex,serie:pd.Series,time_steps:int=48):
    stock = np.zeros(time_steps)
    training_hours = pd.date_range(start='00:00:00',periods=time_steps,freq='30min')
    training_30min = pd.date_range(start=training_days[0],periods=time_steps,freq='30min')
    for training_day in training_days[1:]:
        new_day = pd.date_range(start=training_day,periods=time_steps,freq='30min')
        training_30min = training_30min.union(new_day)

    # Select the hours for which an individual forecast should be made
    for index in np.arange(0,time_steps):
        temp = training_30min[training_30min.hour == training_hours[index].hour]
        data = temp[temp.minute == training_hours[index].minute]
        n,bins,loads = histogram(serie[data])
        probability = n/sum(n)
        ########################
        # axis = figure_layout(figsize=(18, 12), titel= "Histogram", xlabel= "Consumption [kWh]", ylabel= "Counts [-]", grid=False)
        # axis.hist(loads,bins=bins,rwidth=0.95)
        # fname = "D:\Onedrive\Leuven\Final project\Results\Forecasting\histogram"+str(index)+".png"
        # plt.savefig(fname, dpi=300, facecolor='w', edgecolor='w', orientation='portrait', format=None,
        #             transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)
        ########################
        discretized_values = []
        for j in np.arange(0,len(bins)-1):
            if probability[j] != 0:
                discretized_values.append((bins[j] + bins[j+1])/2)
        # remove zero probabilities
        probability = probability[probability != 0]
        discretized_values = np.array(discretized_values)
        p_hat = discretized_values.mean()
        if len(probability) != len(discretized_values) or np.around(sum(probability),2) != 1:
            print(sum(probability))
            raise Exception("The lenghts are not equal or prob is not one.")

        # solve the optimization problem
        opti = ca.Opti()
        p = opti.variable()
        obj = 0
        for j in np.arange(0,len(discretized_values)):
            pi = discretized_values[j]
            if pi != 0:
                ei =  probability[j]*((p-pi)/pi)
                Li = opti.variable()
                obj = obj + Li
                opti.subject_to(opti.bounded(-Li, ei, Li))
        opti.minimize(obj)
        opti.subject_to(p>=0)
        opti.set_initial(p,p_hat)
        opti.solver('ipopt')
        blockPrint()
        sol = opti.solve()
        enablePrint()
        stock[index] = sol.value(p)

    return stock

# Evaluation metrics
def RMSE(forecast: pd.Series,real: pd.Series):
    return np.sqrt(sum((forecast.values - real.values)**2)/len(forecast.values))

def MSE(forecast: pd.Series,real: pd.Series):
    return sum((forecast.values - real.values)**2)/len(forecast.values)

def NRMSE(forecast: pd.Series,real: pd.Series):
    ymax = real.max()
    ymin = real.min()
    RMSE_value = RMSE(forecast,real)
    return RMSE_value/(ymax-ymin)

def MAE(forecast: pd.Series,real: pd.Series):
    return sum(abs(forecast.values - real.values))/len(forecast.values)

def MAPE(forecast: pd.Series,real: pd.Series):
    return sum(abs((forecast.values - real.values)/real.values))/len(forecast.values)

def Switcher(name: str,forecast,real_values):
    switcher = {
                "MSE": MSE(forecast,real_values),
                "RMSE": RMSE(forecast,real_values),
                "NRMSE": NRMSE(forecast, real_values),
                "MAE": MAE(forecast, real_values),
                "MAPE": MAPE(forecast, real_values),
            }
    return switcher.get(name)

def autolabel(rects,ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%s' % np.around(height,4),
                ha='center', va='bottom', fontweight='bold',fontsize=16)






