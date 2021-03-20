"""
This file is testing the different base models.
The test set used are the 31 days of December of the 261 time-series with measurements of the full year 2017.
Forecasts are only included in the evaluation metrics when all the different models output a forecast
different then nan.

Evaluation metrics used: MSE, RMSE, NRMSE, MAE, MAPE

"""

import datetime as dt
import pandas as pd # pandas
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





