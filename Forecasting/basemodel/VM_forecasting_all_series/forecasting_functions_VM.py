from VM_forecasting_all_series.Test_basemodel_functions_VM import figure_layout_VM
from azureml.core import Run
import pandas as pd
import matplotlib.pyplot as plt

def get_all_days_of_year(serie: pd.Series)->set:
    collection = set()
    for date in serie.index:
        collection.add(date.dayofyear)
    return collection

def show_forecast(all_predictions, all_references, ID: str, day_int: str, run:Run, name):
    axis,fig = figure_layout_VM(figsize=(10,8),titel="",xlabel="date",ylabel="kWh", dpi= 300)
    labels = ["True Future", "Model Prediction"]

    all_references.plot(ax=axis, lw= 3.0, grid= True)
    all_predictions.plot(ax=axis, lw= 3.0, grid= True)
    axis.legend(labels)
    fig = plt.gcf()
    fname = name+"_"+"ID" + ID + "_Day" + day_int
    run.log_image(name= fname + "_pred.png", plot=fig,
                  description="ref vs pred")
    return run


def show_all_forecasts(all_predictions, all_references,ID: str, run:Run, name:str):
    collection = get_all_days_of_year(all_predictions)
    for day_int in collection:
        predictions = all_predictions[all_predictions.index.dayofyear == day_int]
        references = all_references[all_references.index.dayofyear == day_int]
        run = show_forecast(predictions, references, ID, str(day_int), run, name)

    return run
