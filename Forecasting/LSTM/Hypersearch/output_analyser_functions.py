import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from forecasting_functions import figure_layout
from os import path

def get_ID_numbers(name:str, number, DF: pd.DataFrame):
    newSerie = DF.loc[[name],:].squeeze()
    newSerie = pd.to_numeric(newSerie)
    settings = pd.to_numeric(DF.columns[newSerie.values == number])
    return settings

def get_all_error_values(setting_ID,metric:str, DF: pd.DataFrame):
    collection = np.array([])
    for id in setting_ID:
        col = DF[str(id)+"_"+metric].values
        # print(col)
        collection = np.concatenate((collection,col), axis=None)
    return collection

def create_scatter_plot(name:str,numbers:list,metric:str,DF_log:pd.DataFrame,DF_error:pd.DataFrame, name_serie: str, colors:list = ['r','b','g']):
    unit = None
    if metric == "MAE" or metric == "RMSE":
        unit = "[kWh]"
    axis = figure_layout(titel= metric+" "+ name,ylabel="Error " + unit, xlabel="Number of run")
    for i in range(len(numbers)):
        number = numbers[i]
        color = colors[i]
        settings_try = get_ID_numbers(name,number,DF_log)
        out = get_all_error_values(settings_try,metric,DF_error)
        print(len(out),end="")

        axis.scatter(np.arange(1,len(out)+1),out,c=color,label=number)
    plt.legend(loc="upper right")
    out_name: str = name_serie+"_"+metric+"_"+ name+".png"
    # using output_path_pictures as global variable
    plt.savefig(path.join(output_path_pictures,out_name), dpi=300, facecolor='w', edgecolor='w', orientation='portrait', format=None,
                    transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)

def get_the_best_settings_MAE(DF_error: pd.DataFrame):
    """
    :param DF_error: dataFrame with all the errors collected during the parameter search
    :return: settings with the smallest MAE error
    """
    collection_errors = []
    for col_name in DF_error.columns:
        if col_name[-3:] == "MAE":
            collection_errors.append(DF_error[col_name].sum())
    found_min = min(collection_errors)
    found_id = collection_errors.index(found_min) + 1
    return found_id

def get_improvement(name:str,numbers:list,metric:str,DF_log:pd.DataFrame,DF_error:pd.DataFrame):
    collection = {}
    for i in range(len(numbers)):
        number = numbers[i]
        settings = get_ID_numbers(name,number,DF_log)
        out = get_all_error_values(settings,metric,DF_error)
        collection[str(number)] = np.mean(out)
    key_biggest_value = max(collection, key= collection.get)
    biggest_value = collection[key_biggest_value]
    del collection[key_biggest_value]
    collection = {key: (1 - collection[key]/biggest_value)*100 for key in collection.keys()}
    print("The improvement when choosing a certain value: %s"%collection)

    return collection