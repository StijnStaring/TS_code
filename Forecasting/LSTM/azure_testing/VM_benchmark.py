import time as t
imp
# from azureml.core import Experiment, Run, Workspace
# import azureml.core
print("SDK version: %s.\n" % azureml.core.VERSION)


print("#"*250)
print("My name is Stijn Staring and my age is 23.")

filename = str(t.time())
file = open(filename, mode= "a")
file.write("Stijn loves spaghetti.\nHe also loves his family.")
file.close()
