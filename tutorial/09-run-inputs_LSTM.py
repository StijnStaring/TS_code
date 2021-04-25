# 07-run-basemodel.py
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Dataset

if __name__ == "__main__":
    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name='Calculate_LSTM_inputs')
    datastore = ws.get_default_datastore()
    dataset = Dataset.File.from_files(path=(datastore, 'datasets/three_series'))

    config = ScriptRunConfig(
        source_directory="D:\AI_time_series_repos\TS_code\Forecasting\LSTM\VM_calculating_inputs_LSTM\src",
        script='LSTM_inputs_VM.py',
        compute_target='gpu-cluster',
        arguments=[
            '--data_path', dataset.as_named_input('input').as_mount(),
        ],
    )
    # set up environment
    env = Environment.from_conda_specification(
        name='forecasting-env',
        file_path='./.azureml/forecasting-env.yml'
    )
    config.run_config.environment = env
    # the first time that the run is submitted -> env is automatically registered in Azure
    run = experiment.submit(config)
    run.tag("Description", "Compute LSTM inputs.")
    aml_url = run.get_portal_url()
    print("Submitted to compute cluster. Click link below")
    print("")
    print(aml_url)
    print(50*"*")
    run.wait_for_completion(show_output=True)

    # metrics = run.get_metrics() # no metrics are assigned!

    # downloading the files
    import os
    project_folder = "D:\AI_time_series_repos\TS_code\Forecasting\LSTM\VM_calculating_inputs_LSTM"
    outputs_path = os.path.join(project_folder, "output_arrays")
    os.makedirs(outputs_path, exist_ok=True)

    for filename in run.get_file_names():
        print("filename: %s."%filename)
        if filename.startswith('outputs'):
            print("Downloading " + filename)
            run.download_file(filename, output_file_path=outputs_path)
        # _, file_extension = os.path.splitext(filename)
        # if file_extension == ".npy":
        # elif filename.startswith("output_file"):
        #     print("Downloading " + filename)
        #     run.download_file(filename, output_file_path=outputs_path)