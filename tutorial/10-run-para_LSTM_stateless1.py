from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Dataset
from time import time

if __name__ == "__main__":
    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name='Para_Stateless1_phase2')
    datastore = ws.get_default_datastore()
    dataset = Dataset.File.from_files(path=(datastore, 'datasets/three_series'))

    config = ScriptRunConfig(
        source_directory="D:\AI_time_series_repos\TS_code\Forecasting\LSTM\Hypersearch\VM_parametersearch_stateless1\src",
        script='parameterSearch_VM.py',
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
    run.tag("Description", "Try different parameters.")
    aml_url = run.get_portal_url()
    print("Submitted to compute cluster. Click link below")
    print("")
    print(aml_url)
    print(50*"*")
    run.wait_for_completion(show_output=True)

    # metrics = run.get_metrics() # no metrics are assigned!

    # downloading the files
    import os
    project_folder = "D:\AI_time_series_repos\TS_code\Forecasting\LSTM\Hypersearch\VM_parametersearch_stateless1"
    name = str(time())
    outputs_path = os.path.join(project_folder, name)
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