# 07-run-basemodel.py
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Dataset

if __name__ == "__main__":
    ws = Workspace.from_config()
    datastore = ws.get_default_datastore()
    dataset = Dataset.File.from_files(path=(datastore, 'datasets/all_series'))

    experiment = Experiment(workspace=ws, name='Run1_base_models_three')

    config = ScriptRunConfig(
        source_directory='D:\AI_time_series_repos\TS_code\Forecasting\\basemodel\src',
        script='Test_basemodel_VM.py',
        compute_target='cpucluster',
        arguments=[
            '--data_path', dataset.as_named_input('input').as_mount(),
        ],
    )
    # set up environment
    env = Environment.from_conda_specification(
        name='basemodel-env',
        file_path='./.azureml/basemodel-env.yml'
    )
    config.run_config.environment = env
    # the first time that the run is submitted -> env is automatically registered in Azure
    run = experiment.submit(config)
    run.tag("Description", "Compute the performance of the base models.")
    aml_url = run.get_portal_url()
    print("Submitted to compute cluster. Click link below")
    print("")
    print(aml_url)
    print(50*"*")
    run.wait_for_completion(show_output=True)

    # metrics = run.get_metrics() # no metrics are assigned!

    # downloading the files
    # import os
    # project_folder = "D:\AI_time_series_repos\TS_code\Forecasting\\basemodel"
    # outputs_path = os.path.join(project_folder, "outputs")
    # os.makedirs(outputs_path, exist_ok=True)
    #
    # for filename in run.get_file_names():
    #     print("filename: %s."%filename)
    #     if filename.startswith('csv_files'):
    #         print("Downloading " + filename)
    #         run.download_file(filename, output_file_path=outputs_path)