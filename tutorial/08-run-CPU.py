# 07-run-basemodel.py
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Dataset

if __name__ == "__main__":
    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name='Check_cores_CPU')

    config = ScriptRunConfig(
        source_directory='./src',
        script='CPU.py',
        compute_target='cpucluster'
    )

    run = experiment.submit(config)
    aml_url = run.get_portal_url()
    print("Submitted to compute cluster. Click link below")
    print("")
    print(aml_url)
    print(50*"*")
    run.wait_for_completion(show_output=True)
