# tutorial/01-create-workspace.py
from azureml.core import Workspace

ws = Workspace.create(name='VM_thesiswrksp', # provide a name for your workspace
                      subscription_id='94bcbef3-faf7-461f-9e99-e6365f006a56', # provide your subscription ID
                      resource_group='VM_thesis', # provide a resource group name
                      create_resource_group=True,
                      location='westeurope') # For example: 'westeurope' or 'eastus2' or 'westus2' or 'southeastasia'.

# write out the workspace details to a configuration file: .azureml/config.json
ws.write_config(path='.azureml')