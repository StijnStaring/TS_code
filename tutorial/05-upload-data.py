# 05-upload-data.py
from azureml.core import Workspace
ws = Workspace.from_config()
datastore = ws.get_default_datastore()
datastore.upload(src_dir= "D:\AI_time_series_repos\TS_code\\upload_data",
                 target_path='datasets/all_series',
                 overwrite=True)