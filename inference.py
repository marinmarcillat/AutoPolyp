from inference_fiftyone import fiftyone_inference
from fiftyone_utils import delete_all_datasets
import configparser
import fiftyone as fo


biigle_export = True

config_path = r"D:\ARDECO_Elena\config\config_ATM -2.ini"
#volume = ['A3S1', 'A3S2', 'A3S3', 'A3S4', 'A3S6', 'A3S7', 'A3S8', 'A3S10', 'A3S11', 'A3S12', 'A3S14-1', 'A3S14-2', 'A3S15', 'A3S16']
#OR
volume = 'A2S9'
#volume = ['A3S3', 'A2S3', 'A1S3', 'A3S2', 'A2S2', 'A1S2', 'A3S1', 'A2S1', 'A1S1']

model = "yolo"  # or model = "resnet"
model_path = r"D:\ARDECO_Elena\model\ATM\model_export.pkl"

config = configparser.ConfigParser()
config.read(config_path)

delete_all_datasets()
dataset = fiftyone_inference(config, volume, model_path, biigle_export = biigle_export, model = model)

if type(dataset) == fo.Dataset:
    session = fo.launch_app(dataset)
    session.wait()

print("ok")