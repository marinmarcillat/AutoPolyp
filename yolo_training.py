import fiftyone as fo
import fiftyone.utils.random as four
from ultralytics import YOLO
import os

DATASET_DIR = r"E:\training_Elena\data\ATM"
export_dir = r"D:\tests\yolo_ardeco\data"

model_path = r""

dataset = fo.Dataset.from_dir(
    DATASET_DIR,
    fo.types.ImageClassificationDirectoryTree,
    name="yolo_atm_dataset",
)

four.random_split(dataset, {"train": 0.8, "test": 0.1, "val": 0.1})
train_view = dataset.match_tags("train")
test_view = dataset.match_tags("test")
val_view = dataset.match_tags("val")

train_dir = os.path.join(export_dir, "train")
test_dir = os.path.join(export_dir, "test")
val_dir = os.path.join(export_dir, "val")

train_view.export(
   export_dir=train_dir,
   dataset_type=fo.types.ImageClassificationDirectoryTree,
)
test_view.export(
   export_dir=test_dir,
   dataset_type=fo.types.ImageClassificationDirectoryTree,
)
val_view.export(
    export_dir=val_dir,
    dataset_type=fo.types.ImageClassificationDirectoryTree,
)

model = YOLO(model_path)  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data=export_dir, epochs=100, imgsz=64, patience = 20, batch=64)