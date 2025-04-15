import feature_matching as fm
import utils_polyps as up
import image_utils as iu
import fiftyone_utils as fou
from biigle import Api
from annotation_conversion_toolbox import biigle_dataset

import fiftyone as fo
from fastai.vision.all import *
import torch.cuda as tc
import os
import json
import pandas as pd
import configparser
from shutil import copy2
from ultralytics import YOLO


def export_dataset(dataset, output_path):
    dataset.export(
        export_dir=output_path,
        dataset_type=fo.types.CSVDataset,
        export_media = False,
        fields=["polyp_ref_index", "image_id", "classifications.classifications.label", "classifications.classifications.confidence", "original_image"],
    )

def fiftyone_inference(config, volume, model_path, model = "resnet", debug = True, biigle_export = False):
    
    if type(volume) == list:  # Recursive call
        return [
            fiftyone_inference(config, v, model_path, debug=debug, model = model)
            for v in volume
        ]


    print(f"Preparing inference for volume {volume}")

    print("Config : ")
    print(dict(config.items(volume)))

    try:
        img_ref_polyps = json.loads(config.get(volume, "img_ref_polyps"))
        labels_name_ref = json.loads(config.get(volume, "labels_name_ref"))
        labels_name_classification = json.loads(config.get(volume, "labels_name_classification"))
        labels_name_all = labels_name_ref
        labels_name_all.extend(labels_name_classification)
        output_path = config.get(volume, "output_path")
        report_path = config.get(volume, "report_path")
        images_path = config[volume]['images_path']
        img_ref_pos = config[volume]['img_ref_pos']
    except configparser.NoOptionError:
        print("missing required fields in config file")
        print("required fields are: img_ref_polyps, labels_name_ref, labels_name_classification, output_path, report_path, images_path, img_ref_pos")
        return 0

    if biigle_export:
        try:
            biigle_project_dir = config.get('DEFAULT', 'biigle_project_dir')
            project_id = config.get('DEFAULT', 'biigle_project_id')
            label_tree_id = config.get('DEFAULT', 'label_tree_id')
            nb_sampling = config.getint('DEFAULT', 'nb_sampling')
            biigle_api_token = config.get('DEFAULT', 'token')
            biigle_api_email = config.get('DEFAULT', 'email')
            api = Api(biigle_api_email, biigle_api_token)
        except configparser.NoOptionError:
            print("missing required fields in config file, default section, for biigle export")
            print(
                "required fields are: biigle_project_dir, biigle_project_id, biigle_project_id, nb_sampling, token, email")
            return 0

    vign_path = os.path.join(output_path, 'vign')


    for path in [vign_path]:  # Create dirs if they do not exist
        if not os.path.exists(path):
            os.makedirs(path)

    h_matrix_path = os.path.join(output_path, 'h_matrixs.txt')
    polyp_ref_path = os.path.join(output_path, 'polyp_ref.csv')

    annotations = pd.read_csv(report_path)

    print("Get homography matrixs...")
    if not os.path.exists(h_matrix_path):
        print("Calculating homography matrixs...")
        fm.get_h_matrixs(images_path, img_ref_pos, h_matrix_path)

    h_matrixs = pd.read_csv(h_matrix_path)  # Charger les matrices homographiques déjà calculées

    print("Get reference polyps...")
    if not os.path.exists(polyp_ref_path):
        print("Convert polyps coordinates...")
        polyps_positions = up.get_polyps_coords(images_path, annotations, h_matrixs,
                                                labels_name_all,
                                                output_path)
        polyp_ref = up.get_ref_polyps(polyps_positions, img_ref_polyps, labels_name_ref,
                                      output_path, True)
    else:
        polyp_ref = pd.read_csv(polyp_ref_path)

    if len(polyp_ref) <= 3:
        print(f"No or too few ({len(polyp_ref)}) ref polyps found")
        return 0

    dataset = fo.Dataset(f"{volume}_dataset")

    iu.crop_all_images(images_path, polyp_ref, h_matrixs, vign_path, dataset)

    if model == "resnet":
        learner = load_learner(model_path, cpu=1)
        fou.do_inference(learner, dataset)
    if model == "yolo":
        yolo_model = YOLO(model_path)
        dataset.apply_model(yolo_model, label_field="classification")
        for sample in dataset.iter_samples(progress=True):
            sample["classifications"] = fo.Classifications(classifications=[sample["classification"]])
            sample.save()

    inference_path = os.path.join(output_path, 'predictions')
    None if os.path.exists(inference_path) else os.makedirs(inference_path)
    export_dataset(dataset, inference_path)

    if biigle_export:
        to_check_dataset = dataset.take(nb_sampling)

        volume_dir = os.path.join(biigle_project_dir, volume)
        if not os.path.exists(volume_dir):
            os.makedirs(volume_dir)

        imgs = []
        for sample in to_check_dataset.iter_samples(progress=True):
            fp = sample["filepath"]
            file_name = os.path.basename(fp)
            dest = os.path.join(volume_dir, file_name)
            if not os.path.exists(dest):
                copy2(fp, dest)
            imgs.append(file_name)

        payload = {
            "name": f"{volume}_inference",
            "url": f"local://{volume_dir[3:]}",
            "media_type": "image",
            "files": imgs,
        }

        result = api.post(f"projects/{project_id}/volumes", json=payload)

        exporter = biigle_dataset.BiigleDatasetExporter(api=api,
                                                        volume_id=result.json()["id"],
                                                        label_tree_id=label_tree_id,
                                                        biigle_image_dir=volume_dir
                                                        )

        fields = to_check_dataset.get_field_schema()
        if "classifications" in fields:
            to_check_dataset.export(dataset_exporter=exporter, label_field="classifications")

    return dataset if debug else 1


    