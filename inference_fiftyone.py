import feature_matching as fm
import utils_polyps as up
import image_utils as iu
import fiftyone_utils as fou

import fiftyone as fo
from fastai.vision.all import *
import torch.cuda as tc
import os
import json
import pandas as pd
import configparser


def export_dataset(dataset, output_path):
    dataset.export(
        export_dir=output_path,
        dataset_type=fo.types.CSVDataset,
        export_media = False,
        fields=["polyp_ref_index", "image_id", "predictions.label", "predictions.confidence", "original_image"],
    )

def fiftyone_inference(config, volume, model_path, debug = True):
    
    if type(volume) == list:  # Recursive call
        for v in volume:
            fiftyone_inference(config, v, model_path, debug = False)
        return 1

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

    fou.delete_all_datasets()
    dataset = fo.Dataset(f"{volume}_dataset")

    iu.crop_all_images(images_path, polyp_ref, h_matrixs, vign_path, dataset)

    learner = load_learner(model_path, cpu=1)

    fou.do_inference(learner, dataset)

    inference_path = os.path.join(output_path, 'predictions')
    os.makedirs(inference_path) if not os.path.exists(inference_path) else None
    export_dataset(dataset, inference_path)

    return dataset if debug else 1


    