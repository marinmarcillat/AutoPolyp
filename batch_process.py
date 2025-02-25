import os
import pandas as pd
import configparser
import json

import utils_polyps as up
import feature_matching as fm
import image_utils as iu
import inference
from biigle import Api
from tqdm import tqdm


def volume_inference(config, volume):
    api = Api(config['DEFAULT']['email'], config['DEFAULT']['token'])

    model_path = config['DEFAULT']['model_path']

    print("Config : ")
    print(dict(config.items(volume)))

    img_ref_polyps = json.loads(config.get(volume, "img_ref_polyps"))
    labels_name_ref = json.loads(config.get(volume, "labels_name_ref"))
    labels_name_classification = json.loads(config.get(volume, "labels_name_classification"))
    labels_name_all = json.loads(config.get(volume, "labels_name_ref"))
    labels_name_all.extend(labels_name_classification)

    train_path = os.path.join(config[volume]['output_path'], 'train')
    temp_path = os.path.join(config[volume]['output_path'], 'temp')
    vign_path = os.path.join(config[volume]['output_path'], 'vign')

    for path in [train_path, temp_path, vign_path]:  # Create dirs if they do not exist
        if not os.path.exists(path):
            os.makedirs(path)

    h_matrix_path = os.path.join(config[volume]['output_path'], 'h_matrixs.txt')
    polyp_ref_path = os.path.join(config[volume]['output_path'], 'polyp_ref.csv')

    annotations = pd.read_csv(config[volume]['report_path'])


    print("Get homography matrixs...")
    if not os.path.exists(h_matrix_path):
        fm.get_h_matrixs(config[volume]['images_path'], config[volume]['img_ref_pos'], h_matrix_path)

    h_matrixs = pd.read_csv(h_matrix_path)  # Charger les matrices homographiques déjà calculées

    print("Get reference polyps...")
    if not os.path.exists(polyp_ref_path):
        print("Convert polyps coordinates...")
        polyps_positions = up.get_polyps_coords(config[volume]['images_path'], annotations, h_matrixs, labels_name_all,
                                                config[volume]['output_path'])
        polyp_ref = up.get_ref_polyps(polyps_positions, img_ref_polyps, labels_name_ref, config[volume]['output_path'], True)
    else:
        polyp_ref = pd.read_csv(polyp_ref_path)

    if len(polyp_ref) <= 3:
        print("No or too few ref polyps found")
        return 0

    iu.crop_all_images(config[volume]['images_path'], polyp_ref, h_matrixs, vign_path)

    inference.model_inference(model_path, vign_path, config[volume]['output_path'], api,
                              config[volume]['label_tree_id'], config[volume]['volume_id'])

    del api
    return 1


if __name__ == "__main__":
    config_path = "D:\ARDECO\config.ini"
    volumes = ['A3S3']

    config = configparser.ConfigParser()
    config.read(config_path)

    for volume in tqdm(volumes):
        volume_inference(config, volume)
