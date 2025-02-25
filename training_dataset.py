import os
from tqdm import tqdm
import imghdr
import cv2
import image_utils as iu
import utils_polyps as up
import pandas as pd


def crop_all_images_training(images_path, annotations, labels_name, train_path):
    for file in tqdm(sorted(os.listdir(images_path))):  # for each image in the directory
        jpg_path = os.path.join(images_path, file)
        if os.path.isfile(jpg_path) and imghdr.what(jpg_path) == "jpeg":
            ann_img = annotations.loc[annotations['filename'] == file].loc[
                annotations['shape_name'].isin(["Circle", "Rectangle"])].loc[
                annotations['label_name'].isin(labels_name)]
            img_input = cv2.imread(jpg_path, cv2.IMREAD_UNCHANGED)
            img_w, img_h = img_input.shape[1], img_input.shape[0]

            for index, row in ann_img.iterrows():
                x, y, w, h = up.extract_annotation_coordinates(row)
                coords = [int(x), int(y), int(w), int(h)]
                valid_coords = iu.check_validity(coords, img_w, img_h)
                if valid_coords is not None:
                    cropped_img = iu.crop_image(img_input, valid_coords)

                    label = row['label_name']
                    label_dir = os.path.join(train_path, label)
                    if not os.path.exists(label_dir):
                        os.makedirs(label_dir)

                    exp_path = os.path.join(label_dir, f"{str(index)}_{file}")
                    cv2.imwrite(exp_path, cropped_img)


if __name__ == "__main__":
    export_dir = r""
    labels_name = ["", ""]

    training_volumes = {
        "volume_name": {
            "image_path": r"image_path",
            "report_path": r"report_path"},

    }
    for config in training_volumes.values():
        annotations = pd.read_csv(config["report_path"])
        crop_all_images_training(config["image_path"], annotations, labels_name, export_dir)
