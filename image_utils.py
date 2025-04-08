import cv2
import os
from tqdm import tqdm
import imghdr
import ast
import numpy as np
import cv2 as cv
import pandas as pd
import fiftyone as fo

from feature_matching import homographic_trsf as ht

def check_validity(coords, img_w, img_h):
    x, y, w, h = coords
    if w <= 0 or h <= 0:
        return None
    area = w * h
    img_ul = [max(min(x, img_w), 0), max(min(y, img_h), 0)]
    img_lr = [max(min(x + w, img_w), 0), max(min(y + h, img_h), 0)]
    new_w, new_h = (img_lr[0] - img_ul[0]), (img_lr[1] - img_ul[1])
    img_area = new_w * new_h
    return [img_ul[0], img_ul[1], new_w, new_h] if img_area / area > 0.5 else None


def crop_image(image, coords):
    x, y, h, w = coords
    return image[y:y + h, x:x + w]


def warp_img(img_input, ref_shape, M):
    return cv2.warpPerspective(img_input, M, (ref_shape[1], ref_shape[0]))


def plot_vignette(image, coords, color=(255, 0, 0)):
    x, y, w, h = coords
    thickness = 2
    image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)
    return image

def reverse_trsf(row, inv_h):
    x, y, w, h = row['x'], row['y'], row['w'], row['h']
    ori_ul = ht(inv_h, [x, y])
    ori_lr = ht(inv_h, [x + w, y + h])
    w = ori_lr[0] - ori_ul[0]
    h = ori_lr[1] - ori_ul[1]
    return [int(ori_ul[0]), int(ori_ul[1]), int(w), int(h)]


def crop_all_images(images_path, polyp_ref, h_matrixs, output_path, dataset):
    """
    Crop all images in image path according to a polyp ref file. Stores the vignettes and the ori_porsition file in the output_path
    :param images_path:
    :param polyp_ref:
    :param h_matrixs: homographic matrixs
    :param output_path:
    :return:
    """
    original_positions = []
    samples = []
    for image_id, file in tqdm(enumerate(sorted(os.listdir(images_path)))):  # for each image in the directory
        jpg_path = os.path.join(images_path, file)
        if not os.path.isfile(jpg_path) or imghdr.what(jpg_path) != "jpeg":
            continue
        h_matrix_img = h_matrixs.loc[h_matrixs['filename'] == file]
        if len(h_matrix_img)==0:
            print(f"Missing h matrix, ignoring: {file}")
            continue
        M = np.array(ast.literal_eval(h_matrix_img['matrix'].to_list()[0]))
        if len(M) == 0:
            M = np.eye(3)  # No transformation

        img_input = cv.imread(jpg_path, cv.IMREAD_UNCHANGED)
        img_w, img_h = img_input.shape[1], img_input.shape[0]
        val, inv_h = cv2.invert(M)

        for index, row in polyp_ref.iterrows():
            coords = reverse_trsf(row, inv_h)
            valid_coords = check_validity(coords, img_w, img_h)
            if valid_coords is not None:
                cropped_img = crop_image(img_input, valid_coords)
                original_positions.append([file, index, valid_coords[0], valid_coords[1], valid_coords[2], valid_coords[3], img_w, img_h])
                exp_path = os.path.join(output_path, f"{str(index)}_" + file)
                cv.imwrite(exp_path, cropped_img)

                sample = fo.Sample(filepath=exp_path)
                sample["polyp_ref_index"] = index
                sample["image_id"] = image_id
                sample["original_image"] = file
                sample["x"] = valid_coords[0]
                sample["y"] = valid_coords[1]
                sample["w"] = valid_coords[2]
                sample["h"] = valid_coords[3]
                samples.append(sample)
            else:
                print("Invalid coords")

    dataset.add_samples(samples)


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
    images_path = r'W:\images\MARLEY_2021'
    output_path = r'D:\ARDECO\MARLEY\temp'
    output_h_matrix_path = os.path.join(r"D:\ARDECO\MARLEY", 'h_matrixs.txt')
    polyp_ref_path = os.path.join(r"D:\ARDECO\MARLEY", 'polyp_ref_updated.csv')

    polyp_ref = pd.read_csv(polyp_ref_path)
    h_matrixs = pd.read_csv(output_h_matrix_path)

    crop_all_images(images_path, polyp_ref, h_matrixs, output_path)

