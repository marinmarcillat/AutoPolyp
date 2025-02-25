from math import dist
import os
import cv2 as cv
import math
import pandas as pd
from tqdm import tqdm
import imghdr, ast
import numpy as np
from scipy.spatial import distance
import utils_pascalVOC
import export_to_biigle

import image_utils as iu
from feature_matching import homographic_trsf as ht


def extract_annotation_coordinates(row):
    if row['shape_name'] == "Circle":
        coord = ast.literal_eval(row['points'])[:2]
        r = ast.literal_eval(row['points'])[2]
        ul = [coord[0] - r, coord[1] - r]
        lr = [coord[0] + r, coord[1] + r]

    if row['shape_name'] == "Rectangle":
        coords = ast.literal_eval(row['points'])
        minx = min(coords[0], coords[2], coords[4], coords[6])
        maxx = max(coords[0], coords[2], coords[4], coords[6])
        miny = min(coords[1], coords[3], coords[5], coords[7])
        maxy = max(coords[1], coords[3], coords[5], coords[7])
        ul = [minx, miny]
        lr = [maxx, maxy]
    x = ul[0]
    w = lr[0] - ul[0]
    y = ul[1]
    h = lr[1] - ul[1]
    return x, y, w, h

def search_collision(new_polyp, db):
    if len(db) == 0:
        return -1
    for polyp_id in range(len(db)):
        d = dist(new_polyp, db[polyp_id][0])
        if d < db[polyp_id][1]:
            return polyp_id
    return -1


def closest_node(node, nodes):
    return distance.cdist(node, nodes).argmin()


def get_polyps_coords(images_path, annotations, h_matrixs, labels_name, output_path):

    polyp_coords = []
    img_list = annotations.filename.unique().tolist()
    for file in tqdm(sorted(os.listdir(images_path))):  # for each image in the directory
        jpg_path = os.path.join(images_path, file)
        if (
            file in img_list
            and os.path.isfile(jpg_path)
            and imghdr.what(jpg_path) == "jpeg"
        ):
            ann_img = annotations.loc[annotations['filename'] == file].loc[annotations['shape_name'].isin(["Circle", "Rectangle"])].loc[annotations['label_name'].isin(labels_name)]
            if len(ann_img) != 0:
                h_matrix_img = h_matrixs.loc[h_matrixs['filename'] == file]
                M = np.array(ast.literal_eval(h_matrix_img['matrix'].to_list()[0]))
                if len(M) == 0:
                    M = np.eye(3)

                for index, ann in ann_img.iterrows():
                    x, y, w, h = extract_annotation_coordinates(ann)
                    ul = [x, y]
                    lr = [x + w, y + h]

                    ul_trsf = ht(M, ul)
                    lr_trsf = ht(M, lr)
                    x = ul_trsf[0]
                    w = lr_trsf[0] - ul_trsf[0]
                    y = ul_trsf[1]
                    h = lr_trsf[1] - ul_trsf[1]
                    polyp_coords.append([file, ann['label_name'], x, y, w, h, ann['shape_name']])

    polyp_coords_pd = pd.DataFrame(polyp_coords, columns=['filename', 'label', 'x', 'y', 'w', 'h', 'shape_name'])
    polyp_coords_path = os.path.join(output_path, 'polyps_coords.csv')
    polyp_coords_pd.to_csv(polyp_coords_path, index=False)
    return polyp_coords_pd

def get_ref_polyps(polyps_positions, images_ref, label_names, output_path, round_if_empty=False):
    polyps_ref = []
    exp_polyps_ref = []
    references = polyps_positions[polyps_positions['filename'].isin(images_ref)][polyps_positions['label'].isin(label_names)]
    if len(references) == 0 and round_if_empty:
        references = polyps_positions[polyps_positions['filename'].isin(images_ref)][polyps_positions['shape_name'] == 'Circle']
    for index, ann_ref in tqdm(references.iterrows()):
        center = [ann_ref['x'] + ann_ref['w'] / 2, ann_ref['y'] + ann_ref['h'] / 2]
        radius = ann_ref['w'] / 2
        keep = 1
        for prev_polyp in polyps_ref:
            prev_center, prev_radius = prev_polyp[0], prev_polyp[1]
            if distance.euclidean(center, prev_center) < prev_radius:  # if within the radius of another annotation
                keep = 0  # Do not keep
        if keep:
            polyps_ref.append([center, radius])
            exp_polyps_ref.append([ann_ref['filename'], ann_ref['label'], ann_ref['x'], ann_ref['y'], ann_ref['w'], ann_ref['h']])

    polyp_ref = pd.DataFrame(exp_polyps_ref, columns=['filename', 'label', 'x', 'y', 'w', 'h'])
    polyp_ref_path = os.path.join(output_path, 'polyp_ref.csv')
    polyp_ref.to_csv(polyp_ref_path, index=False)
    return polyp_ref



def adjust_ref_polyps_other_annotations(polyps_ref, polyps_positions, output_path):

    polyps_ref = polyps_ref.values.tolist()
    for i in range(len(polyps_ref)):
        polyps_ref[i].append([])

    c = 0
    ref_pos = [[i[2] + i[4]/2, i[3] + i[5]/2] for i in polyps_ref]
    for index, ann in tqdm(polyps_positions.iterrows()):
        center = [[ann['x'] + ann['w'] / 2, ann['y'] + ann['h'] / 2]]
        closest_index = closest_node(center, ref_pos)
        dist = distance.euclidean(center[0], ref_pos[closest_index])
        if dist < 10:
            polyps_ref[closest_index][6].append([ann['filename'], ann['label'], ann['x'], ann['y'], ann['w'], ann['h']])
        else:
            c+=1
    print(f'{(c/len(polyps_positions))*100} % of annotations removed due to position inconsistencies')

    for polyp_id in range(len(polyps_ref)):
        mean_coords = [ann[2:6] for ann in polyps_ref[polyp_id][6]]
        mean_coords.append(polyps_ref[polyp_id][2:6])
        m_pd = pd.DataFrame(mean_coords)
        means_coords = list(m_pd.mean())
        polyps_ref[polyp_id][2:6] = means_coords
        polyps_ref[polyp_id].pop()


    polyps_ref_updt = pd.DataFrame(polyps_ref, columns=['filename', 'label', 'x', 'y', 'w', 'h'])
    polyps_ref_updt_path = os.path.join(output_path, 'polyp_ref_updated.csv')
    polyps_ref_updt.to_csv(polyps_ref_updt_path, index=False)
    return polyps_ref_updt


def draw_polyps_from_ref(images_path, freq, h_matrixs, polyp_ref, label_name, export_path, api, label_tree_id, volume_id):

    delete_first = False

    print("Creating label index for API")
    label_idx = export_to_biigle.create_label_index(api, label_tree_id, [label_name])
    print("Creating image index for API")
    images_idx = export_to_biigle.create_image_index(api, volume_id)
    print("Done !")

    if delete_first:
        print("Deleting previous annotations...")
        for id, row in tqdm(images_idx.iterrows()):
            annotations = api.get(f"images/{row['id']}/annotations").json()
            for ann in annotations:
                ann_id = ann['id']
                api.delete(f'image-annotations/{ann_id}')

    img_list = sorted(os.listdir(images_path))
    for file_id in tqdm(range(0, len(img_list), int(1/freq))):
        jpg_path = os.path.join(images_path, img_list[file_id])
        if os.path.isfile(jpg_path) and imghdr.what(jpg_path) == "jpeg":
            annotations_xy = []
            h_matrix_img = h_matrixs.loc[h_matrixs['filename'] == img_list[file_id]]
            M = np.array(ast.literal_eval(h_matrix_img['matrix'].to_list()[0]))
            if len(M) == 0:
                M = np.eye(3)
            val, inv_h = cv.invert(M)
            for index, row in polyp_ref.iterrows():
                x, y, w, h = iu.reverse_trsf(row, inv_h)
                annotations_xy.append([label_name, x, y, x + w, y + h, 1])

            # Save to pascalVOC file
            img = cv.imread(jpg_path, cv.IMREAD_UNCHANGED)
            height, width, deepth = img.shape
            annotations_xy_pd = pd.DataFrame(annotations_xy,
                                             columns=['name', 'xmin', 'ymin', 'xmax', 'ymax', 'confidence'])
            pascalVOC_path = utils_pascalVOC.export_annotations_pascal(img_list[file_id], annotations_xy_pd, width, height,
                                                                       export_path)

            export_to_biigle.pascalVOC_to_biigle(img_list[file_id], pascalVOC_path, label_idx, images_idx, 'Rectangle', api)


if __name__ == "__main__":
    import os
    import pandas as pd

    import feature_matching as fm
    import image_utils as iu
    import training_dataset as td
    from biigle import Api

    email = 'marin.marcillat@ifremer.fr'
    token = '9HTXoupsKlj3YyqH5vKCKYBvG1iwzbZV'

    api = Api(email, token)

    images_path = r'W:\images\ARDECO23\A3_PRES_S4'  # Directory with all the images
    report_path = r'D:\ARDECO\A3_PRES_S4\119-a3-pres-s4.csv'  # Path to Biigle report

    output_path = r'D:\ARDECO\A3_PRES_S4'  # Where to store output files

    img_ref_pos = 'A3_PRES_S4-00001.jpeg'  # reference image for homographic transformation

    ref_img_polyps = ['A3_PRES_S4-00001.jpeg']  # Reference image(s) for annotation positions
    labels_name = ["Madrepora"]  # List of possible labels

    label_tree_id = 64  # Label tree id
    volume_id = 119  # Volume id where images are stored

    train_path = os.path.join(output_path, 'train')
    export_path = os.path.join(output_path, 'temp')
    model_path = os.path.join(output_path, 'models')

    for path in [train_path, export_path, model_path]:
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)

    output_h_matrix_path = os.path.join(output_path, 'h_matrixs.txt')
    annotations = pd.read_csv(report_path)

    h_matrixs = pd.read_csv(output_h_matrix_path)

    print("Convert polyps coordinates...")
    polyps_positions = get_polyps_coords(images_path, annotations, h_matrixs, labels_name, output_path)

    polyp_ref = get_ref_polyps(polyps_positions, ref_img_polyps, output_path)

    iu.crop_all_images(images_path, polyp_ref, h_matrixs, export_path)

    print("ok")



