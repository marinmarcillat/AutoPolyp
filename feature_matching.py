import cv2 as cv
import numpy as np
import os, imghdr, ast
import pandas as pd
from tqdm import tqdm
import math
from scipy.spatial import distance



MIN_MATCH_COUNT = 10

def homographic_trsf(homography, coord):
    """
    Fonction to convert coordinates using a homographic matrix
    """
    x, y = coord
    p = np.array((x, y, 1)).reshape((3, 1))
    temp_p = homography.dot(p)
    sum = np.sum(temp_p, 1)
    px = int(round(sum[0] / sum[2]))
    py = int(round(sum[1] / sum[2]))
    return [px, py]

def convert_ann_homography(M, center, radius, reverse = False):
    if len(M) == 0:
        M = np.eye(3)
    if reverse:
        val, M = cv.invert(M)
    center = ast.literal_eval(center)
    ori_center = homographic_trsf(M, center)
    off_radius = [center[0] + radius, center[1]]
    off_radius = homographic_trsf(M, off_radius)
    ori_radius = int(distance.euclidean(ori_center, off_radius))
    return ori_center, ori_radius


def get_homographic_matrix(img_source, img_dest):
    ##### HOMOGRAPHY #####
    # Determine the homographic matrix using feature detection

    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img_source, None)
    kp2, des2 = sift.detectAndCompute(img_dest, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    else:
        print(f"Not enough matches are found - {len(good)}/{MIN_MATCH_COUNT}")
        M = []

    return M

def get_h_matrixs(data_path, img_ref_name, output_h_matrix_path):
    img_ref = cv.imread(str(os.path.join(data_path, img_ref_name)), cv.IMREAD_UNCHANGED)
    h_matrixs = []
    for file in tqdm(sorted(os.listdir(data_path))):  # for each image in the directory
        filename = os.path.join(data_path, file)
        if os.path.isfile(filename) and imghdr.what(filename) == "jpeg":
            if file == img_ref_name:
                M = []
            else:
                img_input = cv.imread(str(filename), cv.IMREAD_UNCHANGED)
                M = get_homographic_matrix(img_input, img_ref)
                if len(M) != 0:
                    M = M.tolist()
                else:
                    print(f"No match : {str(file)}")
            h_matrixs.append([file, M])

    df = pd.DataFrame(h_matrixs, columns=["filename","matrix"])
    df.to_csv(output_h_matrix_path, index=False)

















