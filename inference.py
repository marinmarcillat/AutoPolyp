from biigle import Api
import imghdr
from tqdm import tqdm
from fastai.vision.all import *
import torch.cuda as tc

import utils_pascalVOC

import export_to_biigle


def model_inference(model_path, data_path, output_path, api, label_tree_id, volume_id):
    sess = load_learner(model_path, cpu=not tc.is_available())
    label_names = sess.dls.vocab

    delete_first = False

    original_positions_path = os.path.join(data_path, 'ori_positions.csv')
    original_positions = pd.read_csv(original_positions_path, )

    label_idx = export_to_biigle.create_label_index(api, label_tree_id, label_names)
    images_idx = export_to_biigle.create_image_index(api, volume_id)

    if delete_first:
        print("Deleting previous annotations...")
        for id, row in tqdm(images_idx.iterrows()):
            annotations = api.get(f"images/{row['id']}/annotations").json()
            for ann in annotations:
                ann_id = ann['id']
                api.delete(f'image-annotations/{ann_id}')

    print("Doing inference...")
    test_files = [os.path.join(data_path, str(row['index']) + "_" + row['filename']) for index, row in
                  original_positions.iterrows()]
    test_dl = sess.dls.test_dl(test_files)
    preds, _, decoded = sess.get_preds(dl=test_dl, with_decoded=True)
    inf_list = decoded.tolist()
    original_positions['prediction'] = inf_list
    original_positions['pred_label'] = original_positions['prediction'].apply(lambda x: label_names[x])
    img_list = original_positions['filename'].unique().tolist()
    original_positions['img_id'] = original_positions['filename'].apply(lambda x: img_list.index(x))
    original_positions.rename(columns={"index": "polyp_index"})
    exp_positions = original_positions[["img_id", 'filename', "polyp_index", "pred_label"]]
    exp_positions.to_csv(os.path.join(output_path, 'predictions.csv'), index=None)

    print("Exporting to Biigle...")
    for image in tqdm(original_positions['filename'].unique().tolist()):  # for each image in the directory
        vignettes = original_positions[original_positions['filename'] == image]
        annotations_xy = []
        for index, row in vignettes.iterrows():
            file_path = os.path.join(data_path, str(row['index']) + "_" + row['filename'])
            if os.path.isfile(file_path) and imghdr.what(file_path) == "jpeg":
                id_pred = test_files.index(file_path)
                inference_label = label_names[inf_list[id_pred]]
                ul = [row['x'], row['y']]
                lr = [row['x'] + row['w'], row['y'] + row['h']]
                annotations_xy.append([inference_label, ul[0], ul[1], lr[0], lr[1], 1])

        # Save to pascalVOC file
        height, width = row['img_h'], row['img_w']
        annotations_xy_pd = pd.DataFrame(annotations_xy, columns=['name', 'xmin', 'ymin', 'xmax', 'ymax', 'confidence'])
        pascalVOC_path = utils_pascalVOC.export_annotations_pascal(image, annotations_xy_pd, width, height, data_path)

        export_to_biigle.pascalVOC_to_biigle(image, pascalVOC_path, label_idx, images_idx, 'Rectangle', api)


if __name__ == "__main__":
    # Biigle email and token
    email = 'marin.marcillat@ifremer.fr'
    token = '9HTXoupsKlj3YyqH5vKCKYBvG1iwzbZV'
    label_tree_id = 64  # Label tree id
    volume_id = 114  # Volume id where images are stored

    images_path = r'W:\images\ARDECO23\A1_PRES_S3'  # Directory with all the images
    model_path = r"D:\ARDECO\A1_PRES_S3\models\model_export.pkl"
    output_path = r'D:\ARDECO\A1_PRES_S3'
    vign_path = os.path.join(output_path, 'temp')

    labels_name = ["Madrepora_extended", "Madrepora_retracted", "Madrepora_shy"]

    api = Api(email, token)

    model_inference(model_path, vign_path, output_path, api, label_tree_id, volume_id)
