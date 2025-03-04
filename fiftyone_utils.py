import fiftyone as fo
from fiftyone import ViewField as F
import numpy as np
import os

def delete_all_datasets():
    for ds in fo.list_datasets():
        dataset = fo.load_dataset(ds)
        dataset.delete()

def import_image_directory(data_path, dataset):
    samples = []
    classes = []
    for root, dirs, files in os.walk(data_path):
        for d in dirs:
            classes.append(d)
            for file in os.listdir(os.path.join(root, d)):
                sample = fo.Sample(filepath=os.path.join(root, d, file))
                sample["ground_truth"] = fo.Classification(label=d)
                samples.append(sample)
    dataset.add_samples(samples)
    dataset.save()

def import_dls(dl, dataset):
    samples = []
    for filepath in dl.items:
        sample = fo.Sample(filepath=filepath)
        sample["ground_truth"] = fo.Classification(label=filepath.parts[-2])
        samples.append(sample)
    dataset.add_samples(samples)
    dataset.save()

def do_inference(learner, dataset):
    classes = learner.dls.vocab

    images = [sample.filepath for sample in dataset]

    dl = learner.dls.test_dl(images)
    preds, _ = learner.get_preds(dl=dl)
    preds = preds.numpy()

    # Save predictions to FiftyOne dataset
    with fo.ProgressBar() as pb:
        for sample, scores in zip(pb(dataset), preds):
            target = np.argmax(scores)
            sample["predictions"] = fo.Classification(
                label=classes[target],
                confidence=scores[target],
                logits=np.log(scores),
            )
            sample.save()