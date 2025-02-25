from fastai.vision.all import *
import os

class TrainingSession:
    def __init__(self, data_path, model_type = 101):
        if model_type == 101:
            model = resnet101
        elif model_type == 34:
            model = resnet34
        self.data_path = data_path
        self.dls = ImageDataLoaders.from_folder(self.data_path, valid_pct=0.1, item_tfms=Resize(64), batch_tfms=aug_transforms(size=64),
                                           num_workers=0)
        self.learn = vision_learner(self.dls, model, metrics=error_rate)

    def find_lr(self):
        self.learn.lr_find()

    def launch_train(self, model_dir, epochs = 30, lr = 5e-4):
        self.learn.fine_tune(epochs, lr)
        self.learn.save(os.path.join(model_dir, 'model'))
        self.learn.export(os.path.join(model_dir, 'model_export.pkl'))

    def plot_matrixs(self):
        interp = Interpretation.from_learner(self.learn)
        interp.plot_top_losses(9, figsize=(15, 10))
        interp = ClassificationInterpretation.from_learner(self.learn)
        interp.plot_confusion_matrix()

    def load_model(self, model_dir):
        self.learn = load_learner(os.path.join(model_dir, 'model_export.pkl'), cpu=False)


_

