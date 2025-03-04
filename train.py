from fastai.vision.all import *
import fiftyone as fo
import fiftyone_utils as fou
import os

class TrainingSession:
    def __init__(self, data_path, model_type = 101, name = 'classification_model', split = 0.2):
        if model_type == 101:
            model = resnet101
        elif model_type == 34:
            model = resnet34
        self.data_path = data_path
        self.name = name

        fou.delete_all_datasets()
        self.dataset = fo.Dataset(f"{self.name}_dataset")

        self.dls = ImageDataLoaders.from_folder(self.data_path, valid_pct=split, item_tfms=Resize(64), batch_tfms=aug_transforms(size=64),
                                           num_workers=0)
        fou.import_dls(self.dls.valid, self.dataset)
        self.learn = vision_learner(self.dls, model, metrics=error_rate)

    def find_lr(self):
        self.learn.lr_find()

    def launch_train(self, model_dir, epochs = 30, lr = 5e-4):
        self.learn.fine_tune(epochs, lr)
        self.learn.save(os.path.join(model_dir, self.name))
        self.learn.export(os.path.join(model_dir, f"{self.name}_full_export.pkl"))

    def evaluate_model(self):
        fou.do_inference(self.learn, self.dataset)
        results = self.dataset.evaluate_classifications(
            "predictions", gt_field="ground_truth"
        )
        results.print_report()
        plot = results.plot_confusion_matrix()
        plot.show()

    def show_dataset(self):
        session = fo.launch_app(self.dataset)
        session.wait()

    def load_model(self, model_path):
        self.learn = load_learner(model_path, cpu=False)


if __name__ == '__main__':
    model_path = r""
    train_path = r""

    training_session = TrainingSession(train_path, model_type=101)
    training_session.find_lr()
    training_session.launch_train(model_path, epochs=100, lr=2e-3)

