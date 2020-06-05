# Import the dataset.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from astrorapid.classify import Classify
import astrolightinterop.RAPID.plasticc2rapid as p2r


class Model:
    def __init__(self):
        self.classifier = Classify(known_redshift=True)

    def __call__(self, *args, **kwargs):
        self.train()

    def train(self):
        pass

    def test(self, curvefile, metafile):
        curvedata = pd.read_csv(curvefile, index_col=['object_id', 'mjd'])
        metadata = pd.read_csv(metafile, index_col='object_id')
        light_list, target_list = p2r.plasticc_to_rapid(metadata, curvedata)
        predictions, steps = self.classifier.get_predictions(light_list)
        assert len(target_list) == len(predictions)
        predictions_list = []
        for index, pred in enumerate(predictions):
            pred_class = np.argmax(pred[-1])
            # if pred_class is forbidden (0 or 9) don't bother
            # if pred_class < 0 or pred_class > 8:
            #     continue
            predictions_list.append(pred_class)
        target_list = np.add(target_list, 1)
        conf_matrix = normalize(confusion_matrix(target_list, predictions_list))

        plt.imshow(conf_matrix, cmap=plt.cm.RdBu)
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.savefig("cm.png")

        print(conf_matrix)
        pass
