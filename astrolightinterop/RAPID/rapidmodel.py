# Import the dataset.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from astrorapid.classify import Classify
from astrorapid.process_light_curves import read_multiple_light_curves
import logging
import astrolightinterop.RAPID.plasticc2rapid as p2r

logger = logging.getLogger(__name__)
# API standards
class_names = ('Pre-explosion', 'SNIa-norm', 'SNIbc', 'SNII', 'SNIa-91bg', 'SNIa-x', 'Kilonova', 'SLSN-I', 'TDE')


class Model:
    def __init__(self, curvefile: str, metafile: str, model: str):
        if model:
            self.classifier = Classify(known_redshift=True, model_filepath=model)
        else:
            self.classifier = Classify(known_redshift=True)
        try:
            self._curvefile = curvefile
            self._metafile = metafile
        except NameError:
            print("missing a filetype")
            raise NameError

    def set_metafile(self, metafile: str):
        self._metafile = metafile

    def set_curvefile(self, curvefile: str):
        self._curvefile = curvefile

    def _get_custom_data(self, class_num, data_dir, save_dir, passbands, known_redshift, nprocesses, redo):
        curvedata = pd.read_csv(self._curvefile, index_col=['object_id', 'mjd'])
        metadata = pd.read_csv(self._metafile, index_col='object_id')
        light_list, target_list = p2r.convert(metadata, curvedata)
        # now we need to preprocess
        return read_multiple_light_curves(light_list)

    def train(self):
        # we need to create a new model
        pass

    def test(self, return_probabilities: bool = False) -> (list, list):
        """
        :param return_probabilities: If the function should return the raw probabilities (and not just the most likely)
        :return: a tuple of lists, one containing the target class and the other containing the output of the classifier
        """
        logger.info("testing model")
        curvedata = pd.read_csv(self._curvefile, index_col=['object_id', 'mjd'])
        metadata = pd.read_csv(self._metafile, index_col='object_id')
        light_list, target_list = p2r.convert(metadata, curvedata)
        predictions, steps = self.classifier.get_predictions(light_list)
        assert len(target_list) == len(predictions)
        target_list = np.add(target_list, 1)
        logger.info("done testing model")
        if return_probabilities:
            return target_list, predictions

        predictions_list = []
        for index, pred in enumerate(predictions):
            pred_class = np.argmax(pred[-1])
            # if pred_class is forbidden (0 or 9) don't bother
            # if pred_class < 0 or pred_class > 8:
            #     continue
            predictions_list.append(pred_class)
        return target_list, predictions_list
