"""

"""
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
class_names = (
    'Pre-explosion', 'SNIa-norm', 'SNIbc', 'SNII', 'SNIa-91bg', 'SNIa-x', 'Kilonova', 'SLSN-I',
    'TDE')


class RAPIDModel:
    """Wrapper for the model outlined by the RAPID paper (Muthukrishna et al 2019)"""
    def __init__(self, curves: pd.DataFrame, metadata: pd.DataFrame, model: str = None):
        """

        Parameters
        ----------
        curves: pd.Dataframe

        metadata
        model
        """
        if model is not None:
            self.classifier = Classify(known_redshift=True, model_filepath=model)
        else:
            self.classifier = Classify(known_redshift=True)
        try:
            self._curves = curves
            self._metadata = metadata
        except NameError:
            print("missing a filetype")
            raise NameError

    def set_metadata(self, metadata: pd.DataFrame):
        """Sets the loaded

        Parameters
        ----------
        metadata: pd.DataFrame :
            

        Returns
        -------

        
        """
        assert isinstance(metadata, pd.DataFrame)
        self._metadata = metadata

    def set_curves(self, curves: pd.DataFrame):
        """

        Parameters
        ----------
        curves: pd.DataFrame :
            

        Returns
        -------

        """
        assert isinstance(curves, pd.DataFrame)
        self._curves = curves

    def set_data(self, curves: pd.DataFrame, metadata: pd.DataFrame):
        """

        Parameters
        ----------
        curves: pd.DataFrame :
            
        metadata: pd.DataFrame :
            

        Returns
        -------

        """
        assert isinstance(curves, pd.DataFrame)
        assert isinstance(metadata, pd.DataFrame)
        self._curves = curves
        self._metadata = metadata

    def _get_custom_data(self, class_num, data_dir, save_dir, passbands, known_redshift, nprocesses,
                         redo):
        """

        Parameters
        ----------
        class_num :
            
        data_dir :
            
        save_dir :
            
        passbands :
            
        known_redshift :
            
        nprocesses :
            
        redo :
            

        Returns
        -------

        """
        light_list, target_list = p2r.convert(self._curves, self._metadata)
        # now we need to preprocess
        return read_multiple_light_curves(light_list)

    def train(self):
        """ """
        # we need to create a new model
        pass

    def test(self, return_probabilities: bool = False) -> (list, list):
        """Tests the model on the currently loaded data.

        Parameters
        ----------
        return_probabilities : bool, optional
            If the predictions should be a probability of arrays as opposed to the most likely class
        return_probabilities: bool :
             (Default value = False) -> (list)
        list :
            

        Returns
        -------

        
        """
        logger.info("testing model")
        light_list, target_list = p2r.convert(self._curves, self._metadata)
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
