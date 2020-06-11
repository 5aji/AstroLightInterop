"""Wrapper Module for RAPID

Follows the generic model API including train/test and init functions.

"""
import pandas as pd
import numpy as np
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

    def __init__(self, model: str = None):
        """
        Creates a new instance of RAPID
        Parameters
        ----------
        model: str, optional
            The filepath of the model to be loaded, if any.
        """
        if model is not None:
            self.classifier = Classify(known_redshift=True, model_filepath=model)
        else:
            self.classifier = Classify(known_redshift=True)

    def set_metadata(self, metadata: pd.DataFrame):
        """Sets the loaded metadata

        Parameters
        ----------
        metadata: pd.DataFrame :
            The new metadata to be loaded
        
        """
        assert isinstance(metadata, pd.DataFrame)
        self._metadata = metadata

    def set_curves(self, curves: pd.DataFrame):
        """Sets the loaded transient data

        Parameters
        ----------
        curves: pd.DataFrame :
            The new transient data to be loaded
        """
        assert isinstance(curves, pd.DataFrame)
        self._curves = curves

    def set_data(self, curves: pd.DataFrame, metadata: pd.DataFrame):
        """Set both the transient data and the metadata

        Parameters
        ----------
        curves: pd.DataFrame :
            The new transient data to be loaded
        metadata: pd.DataFrame :
            The new metadata to be loaded
        """
        self.set_curves(curves)
        self.set_metadata(metadata)

    def _get_custom_data(self, class_num, data_dir, save_dir, passbands, known_redshift, nprocesses,
                         redo):
        """Function for traning purposes.
        Notes
        -----
        See astrorapid for API usage.
        """
        light_list, target_list = p2r.convert(self._curves, self._metadata)
        # now we need to preprocess
        return read_multiple_light_curves(light_list)

    def train(self):
        """Train the model on the loaded data."""
        # we need to create a new model
        pass

    def test(self, curves: pd.DataFrame, metadata: pd.DataFrame, return_probabilities: bool =
        False) -> (list, list):
        """Tests the model on the currently loaded data.

        Parameters
        ----------
        curves : pd.DataFrame
            The transient data for each object.
        metadata : pd.DataFrame
            The metadata for each object.
        return_probabilities : bool, optional
            If the predictions should be a probability of arrays as opposed to the most likely class

        Returns
        -------
        target_list : list of int
            A list of true classes, ordered.
        predictions_list : list of int or list of list of int
            A list of predictions, ordered. Depending on the value of `return_probabilities`,
            can either be an int
        
        """
        logger.info("testing model")
        light_list, target_list = p2r.convert(curves, metadata)
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
