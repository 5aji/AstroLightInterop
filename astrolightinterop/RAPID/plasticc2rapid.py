# Converts PLAsTiCC data to a format digestible by RAPID.

import time
import pandas as pd
import logging

class_map = {90: 1, 62: 2, 42: 3, 67: 4, 52: 5, 64: 6, 95: 7, 15: 8}
class_names = ('Pre-explosion', 'SNIa-norm', 'SNIbc', 'SNII', 'SNIa-91bg', 'SNIa-x', 'Kilonova', 'SLSN-I', 'TDE')

logger = logging.getLogger(__name__)


def _remap_class_values(metadata: pd.DataFrame, curves: pd.DataFrame, class_map: dict) -> (pd.DataFrame, pd.DataFrame):
    """
    Maps class values and removes unused classes from the dataset.

    :param class_map: The dictionary to use for mapping in {orig:result} form.
    :param metadata: The metadata to remap targets.
    :param curves: The curve data associated with the metadata.
    """

    logger.info("remap ping class values")
    start = time.process_time()
    # remap target class numbers to match output of RAPID.
    # Change this to change what each class maps to.
    metadata = metadata[metadata['target'].isin(class_map.keys())]
    metadata.loc[:, 'target'] = metadata.loc[:, 'target'].map(class_map)
    curves = curves[curves.index.isin(metadata.index, level=0)]
    logger.info("classes remapped in {0}".format(time.process_time() - start))
    return metadata, curves


def _remove_unused_bands(curves: pd.DataFrame) -> pd.DataFrame:
    logger.info("removing unused bands")
    start = time.process_time()
    # filter unused bands
    curves = curves[curves['passband'].isin([1, 2])]  # 1 and 2 are rgb bands.
    # FIXME: use loc instead of chain?
    curves.loc[:, 'passband'] = curves.loc[:, 'passband'].map({1: 'g', 2: 'r'})
    logger.info(curves)
    logger.info("unused bands removed in {0}".format(time.process_time() - start))
    return curves


def _calculate_triggers(curve: pd.DataFrame) -> pd.DataFrame:
    """
    Modify the curve (A dataframe) to have the correct triggering.

    :param curve: The curve to find the trigger frame for
    :return curve: The modified curve with first-detect triggering.
    """
    # map the detected column values (0,1) to the expected photflag values (0,4096, 6144)
    curve['detected'] = curve['detected'].map({0: 0, 1: 4096})

    # get the triggerpoint.
    # we are assuming that it is mjd sorted. (it should be)
    first_detect_index = curve['detected'].idxmax()
    # idxmax returns the index of the first occurrence of the max value
    # which in this case is the first occurance of 4096.
    # if curve.at[first_detect_index, 'detected'] == 4096:
    # we don't want to overwrite a non-detect frame
    curve.at[first_detect_index, 'detected'] = 6144
    return curve


def plasticc_to_rapid(metadata: pd.DataFrame, curves: pd.DataFrame) -> (list, list):
    """
    Converts the PLAsTiCC dataset into a set that RAPID can use natively.

    :param metadata: The metadata from PLAsTiCC
    :param curves: The light curve data from PLAsTiCC
    :returns light_list: a list of light curve tuples that RAPID takes as input
    :returns target_list: A list containing the matching targets from the dataset.
    """
    curves = _remove_unused_bands(curves)
    metadata, curves = _remap_class_values(metadata, curves, class_map)

    light_list = []
    target_list = []
    for meta in metadata.sample(frac=0.75).itertuples():
        curve = curves.loc[meta.Index]
        curve = _calculate_triggers(curve)
        light_list.append((
            curve.index.to_list(), curve['flux'].to_list(), curve['flux_err'].to_list(), curve['passband'].to_list(),
            curve['detected'].to_list(), meta.ra, meta.decl, meta.Index, meta.hostgal_specz, meta.mwebv))
        target_list.append(int(meta.target - 1))

    logger.info("Done processing light curves.")
    return light_list, target_list
