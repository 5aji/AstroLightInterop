# Converts PLAsTiCC data to a format digestible by RAPID.

import time
import pandas as pd
import logging

class_map = {90: 1, 62: 2, 42: 3, 67: 4, 52: 5, 64: 6, 95: 7, 15: 8}
class_names = (
    'Pre-explosion', 'SNIa-norm', 'SNIbc', 'SNII', 'SNIa-91bg', 'SNIa-x', 'Kilonova', 'SLSN-I',
    'TDE')

# passband: The specific LSST passband integer, such that u, g, r, i, z, Y = 0, 1, 2, 3, 4, 5
# in which it was viewed.
band_map = {0: 'u', 1: 'g', 2: 'r', 3: 'i', 4: 'z', 5: 'Y'}
logger = logging.getLogger(__name__)


def _remap_class_values(metadata: pd.DataFrame, curves: pd.DataFrame, classes=None) -> (
        pd.DataFrame, pd.DataFrame):
    """Maps class values and removes unused classes from the dataset.

    Parameters
    ----------
    classes :
        The dictionary to use for mapping in {orig:result} form. (Default value = None) -> (pd.DataFrame)
    metadata :
        The curves to remap targets.
    curves :
        The curve data associated with the curves.
    metadata :
        pd.DataFrame:
    curves :
        pd.DataFrame:
    pd :
        DataFrame:
    metadata: pd.DataFrame :
        
    curves: pd.DataFrame :
        
    pd.DataFrame :
        

    Returns
    -------

    """

    if classes is None:
        classes = class_map
    logger.info("remap ping class values")
    start = time.process_time()
    # remap target class numbers to match output of RAPID.
    # Change this to change what each class maps to.
    metadata = metadata[metadata['target'].isin(classes.keys())]
    metadata.loc[:, 'target'] = metadata.loc[:, 'target'].map(classes)
    curves = curves[curves.index.isin(metadata.index, level=0)]
    logger.info("classes remapped in {0}".format(time.process_time() - start))
    return metadata, curves


def _remove_unused_bands(curves: pd.DataFrame, bands: dict = None) -> pd.DataFrame:
    """Removes unused bands and maps band ints to their string representation for RAPID.

    Parameters
    ----------
    curves :
        param bands:
    curves :
        pd.DataFrame:
    bands :
        dict:  (Default value = None)
    curves: pd.DataFrame :
        
    bands: dict :
         (Default value = None)

    Returns
    -------

    """
    if bands is None:
        bands = {1: 'g', 2: 'r'}
    logger.info("removing unused bands")
    start = time.process_time()
    # filter unused bands
    curves = curves[curves['passband'].isin(bands.keys())]  # 1 and 2 are rgb bands.
    # FIXME: use loc instead of chain?
    curves.loc[:, 'passband'] = curves.loc[:, 'passband'].map(bands)
    logger.info("unused bands removed in {0}".format(time.process_time() - start))
    return curves


def _calculate_triggers(curve: pd.DataFrame) -> pd.DataFrame:
    """Modify the curve (A dataframe) to have the correct triggering.

    Parameters
    ----------
    curve :
        The curve to find the trigger frame for
        :return curve: The modified curve with first-detect triggering.
    curve :
        pd.DataFrame:
    curve: pd.DataFrame :
        

    Returns
    -------

    """
    # map the detected column values (0,1) to the expected photflag values (0,4096, 6144)
    curve['detected'] = curve['detected'].map({0: 0, 1: 4096})
    # get the triggerpoint.
    # we are assuming that it is mjd sorted. (it should be)
    first_detect_index = curve['detected'].idxmax()
    # idxmax returns the index of the first occurrence of the max value
    # which in this case is the first occurance of 4096.
    curve.at[first_detect_index, 'detected'] = 6144
    return curve


def convert(curves: pd.DataFrame, metadata: pd.DataFrame, bands: dict = None,
            classes: dict = None) -> (list, list):
    """Converts the PLAsTiCC dataset into a set that RAPID can use natively.

    Parameters
    ----------
    classes :
        The class map to use. This will specify what classes are included and their new
        values.
    bands :
        The band mapping to use. Bands not specified will be removed from the returned
        lists.
    metadata :
        The curves from PLAsTiCC
    curves :
        The light curve data from PLAsTiCC
    curves :
        pd.DataFrame:
    metadata :
        pd.DataFrame:
    bands :
        dict:  (Default value = None)
    classes :
        dict:  (Default value = None) -> (list)
    list :
        returns: light_list: a list of light curve tuples that RAPID takes as input
    curves: pd.DataFrame :
        
    metadata: pd.DataFrame :
        
    bands: dict :
         (Default value = None)
    classes: dict :
         (Default value = None) -> (list)

    Returns
    -------
    type
        light_list: a list of light curve tuples that RAPID takes as input

    """
    if bands is None:
        bands = {1: 'g', 2: 'r'}
    if classes is None:
        classes = class_map
    curves = _remove_unused_bands(curves, bands)
    metadata, curves = _remap_class_values(metadata, curves, classes)

    light_list = []
    target_list = []
    for meta in metadata.itertuples():
        curve = curves.loc[meta.Index]
        curve = _calculate_triggers(curve)
        light_list.append((
            curve.index.to_list(), curve['flux'].to_list(), curve['flux_err'].to_list(),
            curve['passband'].to_list(),
            curve['detected'].to_list(), meta.ra, meta.decl, meta.Index, meta.hostgal_specz,
            meta.mwebv))
        target_list.append(int(meta.target - 1))

    logger.info("Done processing light curves.")
    return light_list, target_list
