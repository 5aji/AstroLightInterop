# ingests ZTF data and output PLAsTiCC-style dataframes (which can then be saved as csv)
import glob
import gzip
import logging
import os
import re
from pathlib import Path
from multiprocessing import Pool
import pandas as pd
from astropy.cosmology import Planck18_arXiv_v2 as cosmo
from astropy.table import Table

logger = logging.getLogger(__name__)

# Define some constants
# Note that the data sometimes uses DECL or DEC. It can't decide?
head_map = {"SNID": "object_id", "RA": "ra", "DECL": "decl","DEC": "decl", "HOSTGAL_SPECZ": "hostgal_specz",
            "SIM_REDSHIFT_HOST": "hostgal_photz", "HOSTGAL_PHOTOZ_ERR": "hostgal_photz_err", "SIM_MWEBV": "mwebv",
            "SIM_TYPE_INDEX": "target"}
# PHOT object_id,mjd,passband,flux,flux_err,detected
phot_map = {"MJD": "mjd", "FLT": "passband", "FLUXCAL": "flux", "FLUXCALERR": "flux_err", "PHOTFLAG": "detected"}
band_map = {b"g ": 1, b"r ": 2}

def get_data(fits_archive: Path, use_gzip=True):
    # gets the data from a fits archive.
    if use_gzip:
        file = gzip.open(fits_archive, 'r')
    else:
        file = fits_archive
    table = Table.read(file, format='fits')
    if use_gzip:
        file.close()
    return table.to_pandas()


def convert(head, phot):
    # take the ZTF data and ouput PLaSTiCC metadata and curve data frames
    # use model num and file num to hash object_id.

    phot = phot[phot.MJD != -777]  # For some reason there's a rogue entry here.
    curves = phot[list(phot_map.keys())].rename(columns=phot_map)
    meta = head.rename(columns=head_map) # dont discard yet, we need some of the columns.
    curves['passband'] = curves['passband'].map(band_map)

    # assign correct object ids
    # use the PTROBS_MAX column which points to the last row of the curve
    meta['PTROBS_MAX'] -= 1
    # create a series with ptrmax indx and object_id
    object_id = pd.to_numeric(meta.set_index('PTROBS_MAX').loc[:, 'object_id'], downcast='integer')
    # add the series and backfill to propagate. Then set new indices
    curves = curves.assign(object_id=object_id).fillna(method='bfill').set_index(['object_id', 'mjd'])

    # finally, discard unneeded data.
    meta = meta[list(head_map.values())]
    # Calc and append distmod for the metadata
    distmod = pd.Series(cosmo.distmod(meta.hostgal_photz).value, index=meta.index)
    meta = meta.assign(distmod=distmod)

    meta.loc[:, 'ddf'] = 0 # Add a blank ddf col
    # also, set proper index
    meta['object_id'] = pd.to_numeric(meta['object_id'])
    meta = meta.set_index('object_id')
    # TODO: remap targets.
    return meta, curves


def model_loader(directory: Path = None):
    """
    Loads an individual model from a directory
    Parameters
    ----------
    directory : Path
        The base path of the model. In the SNANA files, this would be ZTF_MSIP_MODELXX

    Returns
    -------
    meta : pd.DataFrame
        The metadata for the curves.
    curves : pd.DataFrame
        The light curves themselves.
    """
    print("Loading model at", directory.stem)
    assert directory is not None
    # create lists of HEAD and PHOT files.
    heads = sorted(directory.glob("*_HEAD.FITS.gz"))
    phots = sorted(directory.glob("*_PHOT.FITS.gz"))
    assert len(heads) == len(phots)  # we should have 1-to-1 mapping.
    meta = pd.DataFrame()
    curves = pd.DataFrame()
    # iterate, convert, append. Simple stuff.
    # Could be multiprocess if you want? you might kill the kernel with short-lived processes.
    for i in range(len(heads)):
        curr_head = get_data(heads[i])
        curr_phot = get_data(phots[i])
        curr_meta, curr_curves = convert(curr_head, curr_phot)
        meta = meta.append(curr_meta)
        curves = curves.append(curr_curves)

    return meta, curves


def loader(directory: Path = None, aggregate: bool = False ):
    """Loads ZTF data from a directory.
    Parameters
    ----------
    directory : str
        The directory to load the data from.
    aggregate : bool
        If true, combines all DataFrames into one before inserting into the list
    Returns
    -------
    A list of tuples containing curves and their metadata.
    """
    assert directory is not None
    model_dirs = []
    for entry in directory.iterdir():
        if entry.is_dir() and len(list(entry.glob("*.LIST"))) > 0: # it has a .list file. This is slow.
            model_dirs.append(entry)
    with Pool() as pool:
        models = pool.map(model_loader, model_dirs)
    return models


if __name__ == "__main__":
    p = Path("../../data/DATA/ZTF_20190512")
    results = loader(p)
    print(results)
