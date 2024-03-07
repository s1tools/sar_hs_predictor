import sys

import numpy as np
import xarray

from . import preprocess
from .preproc_ocn_wv import preproc_ocn_wv


def prepare_ocn_wv_data(pattern_path, log):
    """
    :param pattern_path: could also be a list of path
    :param log: logger
    :return:
    """
    log.info('start reading S1 WV OCN data')
    ocn_wv_ds = xarray.open_mfdataset(pattern_path, combine='by_coords', concat_dim='time', preprocess=preproc_ocn_wv)
    log.info('Nb pts in dataset: %s' % ocn_wv_ds['todSAR'].size)
    log.info('SAR data ready to be used')
    cspcRe = ocn_wv_ds['oswQualityCrossSpectraRe'].values
    cspcIm = ocn_wv_ds['oswQualityCrossSpectraIm'].values

    re = preprocess.conv_real(cspcRe)
    im = preprocess.conv_imaginary(cspcIm)
    spectrum = np.stack((re, im), axis=3)
    return spectrum, ocn_wv_ds


def define_features(ds):
    """
    :param ds: xarrayDataArray of OCN WV data
    :return:
    """
    features = None
    for jj in ['cwave', 'dxdt', 'latlonSARcossin', 'todSAR', 'incidence', 'satellite']:
        addon = ds[jj].values
        if len(addon.shape) == 1:
            addon = addon.reshape((addon.size, 1))
        if features is None:
            features = addon
        else:
            features = np.hstack([features, addon])

    return features


def define_input_test_dataset(features, spectrum):
    """

    :param features: 2D np matrix
    :param spectrum: 4D np matrix
    :return:
    """
    outputs = np.zeros(features.shape[0])
    inputs = [spectrum, features]
    test = (inputs, outputs)
    return test


def main_level_1(pattern_path, model, log):
    """
    :param pattern_path: (str) or list of str path
    :param model:
    :return:
    """
    spectrum, s1_ocn_wv_ds = prepare_ocn_wv_data(pattern_path, log)
    features = define_features(s1_ocn_wv_ds)
    inputs, _ = define_input_test_dataset(features, spectrum)

    yhat = np.vstack([model.predict_on_batch(inputs)])

    s1_ocn_wv_ds['swh'] = xarray.DataArray(data=yhat[:, 0], dims=['time'])
    s1_ocn_wv_ds['swh_uncertainty'] = xarray.DataArray(data=yhat[:, 1], dims=['time'])
    return s1_ocn_wv_ds
