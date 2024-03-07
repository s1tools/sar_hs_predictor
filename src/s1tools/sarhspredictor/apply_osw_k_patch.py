import datetime
import logging

import numpy as np

from .reference_oswk import reference_oswK_954m_30pts, reference_oswK_954m_60pts, reference_oswK_1145m_60pts


def patch_osw_k(k_sar, ipfvesion=None, datedtsar=None):
    """
    :args:
        k_sar (nd.array): oswK content
        ipfvesion (str): '002.53' for instance
        datedtsar (datetime): start date of the product considered
    #first value seen in oswK:
    20200101 -> 60 pts 0.005235988 IPF 3.10
    20190101 -> 60 pts 0.005235988 IPF 2.91
    20180101 -> 60 pts 0.005235988 IPF 2.84
    20170101 -> 60 pts 0.005235988 IPF 2.72
    20151125 -> 60 pts 0.005235988 IPF 2.60
    20150530T195229 -> passage 2.53 vers 2.60 -> oswK extended from 954m up to 1145m wavelength
    20151124 -> 60 pts 0.006283185 IPF 2.53
    20150530T195229 -> passage 2.43 vers 2.53 -> resolution 72x60
    20150530 -> 30 pts 0.006283185 IPF 2.43
    20150203 -> 30 pts 0.006283185 IPF 2.36
    to patch for oswK vectors from WV S-1 osw products that could contains NaN or masked values
    :return:
    """
    if isinstance(k_sar, np.ma.core.MaskedArray):
        test_oswk_KO = k_sar.mask.any()
    else:
        test_oswk_KO = not np.isfinite(k_sar).all()  # or k_sar.mask.any()

    if test_oswk_KO:
        # starting from 3 july 2015 (+1 month ok: june 2015) in we have 60 elements in oswK
        # but still some oswK can contains erroneous masked values
        if len(k_sar) == 30:
            k_sar = reference_oswK_954m_30pts
            logging.info('ref k 30 elements (instead of 60) %s' % k_sar.shape)
        else:
            if ipfvesion is not None:
                if ipfvesion in ['002.53']:
                    k_sar = reference_oswK_954m_60pts
                else:
                    k_sar = reference_oswK_1145m_60pts
            else:
                if datedtsar < datetime.datetime(2015, 11, 24, 19, 52, 29):  # date of IPF 2.53 -> 2.60
                    k_sar = reference_oswK_954m_60pts
                else:
                    k_sar = reference_oswK_1145m_60pts  # latest version of oswK on 60 points
    return k_sar
