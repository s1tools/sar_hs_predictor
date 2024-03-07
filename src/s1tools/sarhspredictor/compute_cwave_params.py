"""
copy paste from compute_hs_total_SAR_v2.py 31 January 2021

v2 is copy/paste from v1c, it is a CWAVE algo python that use the polSpec from OCN products,
it extracts the 20 param and use a CWAVE v2 model tuned on altimeters (only numpy dependencies) provided by Yannick Glaser (University of Hawaii)
date creation: 9 July 2019

C-WAVE params consist in 20 values computed from  Sentinel-1 C-band SAR WV image cross spectra

:env: export PYTHONPATH=~/sources/git/npCWAVE/
:purpose: methods to get total empirical hs from L2 SAR S-1 WV 
validated with
python 3.7.3
numpy                     1.13.1
scipy                     0.19.1
mkl                       2019.0

"""

import netCDF4
import numpy as np
import pandas as pd

from .pol_cart_trans_jstopa_transcoding_furniture_cls import pol_cart_trans


def format_input_cwave_vector_from_ocn(cspc_re, cspc_im, ths1, ta, incidenceangle, s0, nv, ks1, datedt, lons, lats,
                                       satellite):
    """
    v2 is copy/paste from v1c, it is a CWAVE algo python that use the polSpec from OCN products,
    it extracts the 20 param and use a CWAVE v2 model tuned on altimeters (only numpy dependancies)
    provided by Yannick Glaser (University of Hawaii)
    date creation: 9 Juillet 2019
    example 1:
       s1a-wv1-ocn-vv-20151130t175854-20151130t175857-008837-00c9eb-087.nc
       hsSM=3.4485

    example 2:
       s1a-wv2-ocn-vv-20151130t201457-20151130t201500-008838-00c9f5-046.nc
       hsSM=1.3282

    :param cspc_re: polSpecRe 60x72 (k,phi)
    :param cspc_im: polSpecIm 60x72 (k,phi)
    :param ths1:
    :param ta:
    :param incidenceangle:
    :param s0: oswNrcs
    :param nv:
    :param ks1: oswK
    :param datedt:
    :param lons:
    :param lats:
    :param satellite: s1a or...
    :return: hs_total_sar (float) in meters
    """
    # Constants for CWAVE
    # ===================
    NTH = 72  # number of output wave directions on log-polar grid
    kmax = 2 * np.pi / 60  # kmax for empirical Hs
    kmin = 2 * np.pi / 625  # kmin for empirical Hs
    ns = 20  # number of variables in orthogonal decomposition
    S = np.ones((ns, 1)) * np.nan
    dky = 0.002954987953815
    nky = 85
    dkx = 0.003513732113299
    nkx = 71

    kx = (np.arange(0, nkx).T - np.floor(nkx / 2)) * dkx
    kx[int(np.floor(nkx / 2))] = 0.0001
    ky = (np.arange(0, nky).T - np.floor(nky / 2)) * dky
    ky[int(np.floor(nky / 2))] = 0.0001

    KX, KY = np.meshgrid(ky, kx)
    condition_keep = (abs(KX) >= kmin) & (abs(KX) <= kmax) & (abs(KY) >= kmin) & (abs(KY) <= kmax)
    indices = np.where(condition_keep)
    gdx, gdy = indices
    assert KX.shape == (71, 85)

    KX = KX[gdx, gdy]
    KY = KY[gdx, gdy]
    uniq_gdx = np.unique(gdx)  # 54x1 in matlab
    uniq_gdy = np.unique(gdy)  # 64x1 in matlab
    KX = np.reshape(KX, (len(uniq_gdx), len(uniq_gdy)))
    KY = np.reshape(KY, (len(uniq_gdx), len(uniq_gdy)))

    DKX = np.ones(KX.shape) * 0.003513732113299
    DKY = np.ones(KX.shape) * 0.002954987953815

    flagKcorrupted = (ks1 > 1000).any()

    subset_ok = dict()
    subset_ok['todSAR'] = _conv_time(netCDF4.date2num(datedt, 'hours since 2010-01-01T00:00:00Z UTC'))
    subset_ok['lonSAR'] = lons
    subset_ok['latSAR'] = lats
    subset_ok['incidenceAngle'] = incidenceangle
    subset_ok['sigma0'] = s0
    subset_ok['normalizedVariance'] = nv

    if cspc_re.shape[0] == 60 and (cspc_re > 0).any():
        # Convert to kx,ky spectrum
        a1 = np.radians(ta)
        a2 = np.radians(ths1)
        dif = np.arctan2(np.sin(a2 - a1), np.cos(a2 - a1))
        strr = np.argmin(abs(dif))
        idd = np.mod(np.arange(strr, strr + NTH), NTH)

        interpmethod = 'linear'  # D max diff 38 too smooth 0.05m diff (difference au centre du plot (k tres petit)
        #         interpmethod = 'nearest' #D max diff 25 too smooth but worst hs 0.14m diff
        #         interpmethod = 'cubic' #D max diff 28 too smooth but worst hs 0.05m diff

        cspcReX, cspcReX_not_conservativ = pol_cart_trans(d=cspc_re[:, idd], k=ks1, t=np.radians(ths1), x=kx, y=ky,
                                                          name='re', interpmethod=interpmethod)
        assert cspcReX.shape == (71, 85)  # or cspcReX.shape==(85,71) #info de justin
        cspcImX, cspcImX_not_conservativ = pol_cart_trans(d=cspc_im[:, idd], k=ks1, t=np.radians(ths1), x=kx, y=ky,
                                                          name='im', interpmethod=interpmethod)

        cspc = np.sqrt(cspcReX ** 2 + cspcImX ** 2)

        cspc = cspc[gdx, gdy]
        cspc = np.reshape(cspc, (len(uniq_gdx), len(uniq_gdy)))

        # Compute Orthogonal Moments
        # ==========================
        gamma = 2
        a1 = (gamma ** 2 - np.power(gamma, 4)) / (gamma ** 2 * kmin ** 2 - kmax ** 2)
        a2 = (kmax ** 2 - np.power(gamma, 4) * kmin ** 2) / (kmax ** 2 - gamma ** 2 * kmin ** 2)

        # Ellipse
        tmp = a1 * np.power(KX, 4) + a2 * KX ** 2 + KY ** 2
        # eta
        eta = np.sqrt((2. * tmp) / ((KX ** 2 + KY ** 2) * tmp * np.log10(kmax / kmin)))

        alphak = 2. * ((np.log10(np.sqrt(tmp)) - np.log10(kmin)) / np.log10(kmax / kmin)) - 1
        alphak[(alphak ** 2) > 1] = 1.

        alphat = np.arctan2(KY, KX)

        # Gegenbauer polynomials
        tmp = abs(np.sqrt(1 - alphak ** 2))  # imaginary???

        g1 = 1 / 2. * np.sqrt(3) * tmp
        g2 = 1 / 2. * np.sqrt(15) * alphak * tmp
        g3 = np.dot((1 / 4.) * np.sqrt(7. / 6.), (15. * np.power(alphak, 2) - 3.)) * tmp  #
        g4 = (1 / 4.) * np.sqrt(9. / 10) * (35. * np.power(alphak, 3) - 15. * alphak ** 2) * tmp

        # Harmonic functions
        f1 = np.sqrt(1 / np.pi) * np.cos(0. * alphat)
        f2 = np.sqrt(2 / np.pi) * np.sin(2. * alphat)
        f3 = np.sqrt(2 / np.pi) * np.cos(2. * alphat)
        f4 = np.sqrt(2 / np.pi) * np.sin(4. * alphat)
        f5 = np.sqrt(2 / np.pi) * np.cos(4. * alphat)

        # Weighting functions
        h = np.ones((KX.shape[0], KX.shape[1], 20))
        h[:, :, 0] = g1 * f1 * eta
        h[:, :, 1] = g1 * f2 * eta
        h[:, :, 2] = g1 * f3 * eta
        h[:, :, 3] = g1 * f4 * eta
        h[:, :, 4] = g1 * f5 * eta
        h[:, :, 5] = g2 * f1 * eta
        h[:, :, 6] = g2 * f2 * eta
        h[:, :, 7] = g2 * f3 * eta
        h[:, :, 8] = g2 * f4 * eta
        h[:, :, 9] = g2 * f5 * eta
        h[:, :, 10] = g3 * f1 * eta
        h[:, :, 11] = g3 * f2 * eta
        h[:, :, 12] = g3 * f3 * eta
        h[:, :, 13] = g3 * f4 * eta
        h[:, :, 14] = g3 * f5 * eta
        h[:, :, 15] = g4 * f1 * eta
        h[:, :, 16] = g4 * f2 * eta
        h[:, :, 17] = g4 * f3 * eta
        h[:, :, 18] = g4 * f4 * eta
        h[:, :, 19] = g4 * f5 * eta

        try:
            P = cspc / (np.nansum(np.nansum(cspc * DKX * DKY)))  # original
        except:
            P = np.array([1])  # trick to return something but not usable
            flagKcorrupted = True

        if not np.isfinite(s0) or isinstance(s0, np.ma.core.MaskedConstant):  # or s0.mask.all():
            s0 = 0
            flagKcorrupted = True
        if not np.isfinite(nv) or isinstance(nv, np.ma.core.MaskedConstant):  # or nv.mask.all():
            nv = 0
            flagKcorrupted = True
        if not np.isfinite(incidenceangle) or isinstance(incidenceangle,
                                                         np.ma.core.MaskedConstant):  # or incidenceangle.mask.all():
            flagKcorrupted = True

        for jj in range(ns):
            petit_h = h[:, :, jj].squeeze().T
            S[jj] = np.nansum(np.nansum(petit_h * P.T * DKX.T * DKY.T))

        for iiu in range(len(S)):
            subset_ok['s' + str(iiu)] = S[iiu][0]
        # encodes type A as 1 and B as 0
        if satellite == 's1a':
            subset_ok['sentinelType'] = 1
        else:
            subset_ok['sentinelType'] = 0
        try:
            subset_ok = pd.DataFrame(subset_ok, index=[0])
        except:
            pass
    else:
        cspcReX = np.zeros((71, 85))
        cspcImX = np.zeros((71, 85))
        cspcReX_not_conservativ = np.zeros((71, 85))

    return subset_ok, flagKcorrupted, cspcReX, cspcImX, cspc_re, ks1, ths1, kx, ky, cspcReX_not_conservativ, S


def _conv_time(in_t):
    """
    Converts data acquisition time

    Args:
        in_t: time of data acquisition in format hours since 2010-01-01T00:00:00Z UTC

    Returns:
        Encoding of time where 00:00 and 24:00 are -1 and 12:00 is 1
    """
    in_t = in_t % 24
    return 2 * np.sin((2 * np.pi * in_t) / 48) - 1
