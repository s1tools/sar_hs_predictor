import datetime
import os

import numpy as np
import xarray

from . import preprocess
from .apply_osw_k_patch import patch_osw_k
from .compute_cwave_params import format_input_cwave_vector_from_ocn
from .reference_oswk import reference_oswK_1145m_60pts


def preproc_ocn_wv(ds):
    """
    read and preprocess data for training/usage of the model
    :param ds:
    :return:
    """
    filee = ds.encoding["source"]
    fdatedt = datetime.datetime.strptime(os.path.basename(filee).split('-')[4], '%Y%m%dt%H%M%S')
    try:
        ds['time'] = xarray.DataArray(np.array([fdatedt]), dims=['time'], coords={'time': [0]})
        ds = ds.sortby('time', ascending=True)
    except:
        pass
    newds = xarray.Dataset()
    # format data for CWAVE 22 params computation
    cspc_re = ds['oswQualityCrossSpectraRe'].values.squeeze()
    cspc_im = ds['oswQualityCrossSpectraIm'].values.squeeze()
    ths1 = np.arange(0, 360, 5)
    ks1 = patch_osw_k(ds['oswK'].values.squeeze(), ipfvesion=None, datedtsar=fdatedt)
    if cspc_re.shape == (36, 30):
        cspc_re = np.zeros((72, 60))
        cspc_im = np.zeros((72, 60))
        ks1 = reference_oswK_1145m_60pts  # we decided to not give predictions for spectra with a shape 36 30

    ta = ds['oswHeading'].values.squeeze()
    incidenceangle = ds['oswIncidenceAngle'].values.squeeze()
    s0 = ds['oswNrcs'].values.squeeze()
    nv = ds['oswNv'].values.squeeze()
    lon_sar = ds['oswLon'].values.squeeze()
    lat_sar = ds['oswLat'].values.squeeze()
    satellite = os.path.basename(filee)[0:3]
    subset_ok, flag_k_corrupted, cspc_re_x, cspc_im_x, _, ks1, ths1, kx, ky, \
        cspc_re_x_not_conservativ, s = format_input_cwave_vector_from_ocn(cspc_re=cspc_re.T,
                                                                          cspc_im=cspc_im.T, ths1=ths1, ta=ta,
                                                                          incidenceangle=incidenceangle,
                                                                          s0=s0, nv=nv, ks1=ks1, datedt=fdatedt,
                                                                          lons=lon_sar, lats=lat_sar,
                                                                          satellite=satellite)
    varstoadd = ['S', 'cwave', 'dxdt', 'latlonSARcossin', 'todSAR',
                 'incidence', 'incidence_angle', 'satellite', 'oswQualityCrossSpectraRe', 'oswQualityCrossSpectraIm']
    additional_vars_for_validation = ['oswLon', 'oswLat', 'oswLandFlag', 'oswIncidenceAngle', 'oswWindSpeed',
                                      'platformName',
                                      'nrcs', 'nv', 'heading', 'oswK', 'oswNrcs']
    varstoadd += additional_vars_for_validation
    if 'time' in ds:
        newds['time'] = ds['time']
    else:
        newds['time'] = xarray.DataArray(np.array([fdatedt]), dims=['time'], coords={'time': [0]})
    for vv in varstoadd:
        if vv in ['cwave']:
            dimszi = ['time', 'cwavedim']
            coordi = {'time': [fdatedt], 'cwavedim': np.arange(22)}
            cwave = np.hstack([s.T, s0.reshape(-1, 1), nv.reshape(-1, 1)])  # found L77 in preprocess.py
            cwave = preprocess.conv_cwave(cwave)
            newds[vv] = xarray.DataArray(data=cwave, dims=dimszi, coords=coordi)
        elif vv == 'S':  # to ease the comparison with Justin files
            dimszi = ['time', 'Sdim']
            coordi = {'time': [fdatedt], 'Sdim': np.arange(20)}
            newds[vv] = xarray.DataArray(data=s.T, dims=dimszi, coords=coordi)
        elif vv in ['dxdt']:
            # dx and dt and delta from coloc with alti see /home/cercache/users/jstopa/sar/empHs/cwaveV5,
            # I can put zeros here at this stage
            dx = np.array([0])
            dt = np.array([1])
            dxdt = np.column_stack([dx, dt])
            dimszi = ['time', 'dxdtdim']
            coordi = {'time': [fdatedt], 'dxdtdim': np.arange(2)}
            newds[vv] = xarray.DataArray(data=dxdt, dims=dimszi, coords=coordi)
        elif vv in ['latlonSARcossin']:
            lat_sar_cossin = preprocess.conv_position(subset_ok['latSAR'])  # Gets cos and sin
            lon_sar_cossin = preprocess.conv_position(subset_ok['lonSAR'])
            latlon_sar_cossin = np.hstack([lat_sar_cossin, lon_sar_cossin])
            dimszi = ['time', 'latlondim']
            coordi = {'time': [fdatedt], 'latlondim': np.arange(4)}
            newds[vv] = xarray.DataArray(data=latlon_sar_cossin, dims=dimszi, coords=coordi)
        elif vv in ['todSAR']:
            dimszi = ['time']
            coordi = {'time': [fdatedt]}
            newds[vv] = xarray.DataArray(data=subset_ok['todSAR'], dims=dimszi, coords=coordi)
        elif vv in ['oswK']:
            dimszi = ['time', 'oswWavenumberBinSize']
            coordi = {'time': [fdatedt], 'oswWavenumberBinSize': np.arange(len(ks1))}
            newds[vv] = xarray.DataArray(data=ks1.reshape((1, len(ks1))), dims=dimszi, coords=coordi)
        elif vv in ['incidence', ]:
            dimszi = ['time', 'incdim']
            coordi = {'time': [fdatedt], 'incdim': np.arange(2)}
            incidence = preprocess.conv_incidence(ds['oswIncidenceAngle'].values.squeeze())
            newds[vv] = xarray.DataArray(data=incidence, dims=dimszi, coords=coordi)
        elif vv in ['incidence_angle']:
            dimszi = ['time']
            olddims = [x for x in ds['oswIncidenceAngle'].dims if x not in ['oswAzSize', 'oswRaSize']]
            coordi = {}
            for didi in olddims:
                coordi[didi] = ds['oswIncidenceAngle'].coords[didi].values
            coordi['time'] = [fdatedt]
            incidence = np.array([ds['oswIncidenceAngle'].values.squeeze()])
            newds[vv] = xarray.DataArray(data=incidence, dims=dimszi, coords=coordi)
        elif vv in ['satellite']:
            dimszi = ['time']
            coordi = {'time': [fdatedt]}
            satellite_int = np.array([satellite[2] == 'a']).astype(int)
            newds[vv] = xarray.DataArray(data=satellite_int, dims=dimszi, coords=coordi)
        elif vv in ['platformName']:
            dimszi = ['time']
            coordi = {'time': [fdatedt]}
            satellite_int = np.array([satellite])
            newds[vv] = xarray.DataArray(data=satellite_int, dims=dimszi, coords=coordi)
        elif vv in ['nrcs']:
            dimszi = ['time']
            coordi = {'time': [fdatedt]}
            newds[vv] = xarray.DataArray(data=s0.reshape((1,)), dims=dimszi, coords=coordi)
        elif vv in ['heading']:
            dimszi = ['time']
            coordi = {'time': [fdatedt]}
            newds[vv] = xarray.DataArray(data=ds['oswHeading'].values.reshape((1,)), dims=dimszi, coords=coordi)
        elif vv in ['nv']:
            dimszi = ['time']
            coordi = {'time': [fdatedt]}
            newds[vv] = xarray.DataArray(data=nv.reshape((1,)), dims=dimszi, coords=coordi)
        elif vv in ['oswQualityCrossSpectraRe', 'oswQualityCrossSpectraIm']:
            if vv == 'oswQualityCrossSpectraRe':
                datatmp = cspc_re
            elif vv == 'oswQualityCrossSpectraIm':
                datatmp = cspc_im
            else:
                raise Exception()

            coordi = dict()
            coordi['time'] = [fdatedt]
            coordi['oswAngularBinSize'] = np.arange(len(ths1))
            coordi['oswWavenumberBinSize'] = np.arange(len(ks1))
            dimsadd = ['time', 'oswAngularBinSize', 'oswWavenumberBinSize']
            if datatmp.shape == (72, 60):  # case only one spectra
                datatmp = datatmp.reshape((1, 72, 60))

            newds[vv] = xarray.DataArray(data=datatmp, dims=dimsadd, coords=coordi)
        else:
            datatmp = ds[vv].values.squeeze()
            olddims = [x for x in ds[vv].dims if x not in ['oswAzSize', 'oswRaSize']]
            coordi = dict()
            for didi in olddims:
                coordi[didi] = ds[vv].coords[didi].values
            coordi['time'] = [fdatedt]
            dimsadd = ['time']
            newds[vv] = xarray.DataArray(data=[datatmp], dims=dimsadd, coords=coordi)
    return newds
