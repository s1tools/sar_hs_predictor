import numpy as np
import logging
import xarray as xr
import os
import datetime
from s1tools.sarhspredictor.generator import DataGenerator
from s1tools.sarhspredictor.load_Hs_WV_keras_model import load_wv_model
from s1tools.sarhspredictor.preprocess import apply_normalisation
periods = {'before_WV2_update': (datetime.datetime(2019, 6, 27), # start of release IPF003.10 Optimized resampling of image cartesian cross-spectra
            datetime.datetime(2021, 6, 25)),
            'after_WV2_update':(datetime.datetime(2021, 6, 26),datetime.datetime.today())
        } # D-1 new EAP WV2 definitively adopted both S1A and S1B}
# S1A with new WV2 beam elevation antenna pattern EAP   28/02/2019  12/03/2019  EAP WV2 test
  # S1B with new WV2 beam elevation antenna pattern EAP 14/05/2019  28/05/2019  EAP WV2 test
test_periods = {
    'S1A':(datetime.datetime(2019,2,28),datetime.datetime(2019,3,12)),
    'S1B':(datetime.datetime(2019,5,14),datetime.datetime(2019,5,28)),

}
def compute_kernel(krg, kaz, kmin, kmax, Nk=4, Nphi=5, save_cwave_kernel=False):
    """
    Compute CWAVE kernels
    Args:
        krg (xarray.DataArray) : spectrum wavenumbers in range direction
        kaz (xarray.DataArray) : spectrum wavenumbers in azimuth direction

    Keywords Args:
        Nk (int) :
        Nphi (int) :
        save_cwave_kernel (bool, optional) : save CWAVE kernel on disk in working directory

    Return:
        (xarray.Dataset) : kernels
    """
    # Kernel Computation
    #

    coef = lambda nk: (nk + 3 / 2.0) / ((nk + 2.0) * (nk + 1.0))
    nu = lambda x: np.sqrt(1 - x ** 2.0)

    gamma = 2
    a1 = (gamma ** 2 - np.power(gamma, 4)) / (gamma ** 2 * kmin ** 2 - kmax ** 2)
    a2 = (kmax ** 2 - np.power(gamma, 4) * kmin ** 2) / (
        kmax ** 2 - gamma ** 2 * kmin ** 2
    )
    tmp = a1 * np.power(krg, 4) + a2 * krg ** 2 + kaz ** 2
    # alpha k
    alpha_k = (
        2
        * (
            (np.log10(np.sqrt(tmp)) - np.log10(kmin))
            / (np.log10(kmax) - np.log10(kmin))
        )
        - 1
    )
    # alpha phi
    alpha_phi = np.arctan2(kaz, krg).rename(None)
    # eta
    eta = np.sqrt((2.0 * tmp) / ((krg ** 2 + kaz ** 2) * tmp * np.log10(kmax / kmin)))

    Gnk = xr.combine_by_coords(
        [
            gegenbauer_polynoms(alpha_k, ik - 1, lbda=3 / 2.0)
            * coef(ik - 1)
            * nu(alpha_k).assign_coords({"k_gp": ik}).expand_dims("k_gp")
            for ik in np.arange(Nk) + 1
        ]
    )
    Fnphi = xr.combine_by_coords(
        [
            harmonic_functions(alpha_phi, iphi)
            .assign_coords({"phi_hf": iphi})
            .expand_dims("phi_hf")
            for iphi in np.arange(Nphi) + 1
        ]
    )

    Kernel = Gnk * Fnphi * eta
    Kernel.k_gp.attrs.update({"long_name": "Gegenbauer polynoms dimension"})
    Kernel.phi_hf.attrs.update(
        {"long_name": "Harmonic functions dimension (odd number)"}
    )

    _Kernel = Kernel.rename("cwave_kernel").to_dataset()
    if "pol" in _Kernel:
        _Kernel = _Kernel.drop_vars("pol")
    _Kernel["cwave_kernel"].attrs.update({"long_name": "CWAVE Kernel"})

    ds_G = Gnk.rename("Gegenbauer_polynoms").to_dataset()
    ds_F = Fnphi.rename("Harmonic_functions").to_dataset()
    ds_eta = eta.rename("eta").to_dataset()
    if "pol" in ds_G:
        ds_G = ds_G.drop_vars("pol")
        ds_F = ds_F.drop_vars("pol")
        ds_eta = ds_eta.drop_vars("pol")

    Kernel = xr.merge([_Kernel, ds_G, ds_F, ds_eta])

    if save_cwave_kernel:
        Kernel.to_netcdf("cwaves_kernel.nc")

    return Kernel


def gegenbauer_polynoms(x, nk, lbda=3 / 2.0):
    """

    Args:
        x: np.ndarray
        nk: int
        lbda: float

    Returns:
        Cnk : np.ndarray
    """
    C0 = 1
    if nk == 0:
        return C0 + x * 0
    C1 = 3 * x
    if nk == 1:
        return C1 + x * 0

    Cnk = (1 / nk) * (
        2 * x * (nk + lbda - 1) * gegenbauer_polynoms(x, nk - 1, lbda=lbda)
        - (nk + 2 * lbda - 2) * gegenbauer_polynoms(x, nk - 2, lbda=lbda)
    )
    Cnk = (1 / nk) * (
        2 * x * (nk + lbda - 1) * gegenbauer_polynoms(x, nk - 1, lbda=lbda)
        - (nk + 2 * lbda - 2) * gegenbauer_polynoms(x, nk - 2, lbda=lbda)
    )

    return Cnk


def harmonic_functions(x, nphi):
    """

    Args:
        x: np.ndarray
        nphi: int

    Returns:
        Fn : np.ndarray
    """
    if nphi == 1:
        Fn = np.sqrt(1 / np.pi) + x * 0
        return Fn

    # Even & Odd case
    if nphi % 2 == 0:
        Fn = np.sqrt(2 / np.pi) * np.sin((nphi) * x)
    else:
        Fn = np.sqrt(2 / np.pi) * np.cos((nphi - 1) * x)

    return Fn


def compute_cwave_parameters(
    xs,
    kmin=2 * np.pi / 600,
    kmax=2 * np.pi / 25,
    kx_varname="k_rg",
    ky_varname="k_az",
    dim_x="freq_sample",
    dim_y="freq_line",
    **kwargs
):
    """
    Compute CWAVE parameters.
    Args:
        xs (xarray.Dataset) : complex one sided xspectra
        kmin (float, optional) : minimum wavenumber used in CWAVE computation
        kmax (float, optional) : maximum wavenumber used in CWAVE computation

    Keywords Args:
        kwargs (): keyword arguments passed to compute_kernel
            save_cwave_kernel (bool, optional) : save CWAVE kernel on disk in working directory
            Nk (int, optional) : default is defined in compute_kernel
            Nphi (int, optional) : default is defined in compute_kernel

    Return:
        (xarray.DataArray) : cwave_parameters
    """

    # Cross-Spectra Frequency Filtering
    kk = np.sqrt((xs[kx_varname]) ** 2.0 + (xs[ky_varname]) ** 2.0)
    xxs = xs.where(
        (kk > kmin) & (kk < kmax)
    )  # drop=True here would remove other dimensions we may want to keep
    xxs = xxs.dropna(dim=dim_y, how="all").dropna(
        dim=dim_x, how="all"
    )  # removing only along the chosen dimensions

    # Cross-Spectra normalization
    xxsm = np.abs(xxs)
    dkrg = (
        xxs[kx_varname].diff(dim=dim_x).mean(dim=dim_x)
    )  # works even if there are other dimensions than expected
    dkaz = (
        xxs[ky_varname].diff(dim=dim_y).mean(dim=dim_y)
    )  # works even if there are other dimensions than expected
    xxsmn = xxsm / (xxsm.sum(dim=[dim_y, dim_x]) * dkrg * dkaz)

    # XS decomposition Kernel
    kernel = compute_kernel(
        krg=xxs[kx_varname], kaz=xxs[ky_varname], kmin=kmin, kmax=kmax, **kwargs
    )

    # CWAVE paremeters computation
    cwave_parameters = ((kernel.cwave_kernel * xxsmn) * dkrg * dkaz).sum(
        dim=[dim_x, dim_y], skipna=True, min_count=1
    )
    cwave_parameters.attrs.update({"long_name": "CWAVE parameters"})
    return cwave_parameters.rename("cwave_params")  # .to_dataset()
def get_cwaves(ds_ocn):
    """
    method to compute CWAVE parameters from oswCartSpec ...

    :param ds_ocn: xr.Dataset WV OCN ESA tweaked
    :return:
    """
    xs = ds_ocn["oswCartSpecRe"].T.squeeze() + 1j * ds_ocn["oswCartSpecIm"].T.squeeze()
    xs = xs.assign_coords({"kx": ds_ocn["oswKx_denorm"], "ky": ds_ocn["oswKy_denorm"]})
    cwaves = compute_cwave_parameters(
        xs,
        kmin=2 * np.pi / 625,
        kmax=2 * np.pi / 60,
        kx_varname="kx",
        ky_varname="ky",
        dim_x="oswKxBinSize",
        dim_y="oswKyBinSize",
    )  # 60 and 625 chosen to fit to what was done in Quach 2020

    ds_ocn = ds_ocn.assign({"NEWcwave": cwaves})
    ds_ocn["NEWcwave"].attrs[
        "description"
    ] = "cwave parameters compute from oswCartSpec Re+Im"
    ds_ocn["NEWcwave"].attrs["source_code"] = "xsarslc"
    return ds_ocn

def get_OCN_SAR_date(measurement_file_path) -> xr.Dataset:
    """

    :param measurement_file_path: str Sentinel-1 ESA OCN Level-2 WV measurement full path
    :return:
    """
    basename = os.path.basename(measurement_file_path)
    datesar64 = np.datetime64(
        datetime.datetime.strptime(basename.split("-")[5], "%Y%m%dt%H%M%S")
    )
    dssar = xr.open_dataset(measurement_file_path)
    dssar = dssar.assign(
        {
            "sardate": xr.DataArray(
                np.array([datesar64]).astype("<M8[ns]"),
                dims=["sardate"],
                name="sardate",
                attrs={"decsription": "start date given in ocn measurement name"},
            )
        }
    )
    assert "sardate" in dssar
    variables2drop = []
    for vv in dssar:
        if "owi" in vv or "rvl" in vv:
            variables2drop.append(vv)

    variables2drop += [
        "oswQualityCrossSpectraRe",
        "oswQualityCrossSpectraIm",
        "oswPolSpec",
        "oswPolSpecNV",
        "oswPartitions",
        "oswK",
        "oswPhi",
        "oswSpecRes",
    ]
    consolidate_vars_2drop = []
    for vv in variables2drop:
        if vv in dssar:
            consolidate_vars_2drop.append(vv)
    logging.debug(
        "%i variables to drop (rvl+owi) in SAR OCN dataset", len(consolidate_vars_2drop)
    )
    dssar = dssar.drop_vars(consolidate_vars_2drop)
    if len(dssar["oswLag"]) != 3: # note oswLag and oswCartSpecRe/Im have been introduced 2018-03-13 with IPF 02.90
        logging.warning(
            "file %s contains a oswLag with: %s", measurement_file_path, dssar["oswLag"]
        )
        dssar = None
    else:
        dssar = dssar.isel({"oswLag": 2, "oswAzSize": 0, "oswRaSize": 0})
        dssar["oswCartSpecRe"].attrs["tau_selected"] = "third oswLag (tau2 - tau0)"
        dssar["oswCartSpecIm"].attrs["tau_selected"] = "third oswLag (tau2 - tau0)"
        dssar = dssar.assign(
            {
                "mission": xr.DataArray(
                    [basename[0:3]],
                    dims=["sardate"],
                    attrs={"description": "SAR mission ID"},
                )
            }
        )

        dssar = dssar.assign(
            {
                "oswKx_denorm": dssar["oswKx"].squeeze()
                / dssar["oswGroundRngSize"].squeeze(),
                "oswKy_denorm": dssar["oswKy"].squeeze()
                / dssar["oswAziSize"].squeeze(),
            }
        )
        dssar = get_cwaves(ds_ocn=dssar)
    return dssar

def get_Hs_inference_from_CartesianXspectra(measurement_file_path)->xr.Dataset:
    """
    this method would start from a ocn wv measurement path to get the 2 variables to be added to OSW OCN component

    :param measurement_file_path: str full path S1 ocn wv measurement path
    :return:
    """
    ds_one_wv_imagette = get_OCN_SAR_date(measurement_file_path)
    logging.debug('cwave: %s',ds_one_wv_imagette["NEWcwave"])
    cw = ds_one_wv_imagette["NEWcwave"].values.reshape((20,))

    ds_one_wv_imagette_normed = apply_normalisation(ds_wv_ocn=ds_one_wv_imagette)
    # create numpy vectors expected as input of the architecture NN model
    x_hlf = np.stack(
            [
                ds_one_wv_imagette_normed["oswNrcs"].values,
                ds_one_wv_imagette_normed["oswNv"].values,
                ds_one_wv_imagette_normed["oswIncidenceAngle"].values,
                ds_one_wv_imagette_normed["oswHeading"].values,
                ds_one_wv_imagette_normed["oswSkew"].values,
                ds_one_wv_imagette_normed["oswKurt"].values
            ]
        ).T
    x_hlf = np.hstack([x_hlf, cw])
    x_hlf = x_hlf.reshape((1,x_hlf.size))
    logging.debug('x_hlf: %s',x_hlf.shape)
    x_spec =  ds_one_wv_imagette_normed["oswCartSpecRe"].values
    x_spec = x_spec.reshape((1,x_spec.shape[0],x_spec.shape[1]))
    logging.debug('x_spec: %s',x_spec.shape)
    # y_test = dstest["hs_alti_closest"].values
    targets = np.ones(len(x_hlf))*np.nan # for predictions having the Hs from altimeters is not mandatory
    targets = targets.reshape((1,targets.size))
    logging.debug('targets: %s',targets.shape)
    data_wv = DataGenerator(x_hlf=x_hlf,
                                        x_spectra=x_spec, y_set=targets, batch_size=128)
    logging.debug('sardate: %s',ds_one_wv_imagette['sardate'])
    if np.datetime64(periods['before_WV2_update'][0])>=ds_one_wv_imagette['sardate'] and np.datetime64(periods[''][1]<ds_one_wv_imagette['sardate']):
        sat_acronym = os.path.basename(measurement_file_path).split('-')[0].upper() # S1A or S1B or ..
        if np.datetime64(test_periods[sat_acronym][0])>=ds_one_wv_imagette['sardate'] and np.datetime64(test_periods[sat_acronym][1])<ds_one_wv_imagette['sardate']:
            tag = 'hs_wv_model_after_WV2_EAP'
        else:
            tag = 'hs_wv_model_before_WV2_EAP'

    else:
        tag = 'hs_wv_model_after_WV2_EAP'
    logging.debug('tag : %s',tag)
    model_wv = load_wv_model(model_tag=tag)
    yhat = model_wv.predict(data_wv)[0] # 2 columns Hs and HsStdev
    logging.debug('yhat : %s',yhat)
    # optional part to declare the variable in OSW component
    ds = xr.Dataset()
    ds['oswTotalHs'] = xr.DataArray(name='oswTotalHs',data=np.array([yhat[0]]).reshape(1,1),dims=['oswRaSize','oswAzSize'],
        attrs={'long_name':"Total significant wave height"})
    ds['oswTotalHsStdev'] = xr.DataArray(name='oswTotalHsStdev',data=np.array([yhat[1]]).reshape(1,1),dims=['oswRaSize','oswAzSize'],
        attrs={'long_name':"Standard deviation of total significant wave height"})
    return ds


def test_a_prediction_wv():
    import argparse,time,s1tools
    parser = argparse.ArgumentParser(description='Hs-WV-inference')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--wvpath', required=True, help='S1 WV OCN full path')
    parser.add_argument('--version', action='version',
                                                version='%(prog)s {version}'.format(version=s1tools.__version__))
    args = parser.parse_args()
    fmt = '%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s'
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S',force=True)
    else:
        logging.basicConfig(level=logging.INFO, format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S',force=True)
    t0 =time.time()

    ds_hs_wv = get_Hs_inference_from_CartesianXspectra(measurement_file_path=args.wvpath)
    logging.info('ds_hs_wv : %s',ds_hs_wv)

    logging.info('done in %1.3f min', (time.time() - t0) / 60.)