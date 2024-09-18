from s1tools.sarhspredictor.constant import normalisation_factors


def normIncidence(inc):
    return inc / normalisation_factors["incidence_angle"]


def normHeading(heading):
    return heading / normalisation_factors["heading_angle"]


def normAzCutOff(cutoff):
    return cutoff / normalisation_factors["azimuth_cutoff"]


def normCartSpecRe(spec):
    # assert spec.shape==(185,128)
    return spec / normalisation_factors["oswCartSpecRe"]


def normCartSpecIm(spec):
    # assert spec.shape==(185,128)
    return spec / normalisation_factors["oswCartSpecIm"]


def apply_normalisation(ds_wv_ocn):
    """

    :param ds_wv_ocn: xr.Dataset for one WV imagette
    :return:
    """
    variable_norm = {
        "oswIncidenceAngle": normIncidence,
        "oswHeading": normHeading,
        "oswAzCutoff": normAzCutOff,
        "oswCartSpecRe": normCartSpecRe,
        # "oswCartSpecIm", # useless after tests
        # "oswNrcs", # already quite packed -20dB to 0dB
        # "oswNv", # already quite packed -20dB to 0dB
        # "NEWcwave", # already quite packed -20dB to 0dB
   }

    for variable, norm in variable_norm.items():
        da = ds_wv_ocn[variable]
        ds_wv_ocn = ds_wv_ocn.assign({variable: norm(da)})
    return ds_wv_ocn
