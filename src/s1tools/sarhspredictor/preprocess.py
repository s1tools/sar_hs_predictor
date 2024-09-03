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


variables_2_norm = [
    "oswCartSpecRe",
    # "oswCartSpecIm", # useless after tests
    # "oswNrcs", # already quite packed -20dB to 0dB
    "oswAzCutoff",
    # "oswNv", # already quite packed -20dB to 0dB
    "oswHeading",
    "oswIncidenceAngle",
    # "NEWcwave", # already quite packed -20dB to 0dB
]


def apply_normalisation(ds_wv_ocn):
    """

    :param ds_wv_ocn: xr.Dataset for one WV imagette
    :return:
    """
    for ii in range(len(variables_2_norm)):
        variable = variables_2_norm[ii]
        da = ds_wv_ocn[variable]
        if variable == "oswIncidenceAngle":
            valuess2 = normIncidence(da)
        elif variable == "oswHeading":
            valuess2 = normHeading(da)
        elif variable == "oswAzCutoff":
            valuess2 = normAzCutOff(da)
        elif variable == "oswCartSpecRe":
            valuess2 = normCartSpecRe(da)
        elif variable == "oswCartSpecIm":
            valuess2 = normCartSpecIm(da)
        else:
            raise Exception("variable: %s not expected" % variable)
        ds_wv_ocn = ds_wv_ocn.assign({variable: valuess2})
    return ds_wv_ocn
