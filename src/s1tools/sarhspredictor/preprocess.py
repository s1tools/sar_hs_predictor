# Define utility functions for preprocessing SAR data.
import numpy as np


def _conv_deg(in_angle, is_inverse=False, in_cos=None, in_sin=None):
    """
    Converts measurements in degrees (e.g. angles), using encoding proposed
    at https://stats.stackexchange.com/a/218547
    Encode each angle as tuple theta as tuple (cos(theta), sin(theta)),
    for justification, see graph at bottom
    Args:
        in_angle: measurement of lat/ long in degrees
    Returns:
        tuple of values between -1 and 1
    """
    if is_inverse:
        return np.sign(np.rad2deg(np.arcsin(in_sin))) * np.rad2deg(np.arccos(in_cos))

    angle = np.deg2rad(in_angle)
    return np.cos(angle), np.sin(angle)


def conv_real(x):
    """Scales real part of spectrum.
    Args:
        x: numpy array of shape (notebooks, 72, 60)
    Returns:
        scaled
    """
    assert len(x.shape) == 3
    assert x.shape[1:] == (72, 60)
    x = (x - 8.930369) / 41.090652
    return x


def conv_imaginary(x):
    """Scales imaginary part of spectrum.
    Args:
        x: numpy array of shape (notebooks, 72, 60)
    Returns:
        scaled
    """
    assert len(x.shape) == 3
    assert x.shape[1:] == (72, 60)
    x = (x - 4.878463e-08) / 6.4714637
    return x


def median_fill(x, extremum=1e+15):
    """
    Inplace median fill.
    Args:
    x: numpy array of shape (notebooks, features)
    extremum: threshold for abs value of x. Damn Netcdf fills in nan values with 9.96921e+36.
    Returns:
    rval: new array with extreme values filled with median.
    """
    # assert not np.any(np.isnan(x)) #commented by agrouaze Feb 2021
    medians = np.median(x, axis=0)
    mask = np.abs(x) > extremum
    medians = np.repeat(medians.reshape(1, -1), x.shape[0], axis=0)
    assert medians.shape == x.shape, medians.shape
    x[mask] = medians[mask]  # TODO: MODIFIES x, so this is unsafe.
    return x


def conv_cwave(x):
    """
    Scale 22 cwave features. These were precomputed using following script.
    
    from sklearn import preprocessing
    with h5py.File('aggregate_ALT.h5', 'r') as fs:
        cwave = np.hstack([fs['S'][:], fs['sigma0'][:].reshape(-1,1), fs['normalizedVariance'][:].reshape(-1,1)])
        cwave = scripts.median_fill(cwave) # Needed to remove netcdf nan-filling.
        s_scaler = preprocessing.StandardScaler()
        s_scaler.fit(cwave) # Need to fit to full data.
        print(s_scaler.mean_, s_scaler.v)
    
    """
    # Fill in extreme values with medians.
    x = median_fill(x)

    means = np.array([8.83988852e+00, 9.81496891e-01, 2.04964720e+00, 1.05590932e-01,
                      -6.00710228e+00, 2.54775182e+00, -5.76860655e-01, 2.09000078e+00,
                      -8.44825896e-02, 8.90420253e-01, -1.44932907e+00, -6.79597846e-01,
                      1.03999407e+00, -2.09475628e-01, 2.76214306e+00, -6.35718150e-03,
                      -8.09685487e-01, 1.41905445e+00, -1.85369068e-01, 3.00262098e+00,
                      -1.06865787e+01, 1.33246124e+00])

    vars = np.array([9.95290027, 35.2916408, 8.509233, 10.62053105, 10.72524569,
                     5.17027335, 7.04256618, 3.03664677, 3.72031389, 5.92399639,
                     5.31929415, 8.26357553, 1.95032647, 3.13670466, 3.06597742,
                     8.8505963, 13.82242244, 1.43053089, 1.96215081, 11.71571483,
                     27.14579017, 0.05681891])

    x = (x - means) / np.sqrt(vars)
    return x


def conv_position(angle):
    """
    Return cosine and sine to latitute/longitude feature.
    """
    coord_transf = np.vectorize(_conv_deg)
    cos, sin = coord_transf(angle)
    return np.column_stack([cos, sin])


def conv_incidence(incidence_angle):
    """
    Return two features describing scaled incidence angle and 
    the wave mode label (0 or 1). Wave mode is 1 if angle is > 30 deg.
    """
    incidence_angle[incidence_angle > 90] = 30
    lbl = np.array(incidence_angle > 30, dtype='float32')
    incidence_angle = incidence_angle / 30.
    return np.column_stack([incidence_angle, lbl])
