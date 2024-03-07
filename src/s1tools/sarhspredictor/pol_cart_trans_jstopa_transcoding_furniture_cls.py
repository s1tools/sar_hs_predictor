# coding: utf-8
"""
context: compute Hs total from SAR data using parameters obtained from neural network training with WW3
date: 11 Oct 2018
author: Justin Stopa transcoded from matlab to python by Antoine Grouazel
"""
import numpy as np
from scipy.interpolate import griddata
from scipy.stats import mode


def pol_cart_trans(d, k, t, x, y, name='re', interpmethod='cubic'):
    """
    convert k,t spectrum into cartesian

    original matlab code: Justin Stopa 09/06/2016

    :param d: spectra in polar coordinates of (k,t)
    :param k: wave number in log space
    :param t: theta direction in radians
    :param x: transform spc into cartesian with these x wavenumbers (output grid)
    :param y: transform spc into cartesian with these y wavenumbers (output grid)
    :param name: unused
    :param interpmethod:
    :return:
        D: 71*85 nd array matrix: cartesian cross spectra
        Dbefore: 71*85 nd array matrix: cartesian cross spectra without energy normalization (conversation)
    """
    d = d.astype(np.float64)
    kmax = np.amax(k)  # maximum wavenumber
    kmin = np.amin(k)  # minimum wavenumber
    kmin = np.double(kmin)
    kmax = np.double(kmax)
    first_term = np.power((kmax / kmin), (1. / (len(k) - 1)))
    second_term = -1. / np.power((kmax / kmin), (1. / (len(k) - 1)))
    term_multi = first_term + second_term
    term_multi = np.double(term_multi)

    modal_value, count_value = mode(np.diff(t))
    modal_value = modal_value[0]
    modal_value = np.float64(modal_value)
    a = np.float64(0.5) * modal_value * term_multi * k ** 2
    a = a.astype(np.float64)
    k = k.astype(np.float64)

    #  Make matrix of output cartesian points
    X = np.tile(x, [len(y), 1]).squeeze().T
    Y = np.tile(y, [len(x), 1]).squeeze()

    # dx and dy of output cartesian grid
    dx, _ = mode(np.diff(x))
    dy, _ = mode(np.diff(y))

    # convert polar grid to cartesian grid
    kx = (np.tile(k, [len(t), 1]).T * np.tile(np.cos(t), [len(k), 1]))
    ky = (np.tile(k, [len(t), 1]).T * np.tile(np.sin(t), [len(k), 1]))

    pts = []
    for xx in range(kx.shape[0]):
        for yy in range(kx.shape[1]):
            pts.append((kx[xx, yy], ky[xx, yy]))
    pts = np.array(pts)

    new_pts = []
    for xx in range(X.shape[0]):
        for yy in range(X.shape[1]):
            new_pts.append((X[xx, yy], Y[xx, yy]))
    new_pts = np.array(new_pts)

    D = griddata(points=pts, values=d.ravel(), xi=new_pts, method=interpmethod,
                 rescale=False)  # here I use the scipy linear instead of v4 matlab option
    # nearest = 25 diff Re #meilleur result sur D
    # linear max diff 38 Re
    # cubic D max diff 28 Re
    D = np.reshape(D, X.shape, order='A')

    D = D.astype(np.float64)
    dx = dx.astype(np.float64)
    dy = dy.astype(np.float64)
    eng_car = 4.0 * np.sqrt(np.sum(np.sum(abs(D) * dx * dy)))

    eng_pol = 4.0 * np.sqrt(np.sum(np.sum(abs(d) * np.tile(a, [len(t), 1]).T)))
    # Conserve energy
    Dbefore = D.copy()
    D = D * (eng_pol / eng_car) ** 2
    return D, Dbefore
