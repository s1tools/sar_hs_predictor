# coding: utf-8
import os

from tensorflow.keras.models import load_model

from .heteroskedastic import gaussian_nll, gaussian_mse

CURDIRSCRIPT = os.path.dirname(__file__)


def load_quach2020_model_v2(model_file):
    """
    based on the example notebook provided by P. Sadowsky: predict.ipynb
    :return:
    """
    custom_objects = {'Gaussian_NLL': gaussian_nll, 'Gaussian_MSE': gaussian_mse}
    model = load_model(model_file, custom_objects=custom_objects)
    return model
