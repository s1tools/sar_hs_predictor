from s1tools.sarhspredictor.heteroskedastic import gaussian_nll, gaussian_mse

custom_objects = {"Gaussian_NLL": gaussian_nll, "Gaussian_MSE": gaussian_mse}
from s1tools.sarhspredictor.utils import load_config
from tensorflow.keras.models import load_model
import tensorflow
import os
import s1tools

config = load_config()


def load_wv_model(model_tag="hs_wv_model_before_WV2_EAP") -> tensorflow.keras.models:
    """

    :param model_tag: str hs_wv_model_before_WV2_EAP or hs_wv_model_after_WV2_EAP
    :return:
        modelNN : tensorflow.keras.models
    """
    path_model = config[model_tag]
    if "./" in path_model:
        path_model = os.path.join(os.path.dirname(s1tools.__file__), "sarhspredictor", path_model)
    modelNN = load_model(path_model, custom_objects=custom_objects)
    return modelNN
