from s1tools.sarhspredictor.heteroskedastic import gaussian_nll, gaussian_mse

custom_objects = {"Gaussian_NLL": gaussian_nll, "Gaussian_MSE": gaussian_mse}
from s1tools.sarhspredictor.utils import load_config
from tensorflow.keras.models import load_model
import tensorflow
import os
import s1tools


def load_wv_model(model_tag="hs_wv_model_before_WV2_EAP", config_path=None) -> tensorflow.keras.models:
    """

    :param model_tag: str hs_wv_model_before_WV2_EAP or hs_wv_model_after_WV2_EAP
    :return:
        modelNN : tensorflow.keras.models
    """
    config, config_path = load_config(config_path)
    path_model = os.path.abspath(os.path.join(config_path, config[model_tag]))
    modelNN = load_model(path_model, custom_objects=custom_objects)
    return modelNN
