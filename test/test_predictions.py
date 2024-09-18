import pytest
import os
import pickle
import s1tools
from s1tools.sarhspredictor.load_hs_wv_keras_model import load_wv_model
import numpy as np
from s1tools.sarhspredictor import generator


def test_hs_predictions_before_WV2_EAP():
    input_test_dataset = os.path.join(
        os.path.dirname(s1tools.sarhspredictor.__file__),
        "referencedata",
        "S1_WV_OCN_reference_test_dataset_Hs_model_cartSpec_before_WV2_EAP.pkl",
    )
    fid = open(input_test_dataset, "rb")
    data = pickle.load(fid)
    fid.close()
    model = load_wv_model(model_tag="hs_wv_model_before_WV2_EAP")
    test = generator.DataGenerator(
        x_hlf=data["x_test_hlf"],
        x_spectra=data["x_test_spec"],
        y_set=data["y_test"],
        batch_size=128,
    )
    actual_predictions = model.predict(test)
    expected_predictions = data["yhat"]
    # test Hs
    assert np.allclose(actual_predictions[:, 0], expected_predictions[:, 0])
    # test Hs STD
    assert np.allclose(actual_predictions[:, 1], expected_predictions[:, 1])


def test_hs_predictions_after_WV2_EAP():
    input_test_dataset = os.path.join(
        os.path.dirname(s1tools.sarhspredictor.__file__),
        "referencedata",
        "S1_WV_OCN_reference_test_dataset_Hs_model_cartSpec_after_WV2_EAP.pkl",
    )
    fid = open(input_test_dataset, "rb")
    data = pickle.load(fid)
    fid.close()
    model = load_wv_model(model_tag="hs_wv_model_after_WV2_EAP")
    test = generator.DataGenerator(
        x_hlf=data["x_test_hlf"],
        x_spectra=data["x_test_spec"],
        y_set=data["y_test"],
        batch_size=128,
    )
    actual_predictions = model.predict(test)
    expected_predictions = data["yhat"]
    # test Hs
    assert np.allclose(actual_predictions[:, 0], expected_predictions[:, 0])
    # test Hs STD
    assert np.allclose(actual_predictions[:, 1], expected_predictions[:, 1])
