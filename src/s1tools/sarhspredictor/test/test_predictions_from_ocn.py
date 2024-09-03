import pytest
import os
import s1tools
from s1tools.sarhspredictor.predict_hs_from_wv_ocn_cartesian_xspec import (
    get_Hs_inference_from_CartesianXspectra,
)
import numpy as np

expected = {"hs": 12.17, "hsstd": 1.473}


def test_hs_predictions_before_WV2_EAP():
    # matching model S1_WV_oswCartSpec_hs_NN_model_05BeforeEAP.15_bestmodel_checkpoint.h5
    input_test_dataset = os.path.join(
        os.path.dirname(s1tools.sarhspredictor.__file__),
        "referencedata",
        "S1A_WV_OCN__2SSV_20231003T122250_20231003T124111_050600_061886_3197.SAFE",
        "measurement",
        "s1a-wv2-ocn-vv-20231003t122403-20231003t122406-050600-061886-006.nc",
    )
    actual_ds_hs_predictions = get_Hs_inference_from_CartesianXspectra(input_test_dataset)
    print("ds_hs_predictions", actual_ds_hs_predictions)
    # expected_predictions = data['yhat']
    # test Hs
    assert np.allclose(actual_ds_hs_predictions["oswTotalHs"].values, expected["hs"], atol=1e-02)
    # test Hs STD
    assert np.allclose(actual_ds_hs_predictions["oswTotalHsStdev"], expected["hsstd"], atol=1e-02)


if __name__ == "__main__":
    test_hs_predictions_before_WV2_EAP()
