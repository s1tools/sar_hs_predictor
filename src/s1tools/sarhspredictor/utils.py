import os
import logging
import yaml
import s1tools


def load_config(config_path=None):
    """

    Returns:
        conf: dict
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__),
            ".."," ..", ".."
            "config.yaml"
            )

        config_path = os.environ.get('SAR_HS_PREDICTOR_CONFIG_YAML_PATH', config_path)

    logging.info("config path: %s", config_path)
    stream = open(config_path, "r")
    conf = yaml.load(stream, Loader=yaml.CLoader)
    return conf, config_path
