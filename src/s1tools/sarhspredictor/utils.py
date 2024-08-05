import os
import logging
import yaml
import s1tools
def load_config():
    """

    Returns:
        conf: dict
    """
    local_config_path = os.path.join(os.path.dirname(s1tools.__file__),'sarhspredictor', 'localconfig.yaml')

    if os.path.exists(local_config_path):
        config_path = local_config_path
    else:
        config_path = os.path.join(os.path.dirname(s1tools.__file__),'sarhspredictor', 'config.yaml')

    logging.info('config path: %s', config_path)
    stream = open(config_path, 'r')
    conf = yaml.load(stream, Loader=yaml.CLoader)
    return conf