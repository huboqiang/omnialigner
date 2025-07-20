from typing import Dict
import yaml
from omnialigner.dtypes import ConfigFile

def load_config(config_info: ConfigFile|Dict, **kwargs) -> Dict:
    """
    load config info from yaml file or dict

    Args:
        config_info (ConfigFile|Dict): the config file or dict

    Returns:
        dict: color dict
    """
    if isinstance(config_info, Dict):
        return config_info

    with open(config_info, 'r') as f:
        template_string = f.read()
        data = yaml.load(template_string, Loader=yaml.FullLoader)
        return data