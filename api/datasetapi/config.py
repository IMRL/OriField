import json
import yaml
from permissive_dict import PermissiveDict as Dict


def get_config_from_json(json_file):
    """
    Get the config from a json file
    Input:
        - json_file: json configuration file
    Return:
        - config: namespace
        - config_dict: dictionary
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Dict(config_dict)

    return config, config_dict


def get_config_from_yaml(yaml_file):
    """
    Get the config from yaml file
    Input:
        - yaml_file: yaml configuration file
    Return:
        - config: namespace
        - config_dict: dictionary
    """

    with open(yaml_file) as fp:
        if hasattr(yaml, 'FullLoader'):
            config_dict = yaml.load(fp, Loader=yaml.FullLoader)
        else:
            config_dict = yaml.load(fp)

    # convert the dictionary to a namespace using bunch lib
    config = Dict(config_dict)
    return config, config_dict


def merge_new_config(config, new_config):
    if '_base_' in new_config:
        with open(new_config['_base_'], 'r') as f:
            if hasattr(yaml, 'FullLoader'):
                yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            else:
                yaml_config = yaml.load(f)
            print(yaml_config)
        config.update(Dict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = Dict()
        merge_new_config(config[key], val)

    return config


def cfg_from_file(config_file):
    if config_file.endswith('json'):
        new_config, _ = get_config_from_json(config_file)
    elif config_file.endswith('yaml'):
        new_config, _ = get_config_from_yaml(config_file)
    else:
        raise Exception("Only .json and .yaml are supported!")

    config = Dict()
    merge_new_config(config=config, new_config=new_config)
    return config
