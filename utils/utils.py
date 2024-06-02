import json

def load_config(config):
    if isinstance(config, str):
        # If config is a path to a JSON file, load it
        with open(config, 'r') as f:
            config_dict = json.load(f)
    elif isinstance(config, dict):
        # If config is already a dictionary, use it directly
        config_dict = config
    else:
        raise ValueError("Invalid configuration format. Must be a dictionary or path to a JSON file.")

    return config_dict
