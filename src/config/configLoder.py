import yaml

class ConfigLoder :
    def __init__(self):
        pass

    def load_config(self):
        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)
        return config