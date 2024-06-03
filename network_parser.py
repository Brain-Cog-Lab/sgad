import yaml


class Parse(object):
    def __init__(self, path):
        with open(path, 'r') as f:
            self.parameters = yaml.safe_load(f)

    def __getitem__(self, key):
        return self.parameters[key]

    def save(self, filename):
        with open(filename, 'w') as f:
            yaml.dump(self.parameters, f)
