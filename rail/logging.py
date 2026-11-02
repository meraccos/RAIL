import numpy as np


class DictLogger:
    def __init__(self):
        self.clean()

    def clean(self):
        self.log = {}

    def add(self, key, value):
        if key not in self.log.keys():
            self.log[key] = []
        self.log[key].append(value)

    def dump(self, prefix=None):
        dump = {}
        for key, value in self.log.items():
            full_key = f"{prefix}{key}" if prefix else key
            dump[full_key] = np.mean(value)
        return dump
