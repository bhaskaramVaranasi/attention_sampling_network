import json
from os import path

import numpy as np

class MNIST():
    """Load a Megapixel MNIST dataset."""
    def __init__(self, dataset_dir, train=True):
        """Passing the directory name to the JSON parameters, to load th parameters."""
        with open(path.join(dataset_dir, "parameters.json")) as f:
            self.parameters = json.load(f)

        filename = "train.npy" if train else "test.npy"
        N = self.parameters["n_train" if train else "n_test"]
        W = self.parameters["width"]
        H = self.parameters["height"]
        scale = self.parameters["scale"]

        self._high_shape = (H, W, 1)
        self._low_shape = (int(scale*H), int(scale*W), 1)
        self._data = np.load(path.join(dataset_dir, filename),allow_pickle=True)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError()

        # Placeholders
        x_low = np.zeros(self._low_shape, dtype=np.float32).ravel()
        x_high = np.zeros(self._high_shape, dtype=np.float32).ravel()

        # Fill the sparse representations
        data = self._data[i]
        x_low[data[0][0]] = data[0][1]
        x_high[data[1][0]] = data[1][1]

        # Reshape to their final shape
        x_low = x_low.reshape(self._low_shape)
        x_high = x_high.reshape(self._high_shape)

        return [x_low, x_high], data[2]

