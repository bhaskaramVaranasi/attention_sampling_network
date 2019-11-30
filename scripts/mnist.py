import json
from os import path

from skimage.io import imsave

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


class AttentionSaver():
    def __init__(self, output, att_model, data):
        self._att_path = path.join(output, "attention_{:03d}.png")
        self._patches_path = path.join(output, "patches_{:03d}_{:03d}.png")
        self._att_model = att_model
        (self._x, self._x_high), self._y = data[0]
        self._imsave(
            path.join(output, "image.png"),
            self._x[0, :, :, 0]
        )

    def on_epoch_end(self, e, logs):
        att, patches = self._att_model.predict([self._x, self._x_high])
        self._imsave(self._att_path.format(e), att[0])
        np.save(self._att_path.format(e)[:-4], att[0])
        for i, p in enumerate(patches[0]):
            self._imsave(self._patches_path.format(e, i), p[:, :, 0])

    def _imsave(self, filepath, x):
        x = (x*65535).astype(np.uint16)
        imsave(filepath, x, check_contrast=False)