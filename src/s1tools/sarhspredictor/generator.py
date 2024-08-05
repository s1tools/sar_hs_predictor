from tensorflow.keras.utils import Sequence
import numpy as np

class DataGenerator(Sequence):
    def __init__(self, x_hlf, x_spectra, y_set, batch_size=128):
        self.x_hlf, self.x_spectra, self.y = x_hlf, x_spectra, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.y) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_hlf = self.x_hlf[idx * self.batch_size : (idx + 1) * self.batch_size, :]
        batch_spec = self.x_spectra[
            idx * self.batch_size : (idx + 1) * self.batch_size, :, :
        ]
        batch_x = [batch_spec,batch_hlf]
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        # print("batch_hlf", batch_hlf.shape)
        # print("batch_spec", batch_spec.shape)
        # print("batch_y", batch_y.shape)
        return batch_x, batch_y

