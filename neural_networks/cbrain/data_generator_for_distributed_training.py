"""
Data generator class.

Created on 2019-01-28-10-39
Author: Stephan Rasp, raspstephan@gmail.com
"""

import tensorflow as tf
import xarray as xr
import numpy as np
import h5py
from .utils import return_var_idxs
from .normalization import InputNormalizer, DictNormalizer, Normalizer
import horovod.tensorflow.keras as hvd


class DataGenerator(tf.keras.utils.Sequence):
    """
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

    Data generator class.
    """

    def __init__(
        self,
        data_fn,
        input_vars_dict,
        output_vars_dict,
        norm_fn=None,
        input_transform=None,
        output_transform=None,
        batch_size=1024,
        shuffle=True,
        xarray=False,
    ):
        # Just copy over the attributes
        self.data_fn, self.norm_fn = data_fn, norm_fn
        self.input_vars_dict = input_vars_dict
        self.output_vars_dict = output_vars_dict
        self.input_vars = list(input_vars_dict.keys())
        self.output_vars = list(output_vars_dict.keys())
        self.batch_size, self.shuffle = batch_size, shuffle
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.xarray = xarray

    def __len__(self):
        return self.n_batches_per_worker
    
    def __getitem__(self, index):
        # Compute start and end indices for batch
        worker_id = self.n_batches_per_worker * self.worker
        start_idx = (worker_id + index) * self.batch_size
        end_idx = start_idx + self.batch_size

        # Grab batch from data
        batch = self.data_ds["vars"][start_idx:end_idx]

        # Split into inputs and outputs
        X = batch[:, self.input_idxs]
        Y = batch[:, self.output_idxs]

        # Normalize
        X = self.input_transform.transform(X)
        Y = self.output_transform.transform(Y)

        return X, Y

    def on_epoch_end(self):
        self.indices = np.arange(self.n_batches)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __enter__(self):
        # Open datasets
        self.data_ds = xr.open_dataset(self.data_fn)
        if self.norm_fn is not None:
            self.norm_ds = xr.open_dataset(self.norm_fn)

        # Compute number of samples and batches
        self.n_samples = self.data_ds.vars.shape[0]
        self.n_batches = int(np.floor(self.n_samples) / self.batch_size)
        self.n_batches_per_worker = int( self.n_batches / hvd.size() )
        self.worker = hvd.rank()
        
        # Get input and output variable indices
        self.input_idxs = return_var_idxs(self.data_ds, self.input_vars_dict)
        self.output_idxs = return_var_idxs(self.data_ds, self.output_vars_dict)
        self.n_inputs, self.n_outputs = len(self.input_idxs), len(self.output_idxs)

        # Initialize input and output normalizers/transformers
        if self.input_transform is None:
            self.input_transform = Normalizer()
        elif type(self.input_transform) is tuple:
            self.input_transform = InputNormalizer(
                self.norm_ds,
                self.input_vars_dict,
                self.input_transform[0],
                self.input_transform[1],
                # var_cut_off,
            )
        else:
            self.input_transform = (
                self.input_transform  # Assume an initialized normalizer is passed
            )

        if self.output_transform is None:
            self.output_transform = Normalizer()
        elif type(self.output_transform) is dict:
            self.output_transform = DictNormalizer(
                self.norm_ds, self.output_vars_dict, self.output_transform
            )
        else:
            self.output_transform = (
                self.output_transform  # Assume an initialized normalizer is passed
            )

        # Now close the xarray file and load it as an h5 file instead
        # This significantly speeds up the reading of the data...
        if not self.xarray:
            self.data_ds.close()
            self.data_ds = h5py.File(self.data_fn, "r")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.data_ds.close()
        except:
            pass
        try:
            self.norm_ds.close()
        except:
            pass
