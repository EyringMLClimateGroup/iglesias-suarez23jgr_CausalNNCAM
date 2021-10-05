from pathlib import Path

from .cbrain.data_generator import DataGenerator
from .cbrain.utils import load_pickle


def build_train_generator(input_vars_dict, output_vars_dict, setup):
    out_scale_dict = load_pickle(
        Path(setup.out_scale_dict_folder, setup.out_scale_dict_fn)
    )
    input_transform = (setup.input_sub, setup.input_div)

    train_gen = DataGenerator(
        data_fn=Path(setup.train_data_folder, setup.train_data_fn),
        input_vars_dict=input_vars_dict,
        output_vars_dict=output_vars_dict,
        norm_fn=Path(setup.normalization_folder, setup.normalization_fn),
        input_transform=input_transform,
        output_transform=out_scale_dict,
        batch_size=setup.batch_size,
        shuffle=True,  # This feature doesn't seem to work
    )
    return train_gen


def build_valid_generator(input_vars_dict, output_vars_dict, setup, nlat=64, nlon=128, test=False):
    out_scale_dict = load_pickle(
        Path(setup.out_scale_dict_folder, setup.out_scale_dict_fn)
    )
    input_transform = (setup.input_sub, setup.input_div)
    if test:
        data_fn = setup.test_data_folder
        filenm  = setup.test_data_fn
    else:
        data_fn = setup.train_data_folder
        filenm  = setup.valid_data_fn
    
    ngeo = nlat * nlon
    valid_gen = DataGenerator(
        data_fn=Path(setup.train_data_folder, setup.valid_data_fn),
        input_vars_dict=input_vars_dict,
        output_vars_dict=output_vars_dict,
        norm_fn=Path(setup.normalization_folder, setup.normalization_fn),
        input_transform=input_transform,
        output_transform=out_scale_dict,
        batch_size=ngeo,
        shuffle=False,
        # xarray=True,
    )
    return valid_gen
