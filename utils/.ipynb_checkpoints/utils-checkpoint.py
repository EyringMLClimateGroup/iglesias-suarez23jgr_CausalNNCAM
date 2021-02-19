import numpy    as np
from pathlib    import Path
from .constants import FILENAME_PATTERN, OUTPUT_FILE_PATTERN
from netCDF4    import Dataset

#########################
#    Find value utils
#########################

def find_closest_value(values, target):
    """
    Returns the index of the closest value to the target
    in an array of values. Assumes values are numerical.
    """
    
    differences = np.array(values) - target
    idx = abs(differences).argmin()
    return idx


def find_closest_longitude(longitudes, target):
    """Converts longitudes to 0-360 and returns the index to the closest"""
    
    target = (target+360)%360
    return find_closest_value(longitudes, target)


#########################
#    Load ancil utils
#########################
def read_ancilaries(path):
    with Dataset(path, 'r') as nc_ancil:
        levels     = nc_ancil.variables['lev'][:]
        latitudes  = nc_ancil.variables['lat'][:]
        longitudes = nc_ancil.variables['lon'][:]
    return levels, latitudes, longitudes


#########################
#    Load data utils
#########################

# TODO? Use a dataset instead of this class
class VarData:
    """
    Helper class to store the variable data and metadata
    If variab
    """
    
    def __init__(self, variable, data, level = None):
        self.variable = variable
        self.level = level
        self.data = data
        if level is None:
            self.name = variable.name
        else:
            self.name = f"{variable.name}-{level}"

# def read_spcam(var_name, experiment, path):
#     filename = Path(path, FILENAME_PATTERN.format(var_name, experiment))
#     with Dataset(filename, 'r') as file:
#         data = file.variables[var_name][:]
#     return data
def read_spcam_file(path, var_name):
    with Dataset(path, 'r') as file:
        data = file.variables[var_name][:]
    return data

# def normalize(values):
#     return (values - np.mean(values))/ np.std(values, ddof=1)
def normalize(values):
    anom = values - np.mean(values)
    std = np.std(anom, ddof=1)
    if std != 0:
        return anom/std
    else:
        return values

# This code has sometimes peaks that exceeded the maximum memory of 2.5 GB.
# However, it's only temporarily, as after normalization the space occupied for
# a single cell is small.
# May it be possible to improve the retrieval so it doesn't load the full file?
# def get_normalized_data(
#     variable, experiment, path, idx_levs, idx_lats, idx_lons):
#     """
#     Returns a list of VarData, so both 2d and 3d can be treated the same
#     """
#     data = read_spcam(variable.name, experiment, path)
#     norm_data = list()
#     if variable.dimensions == 3:
#         for target_lvl, idx_lvl in idx_levs:
#             level_data = data[:,idx_lvl,idx_lats,idx_lons]
#             norm_lvl_data = normalize(level_data)
#             norm_data.append(VarData(variable, norm_lvl_data, target_lvl))
#     elif variable.dimensions == 2:
#         level_data = data[:,idx_lats,idx_lons]
#         norm_lvl_data = normalize(level_data)
#         norm_data.append(VarData(variable, norm_lvl_data))
#     return norm_data
def get_normalized_data(
    variable, experiment, folder, idx_lats, idx_lons, level):
    """
    Returns normalized data for one level
    """
    filename = Path(folder, FILENAME_PATTERN.format(
            var_name   = variable.name,
            level      = [1,level+1][variable.dimensions==3],
            experiment = experiment
    ))
    data = read_spcam_file(filename, variable.name)
    
    if variable.dimensions == 3:        
        level_data = data[:,0,idx_lats,idx_lons]
    elif variable.dimensions == 2:
        level_data = data[:,idx_lats,idx_lons]
    return normalize(level_data)

def load_data(var_list, experiment, folder, idx_lvls, idx_lats, idx_lons):
    data = list()
    for var in var_list:
        for target_lvl, idx_lvl in idx_lvls:
            print(f"Loading {var}, level: {idx_lvl+1}")
            norm_data = get_normalized_data(
                var, 
                experiment, 
                folder, 
                idx_lats, 
                idx_lons, 
                idx_lvl)
            if var.dimensions == 3:
                var_data = VarData(var, norm_data, target_lvl)
            elif var.dimensions == 2:
                var_data = VarData(var, norm_data)
            data.append(var_data)
            
            if var.dimensions == 2:     
                break # Stop loading data after the first level
    return data