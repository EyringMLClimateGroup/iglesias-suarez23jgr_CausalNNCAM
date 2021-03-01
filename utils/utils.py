import numpy    as np
from pathlib    import Path
from .constants import DATA_FOLDER, ANCIL_FILE, FILENAME_PATTERN #, OUTPUT_FILE_PATTERN
from netCDF4    import Dataset
import pickle

#########################
#    Find region utils
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


def get_gridpoints(region):
    gridpoints = []
    levels, latitudes, longitudes = read_ancilaries(Path(DATA_FOLDER, ANCIL_FILE))
    idx_lats = [find_closest_value(latitudes, lat) for lat in region[0]]
    lats_lim = latitudes[idx_lats[0]:idx_lats[-1]+1]
    for iLat in lats_lim:
        idx_lons = [find_closest_longitude(longitudes, lon) for lon in region[1]]
        lons_lim = longitudes[idx_lons[0]:idx_lons[-1]+1]
        for iLon in lons_lim:
            gridpoints.append([iLat,iLon]) 
    return gridpoints


def get_levels(pres):
    levels, latitudes, longitudes = read_ancilaries(Path(DATA_FOLDER, ANCIL_FILE))
    idx_levs = [find_closest_value(levels, lev) for lev in pres]
    levs_lim = levels[idx_levs[-1]:idx_levs[0]+1]
    return levs_lim


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
            self.name = f"{variable.name}-{round(level, 2)}"

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

def load_data_concat(var_list, experiment, folder, idx_lvls, idx_lats, idx_lons):
    data = list()
    for var in var_list:
        for target_lvl, idx_lvl in idx_lvls:
            norm_data = get_normalized_data(
                var, 
                experiment, 
                folder, 
                idx_lats, 
                idx_lons, 
                idx_lvl)
            data.append(norm_data)
            if var.dimensions == 2:     
                break # Stop loading data after the first level
    return np.array(data)


def format_data(norm_data, var_list, idx_lvls):
    data  = list()
    count = 0
    for var in var_list:
        for target_lvl, idx_lvl in idx_lvls:
            if var.dimensions == 3:
                var_data = VarData(var, norm_data[count], target_lvl)
            elif var.dimensions == 2:
                var_data = VarData(var, norm_data[count])
            data.append(var_data)
            
            if var.dimensions == 2:     
                break # Stop loading data after the first level
            count+=1
    return data


#########################
#    Save data utils
#########################
def generate_results_filename_single(
        var, level, lat, lon, experiment, pattern, folder):
    results_filename = pattern.format(
            var_name = var.name,
            level = level+1,
            lat = int(lat),
            lon = int(lon),
            experiment = experiment
    )
    return Path(folder, results_filename)


def generate_results_filename_concat(
        var, level, gridpoints, experiment, pattern, folder):
    results_filename = pattern.format(
            var_name   = var.name,
            level      = level+1,
            lat1       = int(gridpoints[0][0]),
            lat2       = int(gridpoints[-1][0]),
            lon1       = int(gridpoints[0][-1]),
            lon2       = int(gridpoints[-1][-1]),
            experiment = experiment
    )
    return Path(folder, results_filename)


def save_results(results, file):
    Path(file).parents[0].mkdir(parents=True, exist_ok=True)
    with open(file, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved results into \"{file}\"")


def load_results(file):
#     print(f"Loading results from \"{file}\"")
    with open(file, "rb") as f:
        return pickle.load(f)
