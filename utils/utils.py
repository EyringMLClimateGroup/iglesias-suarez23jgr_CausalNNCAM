import numpy    as np
import numpy.ma as ma
from   math     import pi
from pathlib    import Path
from .constants import ANCIL_FILE, FILENAME_PATTERN, SPCAM_Vars # DATA_FOLDER
from neural_networks.cbrain.cam_constants  import *
from netCDF4    import Dataset
import pickle
from collections import deque
import getopt, yaml


#########################
#    Region utils
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
    """Converts longitudes to 0-359 and returns the index to the closest"""

    target = (target + 360) % 360
    return find_closest_value(longitudes, target)


def get_gridpoints(region):
    gridpoints = []
    levels, latitudes, longitudes = read_ancilaries(Path(ANCIL_FILE))
    idx_lats = [find_closest_value(latitudes, lat) for lat in region[0]]
    lats_lim = latitudes[idx_lats[0] : idx_lats[-1] + 1]
    for iLat in lats_lim:
        idx_lons = [find_closest_longitude(longitudes, lon) for lon in region[1]]
        lons_lim = longitudes[idx_lons[0] : idx_lons[-1] + 1]
        for iLon in lons_lim:
            gridpoints.append([iLat, iLon])
    return gridpoints


def get_levels(pres):
    levels, latitudes, longitudes = read_ancilaries(Path(ANCIL_FILE))
    idx_levs = [find_closest_value(levels, lev) for lev in pres]
    levs_lim = levels[idx_levs[-1] : idx_levs[0] + 1]
    return levs_lim


def get_weights(region, lat, norm=False):
    
    levels, latitudes, longitudes = read_ancilaries(Path(ANCIL_FILE))
    idx_lats = [find_closest_value(latitudes, ilat) for ilat in region[0]]
    idx_lat  = find_closest_value(latitudes[idx_lats[0]:idx_lats[-1]+1], lat)
    # Use cross section to get at the latitude data
    AreaWeight = np.cos(latitudes[idx_lats[0]:idx_lats[-1]+1]*pi/180.)
    # Get the Area Weighted normalised to 1
    if norm == True:
        AreaWeight = AreaWeight/np.mean(AreaWeight)
    return AreaWeight[idx_lat]


#########################
#    Load ancil utils
#########################
def read_ancilaries(path):
    with Dataset(path, "r") as nc_ancil:
        levels = nc_ancil.variables["lev"][:]
        latitudes = nc_ancil.variables["lat"][:]
        longitudes = nc_ancil.variables["lon"][:]
    return levels, latitudes, longitudes


def get_thresholds_dict(thrs_argv, key_dic=False):
    key_dic = 'thresholds_dict' if not key_dic else key_dic
    opts, args = getopt.getopt(thrs_argv, "hc:a", ["cfg_file=", "add="])
    for opt, arg in opts:
        if opt in ("-c", "--cfg_file"):
            yml_cfgFilenm = arg
        elif opt in ("-a", "--add"):
            pass
    # YAML config file
    yml_filename = yml_cfgFilenm
    yml_cfgFile = open(yml_filename)
    return yaml.load(yml_cfgFile, Loader=yaml.FullLoader)[key_dic]


#########################
#    Load data utils
#########################
class VarData:
    """
    Helper class to store the variable data and metadata
    """

    def __init__(self, variable, data, level=None):
        self.variable = variable
        self.level = level
        self.data = data
        if level is None:
            self.name = variable.name
        else:
            self.name = f"{variable.name}-{round(level, 2)}"


def read_spcam_file(path, var_name):
    with Dataset(path, "r") as file:
        data = file.variables[var_name][:]
    return data


def normalize(values):
    anom = values - np.mean(values)
    std = np.std(anom, ddof=1)
    if std != 0:
        return anom / std
    else:
        # return values
        # Adding random noise avoids causal links,
        # while allowing pcmci to run
        return np.random.normal(scale=1.0, size=values.shape)


def log_normalize(values):
    values = ma.log(values)
    values[values.mask] = ma.mean(values)
    anom = values - np.mean(values)
    std = np.std(anom, ddof=1)
    if std != 0:
        return anom / std
    else:
        # print(f"Values for prect are zero; check data. Stop processing!")
        # exit()
        raise ValueError("Values for prect are zero; check data.")


def get_normalized_data(variable, shifting, experiment, folder, idx_lats, idx_lons, level):
    """
    Returns normalized data for one level
    """
    filename = Path(
        folder,
        FILENAME_PATTERN.format(
            var_name=variable.name,
            level=[1, level + 1][variable.dimensions == 3],
            experiment=experiment,
        ),
    )
    print('Using: ', filename)
    data = read_spcam_file(filename, variable.name)

    if variable.dimensions == 3:
        level_data = data[:, 0, idx_lats, idx_lons]
    elif variable.dimensions == 2:
        level_data = data[:, idx_lats, idx_lons]

    if shifting != 0:
        level_data = deque(level_data)
        level_data.rotate(shifting)
        
    if variable == SPCAM_Vars.prect:
        print(f"Log normalization...")
        return log_normalize(level_data)
    else:
        return normalize(level_data)


def load_data(var_list, shifting, experiment, folder, idx_lvls, idx_lats, idx_lons):
    data = list()
    for var in var_list:
        for target_lvl, idx_lvl in idx_lvls:
            norm_data = get_normalized_data(
                var, shifting, experiment, folder, idx_lats, idx_lons, idx_lvl
            )
            if var.dimensions == 3:
                var_data = VarData(var, norm_data, target_lvl)
            elif var.dimensions == 2:
                var_data = VarData(var, norm_data)
            data.append(var_data)

            if var.dimensions == 2:
                break  # Stop loading data after the first level
    return data


def load_data_concat(var_list, experiment, folder, idx_lvls, idx_lats, idx_lons):
    data = list()
    for var in var_list:
        for target_lvl, idx_lvl in idx_lvls:
            norm_data = get_normalized_data(
                var, experiment, folder, idx_lats, idx_lons, idx_lvl
            )
            data.append(norm_data)
            if var.dimensions == 2:
                break  # Stop loading data after the first level
    return np.array(data)


def format_data(norm_data, var_list, idx_lvls):
    data = list()
    count = 0
    for var in var_list:
        for target_lvl, idx_lvl in idx_lvls:
            if var.dimensions == 3:
                var_data = VarData(var, norm_data[count], target_lvl)
            elif var.dimensions == 2:
                var_data = VarData(var, norm_data[count])
            data.append(var_data)

            if var.dimensions == 2:
                break  # Stop loading data after the first level
            count += 1
    return data


#########################
#    Save data utils
#########################
def generate_results_filename_single(
    var, level, lat, lon, ind_test, experiment, pattern, folder
):
    results_filename = pattern.format(
        var_name=var.name,
        level=level + 1,
        lat=int(lat),
        lon=int(lon),
        ind_test=ind_test,
        experiment=experiment,
    )
    return Path(folder, results_filename)


def generate_results_filename_concat(
    var, level, gridpoints, ind_test, experiment, pattern, folder
):
    results_filename = pattern.format(
        var_name=var.name,
        level=level + 1,
        lat1=int(gridpoints[0][0]),
        lat2=int(gridpoints[-1][0]),
        lon1=int(gridpoints[0][-1]),
        lon2=int(gridpoints[-1][-1]),
        ind_test=ind_test,
        experiment=experiment,
    )
    return Path(folder, results_filename)


def save_results(results, file):
    Path(file).parents[0].mkdir(parents=True, exist_ok=True)
    with open(file, "wb") as f:
        pickle.dump(results, f)
    print(f'Saved results into "{file}"')


def load_results(file):
    # print(f"Loading results from \"{file}\"")
    with open(file, "rb") as f:
        return pickle.load(f)