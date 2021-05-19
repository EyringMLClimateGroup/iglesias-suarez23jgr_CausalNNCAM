from pathlib import Path
from .constants import SPCAM_Vars, DATA_FOLDER, ANCIL_FILE
from . import utils

class Variable_Lev_Metadata:
    """Object that stores a SPCAM variable and one specific level.
    
    Both the level in hPa and its index in ancillaries are stored.
    
    The main way to create these objects is to use `parse_var_name()`.
    
    Attributes
    ----------
    spcam_var : SPCAM_Vars
        SPCAM_Vars object corresponding to the variable.
    level : str
        Altitude in hPa.
    level_idx : int
        Level index as found in the ancillaries file.
    """

    def __init__(self, spcam_var, level_altitude, level_idx):
        self.var = spcam_var
        self.level = level_altitude
        if level_altitude is not None:
            self.level = round(level_altitude, 2)
        self.level_idx = level_idx

    @staticmethod
    def parse_var_name(var_name):
        """Parses a string of variable and level to a
        Variable_Lev_Metadata object
        
        Parameters
        ----------
        var_name : str
            String that represents a variable and a level, with format:
            "{variable name}-{altitude in hPa}"
        
        Returns
        -------
        Variable_Lev_Metadata
            Variable_Lev_Metadata object that contains the information
            referenced in the string
        """
        if type(var_name) is Variable_Lev_Metadata:
            return var_name
        values = var_name.split("-")
        spcam_name = values[0]
        dict_spcam_vars = {v.name: v for v in SPCAM_Vars}
        spcam_var = dict_spcam_vars[spcam_name]

        if spcam_var.dimensions == 2:
            level_altitude = level_idx = None
        elif spcam_var.dimensions == 3:
            levels, _, _ = utils.read_ancilaries(Path(DATA_FOLDER, ANCIL_FILE))
            level_altitude = float(values[1])
            level_idx = utils.find_closest_value(levels, level_altitude)

        return Variable_Lev_Metadata(spcam_var, level_altitude, level_idx)

    def __str__(self):
        if self.var.dimensions == 2:
            return f"{self.var.name}"
        elif self.var.dimensions == 3:
            return f"{self.var.name}-{self.level}"

    def __repr__(self):
        return repr(str(self))
    
    def __eq__(self, other):
        if type(self) is type(other):
            return self.var == other.var and self.level_idx == other.level_idx
        else:
            return False

    def __hash__(self):
        return hash(str(self))
