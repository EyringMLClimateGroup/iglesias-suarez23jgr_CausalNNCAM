from enum import Enum

class SPCAM_Vars(Enum):
    def __init__(self, dimensions, var_type, label):
        self._value_ = self._name_
        self.dimensions = dimensions
        self.type = var_type
        self.label = label
    
    def __str__(self):
        return f"{self.__repr__()}, {self.label}"
        
    
    def __repr__(self):
        return f"({self._name_}, {self.dimensions}, {self.type})"
    
    tbp = (3, 'in', "tbp")            # "Surf. Temperature"
    qbp = (3, 'in', "qbp")            # "Surf. Q"
    vbp = (3, 'in', "vbp")            # "Surf. V"
    
    ps = (2, 'in', "ps")              # "Surf. Pressure"
    solin = (2, 'in', "solin")        # "Solar incoming"
    shflx = (2, 'in', "shflx")        # "Sensible heat flx"
    lhflx = (2, 'in', "lhflx")        # "Latent heat flx"
    
    tphystnd = (3, 'out', "tphystnd") # "Temperature tendency"
    phq = (3, 'out', "phq")           # "Specific humidity tendency"
    
    fsnt = (2, 'out', "fsnt")         # "Net solar flux at top of model"
    fsns = (2, 'out', "fsns")         # "Net solar flux at surface"
    flnt = (2, 'out', "flnt")         # "Net longwave flux at top of model"
    flns = (2, 'out', "flns")         # "Net longwave flux at surface"
    prect = (2, 'out', "prect")       # "Precipitation"


# TODO Move this to a config file?
experiment          = '002_train_1_year'
DATA_FOLDER         = "/work/bd0854/b309172/data/SPCAM_recons"
ANCIL_FILE          = "ancil_spcam.nc"
#FILENAME_PATTERN    = "{}_{}.nc" # variable, experiment
FILENAME_PATTERN    = "{var_name}_{level}_{experiment}.nc"
#OUTPUT_FILE_PATTERN = "{var_name}_{level}_lat-{lat}_lon-{lon}_{experiment}.obj"
#PLOT_FILE_PATTERN = "{var_name}_{level}_lat{lat}_lon{lon}_a{pc_alpha}_{experiment}.png"

tau_min             = 1
tau_max             = 1
significance        = 'analytic'
