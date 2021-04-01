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

    tbp = (3, "in", "Temperature")
    qbp = (3, "in", "Specific humidity")
    vbp = (3, "in", "Meridional wind")

    ps = (2, "in", "Surf. Pressure")
    solin = (2, "in", "Incoming solar radiation")
    shflx = (2, "in", "Sensible heat flux")
    lhflx = (2, "in", "Latent heat flux")

    tphystnd = (3, "out", "Temperature tendency")
    phq = (3, "out", "Specific humidity tendency")

    fsnt = (2, "out", "Net solar flux at top of model")
    fsns = (2, "out", "Net solar flux at surface")
    flnt = (2, "out", "Net longwave flux at top of model")
    flns = (2, "out", "Net longwave flux at surface")
    prect = (2, "out", "Precipitation")


EXPERIMENT = "002_train_1_year"
DATA_FOLDER = "/work/bd0854/b309172/data/SPCAM_recons"
ANCIL_FILE = "ancil_spcam.nc"
FILENAME_PATTERN = "{var_name}_{level}_{experiment}.nc"
TAU_MIN = 1
TAU_MAX = 1
SIGNIFICANCE = "analytic"
