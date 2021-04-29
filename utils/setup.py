from .constants import SPCAM_Vars, DATA_FOLDER, ANCIL_FILE
from .constants import EXPERIMENT, SIGNIFICANCE
from . import utils
import getopt
import yaml
from pathlib import Path
from tigramite.independence_tests import ParCorr, GPDC

INDEPENDENCE_TESTS = {
    "parcorr": lambda: ParCorr(significance=SIGNIFICANCE),
    "gpdc": lambda: GPDC(recycle_residuals=True),
    "gpdc_torch": lambda: _build_GPDCtorch(recycle_residuals=True)
#     "gpdc_torch" : lambda: _build_GPDCtorch(recycle_residuals=False)
}


class Setup:
    def __init__(self, argv):
        try:
            opts, args = getopt.getopt(argv, "hc:a", ["cfg_file=", "add="])
        except getopt.GetoptError:
            print("pipeline.py -c [cfg_file] -a [add]")
            sys.exit(2)
        for opt, arg in opts:
            if opt == "-h":
                print("pipeline.py -c [cfg_file]")
                sys.exit()
            elif opt in ("-c", "--cfg_file"):
                yml_cfgFilenm = arg
            elif opt in ("-a", "--add"):
                pass

        # YAML config file
        self.yml_filename = yml_cfgFilenm
        yml_cfgFile = open(self.yml_filename)
        yml_cfg = yaml.load(yml_cfgFile, Loader=yaml.FullLoader)
        
        self._setup_pc_analysis(yml_cfg)
        self._setup_results_aggregation(yml_cfg)
        self._setup_neural_networks(yml_cfg)

    def _setup_pc_analysis(self, yml_cfg):
        # Load specifications
        self.analysis = yml_cfg["analysis"]
        self.pc_alphas = yml_cfg["pc_alphas"]
        self.verbosity = yml_cfg["verbosity"]
        self.output_folder = yml_cfg["output_folder"]
        self.output_file_pattern = yml_cfg["output_file_pattern"][self.analysis]
        self.experiment = EXPERIMENT

        region = yml_cfg["region"]
        self.gridpoints = _calculate_gridpoints(region)

        ## Model's grid
        self.levels, latitudes, longitudes = utils.read_ancilaries(
            Path(DATA_FOLDER, ANCIL_FILE)
        )

        ## Latitude / Longitude indexes
        self.idx_lats = [
            utils.find_closest_value(latitudes, gridpoint[0])
            for gridpoint in self.gridpoints
        ]
        self.idx_lons = [
            utils.find_closest_longitude(longitudes, gridpoint[1])
            for gridpoint in self.gridpoints
        ]

        ## Level indexes (children & parents)
        self.parents_idx_levs = [[lev, i] for i, lev in enumerate(self.levels)]  # All

        lim_levels = yml_cfg["lim_levels"]
        target_levels = yml_cfg["target_levels"]
        target_levels = _calculate_target_levels(lim_levels, target_levels)
        self.children_idx_levs = _calculate_children_level_indices(
            self.levels, target_levels, self.parents_idx_levs
        )

        ## Variables
        spcam_parents = yml_cfg["spcam_parents"]
        spcam_children = yml_cfg["spcam_children"]
        self.var_list = [
            var for var in SPCAM_Vars if var.name in spcam_parents + spcam_children
        ]
        self.var_parents = [var for var in self.var_list if var.type == "in"]
        self.var_children = [var for var in self.var_list if var.type == "out"]

        self.ind_test_name = yml_cfg["independence_test"]
        # Loaded here so errors are found during setup
        # Note the parenthesis, INDEPENDENCE_TESTS returns functions
        self.cond_ind_test = INDEPENDENCE_TESTS[self.ind_test_name]()
    
    def _setup_results_aggregation(self, yml_cfg):
        self.plots_folder = yml_cfg["plots_folder"]
        self.plot_file_pattern = yml_cfg["plot_file_pattern"][self.analysis]
        self.overwrite = False
        
    def _setup_neural_networks(self, yml_cfg):
        nn_type = yml_cfg["nn_type"]
        self.do_single_nn = self.do_causal_single_nn = False
        if nn_type == "SingleNN":
            self.do_single_nn = True
        elif nn_type == "CausalSingleNN":
            self.do_causal_single_nn = True
        elif nn_type == "all":
            self.do_single_nn = self.do_causal_single_nn = True
#         self.hidden_layers = yml_cfg["hidden_layers"]
#         self.
        # TODO
        
        input_order = yml_cfg["input_order"]
        self.input_order = [SPCAM_Vars[x] for x in input_order if SPCAM_Vars[x].type == "in"]
        output_order = yml_cfg["output_order"]
        self.output_order = [SPCAM_Vars[x] for x in output_order if SPCAM_Vars[x].type == "out"]
        


def _calculate_gridpoints(region):
    ## Region / Gridpoints
    if region is False:
        region = [[-90, 90], [0, -0.5]]  # All
    return utils.get_gridpoints(region)


def _calculate_target_levels(lim_levels, target_levels):
    ## Children levels (parents includes all)
    if lim_levels is not False and target_levels is False:
        target_levels = utils.get_levels(lim_levels)
    return target_levels


def _calculate_children_level_indices(levels, target_levels, parents_idx_levs):
    if target_levels is not False:
        children_idx_levs = [
            [lev, utils.find_closest_value(levels, lev)] for lev in target_levels
        ]
    else:
        children_idx_levs = parents_idx_levs
    return children_idx_levs


def _build_GPDCtorch(**kwargs):
    """
    Helper function to isolate the GPDCtorch import, as it's not
    present in the master version of Tigramite
    """
    from tigramite.independence_tests import GPDCtorch

    return GPDCtorch(**kwargs)
