from .constants import SPCAM_Vars, ANCIL_FILE # DATA_FOLDER 
from .constants import SIGNIFICANCE           # EXPERIMENT
from .variable import Variable_Lev_Metadata
from . import utils
import getopt
import yaml
from pathlib import Path
from tigramite.independence_tests import ParCorr, GPDC


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
        self.yml_cfg = yaml.load(yml_cfgFile, Loader=yaml.FullLoader)

        self._setup_common(self.yml_cfg)

    def _setup_common(self, yml_cfg):
        # Load specifications
        self.analysis = yml_cfg["analysis"]
        self.pc_alphas = yml_cfg["pc_alphas"]
        #         self.verbosity = yml_cfg["verbosity"]
        self.output_folder = yml_cfg["output_folder"]
        self.output_file_pattern = yml_cfg["output_file_pattern"][self.analysis]
        self.experiment  = yml_cfg["experiment"]
        self.data_folder = yml_cfg["data_folder"]

        region = yml_cfg["region"]
        self.gridpoints = _calculate_gridpoints(region)

        ## Model's grid
        self.levels, latitudes, longitudes = utils.read_ancilaries(
            Path(ANCIL_FILE)
        )

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
        self.list_spcam = [
            var for var in SPCAM_Vars if var.name in spcam_parents + spcam_children
        ]
        self.spcam_inputs = [var for var in self.list_spcam if var.type == "in"]
        self.spcam_outputs = [var for var in self.list_spcam if var.type == "out"]

        self.ind_test_name = yml_cfg["independence_test"]

        # # Loaded here so errors are found during setup
        # # Note the parenthesis, INDEPENDENCE_TESTS returns functions
        # self.cond_ind_test = INDEPENDENCE_TESTS[self.ind_test_name]()


class SetupPCAnalysis(Setup):

    INDEPENDENCE_TESTS = {
        "parcorr": lambda: ParCorr(significance=SIGNIFICANCE),
        "gpdc": lambda: GPDC(recycle_residuals=True),
        "gpdc_torch": lambda: _build_GPDCtorch(recycle_residuals=True)
        # "gpdc_torch" : lambda: _build_GPDCtorch(recycle_residuals=False)
    }

    def __init__(self, argv):
        super().__init__(argv)
        self._setup_pc_analysis(self.yml_cfg)

    def _setup_pc_analysis(self, yml_cfg):
        # Load specifications
        # self.analysis = yml_cfg["analysis"]
        # self.pc_alphas = yml_cfg["pc_alphas"]
        self.verbosity_pc = yml_cfg["verbosity"]
        # self.output_folder = yml_cfg["output_folder"]
        # self.output_file_pattern = yml_cfg["output_file_pattern"][self.analysis]
        # self.experiment = EXPERIMENT

        # region = yml_cfg["region"]
        # self.gridpoints = _calculate_gridpoints(region)

        ## Model's grid
        self.levels, latitudes, longitudes = utils.read_ancilaries(
            Path(ANCIL_FILE)
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

        # ## Level indexes (children & parents)
        # self.parents_idx_levs = [[lev, i] for i, lev in enumerate(self.levels)]  # All

        # lim_levels = yml_cfg["lim_levels"]
        # target_levels = yml_cfg["target_levels"]
        # target_levels = _calculate_target_levels(lim_levels, target_levels)
        # self.children_idx_levs = _calculate_children_level_indices(
        #     self.levels, target_levels, self.parents_idx_levs
        # )

        #  self.ind_test_name = yml_cfg["independence_test"]

        # Loaded here so errors are found during setup
        # Note the parenthesis, INDEPENDENCE_TESTS returns functions
        self.cond_ind_test = self.INDEPENDENCE_TESTS[self.ind_test_name]()

        self.overwrite_pc = yml_cfg.get("overwrite_pc", False)


class SetupPCMCIAggregation(Setup):
    def __init__(self, argv):
        super().__init__(argv)
        self._setup_results_aggregation(self.yml_cfg)
        self._setup_plots(self.yml_cfg)

    def _setup_results_aggregation(self, yml_cfg):
        self.thresholds = yml_cfg["thresholds"]

    def _setup_plots(self, yml_cfg):
        self.plots_folder = yml_cfg["plots_folder"]
        self.plot_file_pattern = yml_cfg["plot_file_pattern"][self.analysis]
        self.overwrite_plots = yml_cfg.get("overwrite_plot", False)


class SetupNeuralNetworks(Setup):
    def __init__(self, argv):
        super().__init__(argv)
        self._setup_results_aggregation(self.yml_cfg)
        self._setup_neural_networks(self.yml_cfg)

    def _setup_results_aggregation(self, yml_cfg):
        self.thresholds = yml_cfg["thresholds"]

    def _setup_neural_networks(self, yml_cfg):
        nn_type = yml_cfg["nn_type"]
        self.do_single_nn = self.do_causal_single_nn = False
        if nn_type == "SingleNN":
            self.do_single_nn = True
        elif nn_type == "CausalSingleNN":
            self.do_causal_single_nn = True
        elif nn_type == "all":
            self.do_single_nn = self.do_causal_single_nn = True

        self.nn_output_path = yml_cfg["nn_output_path"]

        input_order = yml_cfg["input_order"]
        self.input_order = [
            SPCAM_Vars[x] for x in input_order if SPCAM_Vars[x].type == "in"
        ]
        self.input_order_list = _make_order_list(self.input_order, self.levels)
        output_order = yml_cfg["output_order"]
        self.output_order = [
            SPCAM_Vars[x] for x in output_order if SPCAM_Vars[x].type == "out"
        ]
        self.hidden_layers = yml_cfg["hidden_layers"]
        self.activation = yml_cfg["activation"]
        self.epochs = yml_cfg["epochs"]

        # Training configuration
        self.train_verbose = yml_cfg["train_verbose"]
        self.tensorboard_folder = yml_cfg["tensorboard_folder"]

        self.train_data_folder = yml_cfg["train_data_folder"]
        self.train_data_fn = yml_cfg["train_data_fn"]
        self.valid_data_fn = yml_cfg["valid_data_fn"]

        self.normalization_folder = yml_cfg["normalization_folder"]
        self.normalization_fn = yml_cfg["normalization_fn"]

        self.input_sub = yml_cfg["input_sub"]
        self.input_div = yml_cfg["input_div"]
        self.out_scale_dict_folder = yml_cfg["out_scale_dict_folder"]
        self.out_scale_dict_fn = yml_cfg["out_scale_dict_fn"]
        self.batch_size = yml_cfg["batch_size"]

        self.init_lr = yml_cfg["init_lr"]
        self.step_lr = yml_cfg["step_lr"]
        self.divide_lr = yml_cfg["divide_lr"]

        self.train_patience = yml_cfg["train_patience"]


class SetupDiagnostics(SetupNeuralNetworks):
    def __init__(self, argv):
        super().__init__(argv)
        self._setup_diagnostics(self.yml_cfg)
    
    def _setup_diagnostics(self, yml_cfg):
        self.test_data_folder = yml_cfg["test_data_folder"]
        self.diagnostics      = yml_cfg["diagnostics"]
        self.diagnostics_time = yml_cfg["diagnostics_time"]
        self.diagnostics_time = yml_cfg["diagnostics_time"]


class SetupSherpa(SetupNeuralNetworks):
    def __init__(self, argv):
        super().__init__(argv)
        self._setup_sherpa(self.yml_cfg)
    
    def _setup_sherpa(self, yml_cfg):
        self.sherpa_hyper   = yml_cfg["sherpa_hyper"]
        self.nn_type        = yml_cfg["sherpa_nn_type"]


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


def _make_order_list(order, levels):
    # TODO? Move this out of here, to a utils module
    order_list = list()
    for i_var, spcam_var in enumerate(order):
        if spcam_var.dimensions == 3:
            n_levels = len(levels)
        elif spcam_var.dimensions == 2:
            n_levels = 1
        for i_lvl in range(n_levels):
            if spcam_var.dimensions == 3:
                level = levels[i_lvl]
                var = Variable_Lev_Metadata(spcam_var, level, i_lvl)
            elif spcam_var.dimensions == 2:
                var = Variable_Lev_Metadata(spcam_var, None, None)
            order_list.append(var)
    return order_list
