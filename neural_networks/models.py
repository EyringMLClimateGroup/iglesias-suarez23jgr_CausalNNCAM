import numpy as np
import tensorflow as tf
from pathlib import Path

from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense

from utils.constants import SPCAM_Vars
from utils.variable import Variable_Lev_Metadata
import utils.pcmci_aggregation as aggregation


class ModelDescription:
    """ Object that stores a Keras model and metainformation about it.
    
    Attributes
    ----------
    output : Variable_Lev_Metadata
        Output variable of the model.
    pc_alpha : str
        Meta information. PC alpha used to find the inputs.
    threshold : str
        Meta information. Gridpoint threshold used to select the inputs.
    inputs : list(Variable)
        List of the variables (and variable level) that cause the output
        variable.
    hidden_layers : list(int)
        Description of the hidden dense layers of the model
        (default [32, 32, 32]).
    activation : Keras-compatible activation function
        Activation function used for the hidden dense layers
        (default "relu").
    model : Keras model
        Model created using the given information.
        See `_build_model()`.
    input_vars_dict:
    output_vars_dict:
    
    #TODO
    
    """

    def __init__(self, output, inputs, model_type, pc_alpha, threshold, setup):
        """
        Parameters
        ----------
        output : str
            Output variable of the model in string format. See Variable_Lev_Metadata.
        inputs : list(str)
            List of strings for the variables that cause the output variable.
            See Variable_Lev_Metadata.
        model_type : str
            # TODO
        pc_alpha : str
            Meta information. PC alpha used to find the inputs.
        threshold : str
            Meta information. Gridpoint threshold used to select the inputs.
        hidden_layers : list(int)
            Description of the hidden dense layers of the model.
        activation : Keras-compatible activation function
            Activation function used for the hidden dense layers.
        """
        self.setup = setup
        self.output = Variable_Lev_Metadata.parse_var_name(output)
        self.inputs = sorted(
            [Variable_Lev_Metadata.parse_var_name(p) for p in inputs],
            key=lambda x: self.setup.input_order_list.index(x)
        )
        self.model_type = model_type
        self.pc_alpha = pc_alpha
        self.threshold = threshold
        self.model = self._build_model()
        self.input_vars_dict = ModelDescription._build_vars_dict(self.inputs)
        self.output_vars_dict = ModelDescription._build_vars_dict([self.output])

    def _build_model(self):
        """ Build a Keras model with the given information.
        
        Some parameters are not configurable, taken from Rasp et al.
        """
        input_shape = len(self.inputs)
        input_shape = (input_shape,)
        model = dense_nn(
            input_shape=input_shape,
            output_shape=1,  # Only one output per model
            hidden_layers=self.setup.hidden_layers,
            activation=self.setup.activation,
        )
        model.compile(
            # TODO? Move to configuration
            optimizer="adam",  # From train.py (default)
            loss="mse",  # From 006_8col_pnas_exact.yml
            metrics=[tf.keras.losses.mse],  # From train.py (default)
        )
        return model

    @staticmethod
    def _build_vars_dict(list_variables):
        """ Convert the given list of Variable_Lev_Metadata into a
        dictionary to be used on the data generator.
        
        Parameters
        ----------
        list_variables : list(Variable_Lev_Metadata)
            List of variables to be converted to the dictionary format
            used by the data generator
        
        Returns
        -------
        vars_dict : dict{str : list(int)}
            Dictionary of the form {ds_name : list of levels}, where
            "ds_name" is the name of the variable as stored in the
            dataset, and "list of levels" a list containing the indices
            of the levels of that variable to use, or None for 2D
            variables.
        """
        vars_dict = dict()
        for variable in list_variables:
            ds_name = variable.var.ds_name  # Name used in the dataset
            if variable.var.dimensions == 2:
                vars_dict[ds_name] = None
            elif variable.var.dimensions == 3:
                levels = vars_dict.get(ds_name, list())
                levels.append(variable.level_idx)
                vars_dict[ds_name] = levels
        return vars_dict

    def fit_model(self, x, validation_data, epochs, callbacks, verbose=1):
        """ Train the model """
        self.model.fit(
            x=x,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose,
        )

    def get_path(self, base_path):
        """ Generate a path based on this model metadata """
        path = Path(base_path, self.model_type)
        if self.model_type == "CausalSingleNN":
            path = path / Path(
                "a{pc_alpha}-t{threshold}/".format(
                    pc_alpha=self.pc_alpha, threshold=self.threshold
                )
            )
        str_hl = str(self.setup.hidden_layers).replace(", ", "_")
        str_hl = str_hl.replace("[", "").replace("]", "")
        path = path / Path(
            "hl_{hidden_layers}-act_{activation}-e_{epochs}/".format(
                hidden_layers=str_hl,
                activation=self.setup.activation,
                epochs=self.setup.epochs,
            )
        )
        return path

    def get_filename(self):
        """ Generate a filename to save the model """
        i_var = self.setup.output_order.index(self.output.var)
        i_level = self.output.level_idx
        if i_level is None:
            i_level = 0
        return f"{i_var}_{i_level}"

    def save_model(self, base_path):
        """ Save model, weights and input list """
        folder = self.get_path(base_path)
        filename = self.get_filename()
        print(f"Using filename {filename}.")
        # Save model
        self.model.save(Path(folder, f"{filename}_model.h5"))
        # Save weights
        self.model.save_weights(Path(folder, f"{filename}_weights.h5"))
        # Save input list
        self.save_input_list(folder, filename)

    def save_input_list(self, folder, filename):
        """ Save input list """
        input_list = self.get_input_list()
        with open(Path(folder, f"{filename}_input_list.txt"), "w") as f:
            for line in input_list:
                print(str(line), file=f)

    def get_input_list(self):
        """ Generate input list """
        return [int(var in self.inputs) for var in self.setup.input_order_list]

    def __str__(self):
        name = f"{self.model_type}: {self.output}"
        if self.pc_alpha != None:
            # pc_alpha and threshold should be either both None or both not None
            name += f", a{self.pc_alpha}-t{self.threshold}"
        return name

    def __repr__(self):
        return repr(str(self))

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)


def dense_nn(input_shape, output_shape, hidden_layers, activation):
    """ Create a dense NN in base of the parameters received """
    model = Sequential()
    model.add(Input(shape=input_shape))

    for n_layer_nodes in hidden_layers:
        model.add(Dense(n_layer_nodes, activation=activation))

    model.add(Dense(output_shape))
    return model


def generate_all_single_nn(setup):
    """ Generate all NN with one output and all inputs specified in the setup"""
    model_descriptions = list()

    inputs = list()  # TODO Parents and levels
    for spcam_var in setup.spcam_inputs:
        if spcam_var.dimensions == 3:
            for level, _ in setup.parents_idx_levs:
                # There's enough info to build a Variable_Lev_Metadata list
                # However, it could be better to do a bigger reorganization
                var_name = f"{spcam_var.name}-{round(level, 2)}"
                inputs.append(var_name)
        elif spcam_var.dimensions == 2:
            var_name = spcam_var.name
            inputs.append(var_name)

    output_list = list()
    for spcam_var in setup.spcam_outputs:
        if spcam_var.dimensions == 3:
            for level, _ in setup.children_idx_levs:
                # There's enough info to build a Variable_Lev_Metadata list
                # However, it could be better to do a bigger reorganization
                var_name = f"{spcam_var.name}-{round(level, 2)}"
        elif spcam_var.dimensions == 2:
            var_name = spcam_var.name
        output_list.append(var_name)

    for output in output_list:
        model_description = ModelDescription(
            output, inputs, "SingleNN", pc_alpha=None, threshold=None, setup=setup,
        )
        model_descriptions.append(model_description)
    return model_descriptions


def generate_all_causal_single_nn(setup, aggregated_results):
    """ Generate all NN with one output and selected inputs from a pc analysis """

    model_descriptions = list()

    for output, pc_alpha_dict in aggregated_results.items():
        print(output)
        if len(pc_alpha_dict) == 0:  # May be empty
            # TODO How to approach this?
            print("Empty results")
            pass
        for pc_alpha, pc_alpha_results in pc_alpha_dict.items():
            var_names = np.array(pc_alpha_results["var_names"])
            for threshold, parent_idxs in pc_alpha_results["parents"].items():
                parents = var_names[parent_idxs]
                model_description = ModelDescription(
                    output, parents, "CausalSingleNN", pc_alpha, threshold, setup=setup,
                )
                model_descriptions.append(model_description)
    return model_descriptions


def generate_models(setup):
    """ Generate all NN models specified in setup """
    model_descriptions = list()

    if setup.do_single_nn:
        model_descriptions.extend(generate_single_nn(setup))

    if setup.do_causal_single_nn:
        collected_results, errors = aggregation.collect_results(setup)
        aggregation.print_errors(errors)
        aggregated_results, var_names_parents = aggregation.aggregate_results(
            collected_results, setup
        )
        model_descriptions.extend(generate_causal_single_nn(setup, aggregated_results))

    return model_descriptions
