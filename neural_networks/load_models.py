from pathlib                 import Path
from utils.variable          import Variable_Lev_Metadata
from tensorflow.keras.models import load_model
import                              collections


def get_path(setup, model_type, *, pc_alpha=None, threshold=None):
    """ Generate a path based on this model metadata """
    path = Path(setup.nn_output_path, model_type)
    if model_type == "CausalSingleNN":
        path = path / Path(
            "a{pc_alpha}-t{threshold}/".format(
                pc_alpha=pc_alpha, threshold=threshold
            )
        )
    str_hl = str(setup.hidden_layers).replace(", ", "_")
    str_hl = str_hl.replace("[", "").replace("]", "")
    path = path / Path(
        "hl_{hidden_layers}-act_{activation}-e_{epochs}/".format(
            hidden_layers=str_hl,
            activation=setup.activation,
            epochs=setup.epochs,
        )
    )
    return path


def get_filename(setup, output):
    """ Generate a filename to save the model """
    i_var   = setup.output_order.index(output.var)
    i_level = output.level_idx
    if i_level is None:
        i_level = 0
    return f"{i_var}_{i_level}"


def get_model(setup, output, model_type, *, pc_alpha=None, threshold=None):
    """ Get model and input list """
    folder    = get_path(setup, model_type, pc_alpha=pc_alpha, threshold=threshold)
    filename  = get_filename(setup, output)
    
    modelname = Path(folder,filename+'_model.h5')
    print(f"Load model: {modelname}")
    model     = load_model(modelname, compile=False)
    
    inputs_path = Path(folder, f"{filename}_input_list.txt")
    with open(inputs_path) as inputs_file:
        input_indices = [i for i, v in enumerate(inputs_file.readlines()) if int(v)]

    return (model, input_indices)


def get_var_list(setup, target_vars):
    output_list = list()
    for spcam_var in target_vars:
        if spcam_var.dimensions == 3:
            var_levels = [setup.children_idx_levs,setup.parents_idx_levs]\
            [spcam_var.type == 'in']
            for level, _ in var_levels:
                # There's enough info to build a Variable_Lev_Metadata list
                # However, it could be better to do a bigger reorganization
                var_name = f"{spcam_var.name}-{round(level, 2)}"
                output_list.append(var_name)
        elif spcam_var.dimensions == 2:
            var_name = spcam_var.name
            output_list.append(var_name)
    return output_list


def load_models(setup):
    """ Load all NN models specified in setup """
    models = collections.defaultdict(dict)
    
    output_list = get_var_list(setup, setup.spcam_outputs)
    if setup.do_single_nn:
        for output in output_list:
            output = Variable_Lev_Metadata.parse_var_name(output)
            models['SingleNN'][output] = get_model(
                setup, 
                output, 
                'SingleNN',
                pc_alpha=None,
                threshold=None
            )
    if setup.do_causal_single_nn:
        for pc_alpha in setup.pc_alphas:
            models['CausalSingleNN'][pc_alpha] = {}
            for threshold in setup.thresholds:
                models['CausalSingleNN'][pc_alpha][threshold] = {}
                for output in output_list:
                    output = Variable_Lev_Metadata.parse_var_name(output)
                    models['CausalSingleNN'][pc_alpha][threshold][output] = get_model(
                        setup, 
                        output, 
                        'CausalSingleNN',
                        pc_alpha=pc_alpha, 
                        threshold=threshold
                    )
                    
    return models


def get_save_plot_folder(setup, model_type, output, *, pc_alpha=None, threshold=None):
    folder = get_path(setup, model_type, pc_alpha=pc_alpha, threshold=threshold)
    path   = Path(folder, 'diagnostics')
    return path