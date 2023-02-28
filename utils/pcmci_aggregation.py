import os, pickle
import numpy as np
from pathlib import Path

from . import utils
from .plotting import plot_links, find_linked_variables, plot_links_metrics, plot_matrix, plot_matrix_insets
from .variable import Variable_Lev_Metadata
from .constants import ANCIL_FILE, AGGREGATE_PATTERN
from scipy      import stats
from sklearn    import preprocessing

from collections import defaultdict


# Collect results

levels, latitudes, longitudes = utils.read_ancilaries(Path(ANCIL_FILE))
# lat_wts = utils.get_weights(latitudes,norm=True)

def get_parents_from_links(links):
    """ Return a list of variables that are parents, given a
    dictionary of links of the form {variable : [(parent, lag)]}
    """

    linked_variables = set()  # Avoids duplicates and sorts when iterated
    for parents_list in links.values():
        if len(parents_list) > 0:
            for parent in parents_list:
                linked_variables.add(parent[0])  # Remove the lag
    return [(i in linked_variables) for i in range(len(links))]


def record_error(file, error_type, errors):
    """Record if a file has failed with a given error type"""

    error_list = errors.get(error_type, list())
    if file not in error_list:
        error_list.append(file)
        errors[error_type] = error_list


def collect_in_dict(_dict, key, value):
    """Add a value to a list in a dictionary"""

    collected_values = _dict.get(key, list())
    collected_values.append(value)
    _dict[key] = collected_values


def collect_results_file(key, results_file, collected_results, errors, lat_wtg):
    """Collect pcmci results from a file in a dictionary with other
    results.
    
    Parameters:
    -----------
    
    key
        Identifier of the variable and level to which the results will
        be collected.
    results_file
        Path to a file storing results from a pcmci analysis.
    collected_results
        Dictionary. It is updated with the results read when this
        function is called.
    errors
        Auxiliary dictionary to keep track of errors found during the
        execution. It is updated whenever an error is found in a call.
    """

    collected_pc_alpha = collected_results.get(key, dict())
    if not results_file.is_file():
        record_error(results_file, "not_found", errors)
        return
    results = utils.load_results(results_file)
    for pc_alpha, alpha_result in results.items():
        if len(alpha_result) > 0:
            collected = collected_pc_alpha.get(pc_alpha, dict())
            collected_pc_alpha[pc_alpha] = collected
            # Collect parents
            links = alpha_result["links"]
            parents = get_parents_from_links(links)
            parents = [[lat_wtg,0.][parent == False] for parent in parents]
            collect_in_dict(collected, "parents", parents)
            # Collect val_matrix
            val_matrix = alpha_result["val_matrix"]
            collect_in_dict(collected, "val_matrix", val_matrix)
            # Check var_names
            var_names = alpha_result["var_names"]
            if "var_names" not in collected:
                collected["var_names"] = var_names
            else:
                stored_var_names = collected["var_names"]
                assert stored_var_names == var_names, (
                    "Found different variable names.\n"
                    f"Expected {stored_var_names}\nFound {var_names}"
                )
        else:
            record_error(results_file, "is_empty", errors)
#     collected_results[str(key)] = collected_pc_alpha
    collected_results[key] = collected_pc_alpha
    # Doesn't return, instead modifies the received `collected results` object


def count_total_variables(var_children, levels):
    """ Return the total number of variables given a list of spcam
    variables and a list of levels of for 3D variables
    """
    total = 0
    for child in var_children:

        if child.dimensions == 2:
            total += 1
        elif child.dimensions == 3:
            total += len(levels)
    return total


def collect_results(setup, reuse=False):
    """Collect the pcmci results defined in the setup in a file"""
    collected_results = dict()
    errors            = dict()

    total_vars = count_total_variables(setup.spcam_outputs, setup.children_idx_levs)
    total_files = total_vars * len(setup.gridpoints)
    file_progress = 0
    step_progress = 0
    step = 5

    folder = Path(setup.output_folder+'/'+setup.aggregate_folder)
    Path(folder).mkdir(parents=True, exist_ok=True)
    
    for child in setup.spcam_outputs:
        print(f"Variable: {child.name}")        
        if child.dimensions == 2:
            child_levels = [[setup.levels[-1], 0]]
            child_var = Variable_Lev_Metadata(child, None, None)
        elif child.dimensions == 3:
            child_levels = setup.children_idx_levs
        for level in child_levels:
            if child.dimensions == 3:
                child_var = Variable_Lev_Metadata(child, level[0], level[1])

            collected_results_tmp       = dict()
            errors_tmp                  = dict()
            collected_results_fixed     = dict()
            errors_fixed                = dict()
                
            if setup.analysis == "single":
                
                filename = Path(
                    folder,
                    AGGREGATE_PATTERN.format(
                    var_name=child_var,
                    ind_test_name=setup.ind_test_name,
                    experiment=setup.experiment,
                    ),
                )
                errorname = Path(
                    folder,
                    AGGREGATE_PATTERN.format(
                    var_name=child_var,
                    ind_test_name=setup.ind_test_name,
                    experiment=setup.experiment,
                    )+"_errors",
                )
                                
                if not os.path.isfile(filename) or reuse == False:
                
                    for i_grid, (lat, lon) in enumerate(setup.gridpoints):
                        results_file = utils.generate_results_filename_single(
                            child,
                            level[1],
                            lat,
                            lon,
                            setup.ind_test_name,
                            setup.experiment,
                            setup.output_file_pattern,
                            setup.output_folder,
                        )

                        # Area-weights?
                        if setup.area_weighted:
                            lat_wtg = utils.get_weights(setup.region, lat, norm=True)
                        else:
                            lat_wtg = 1.
#                         print(f"Collecting: {results_file}")
                        collect_results_file(
                            child_var, results_file, collected_results_tmp, errors_tmp, lat_wtg
                        )

                        file_progress += 1
                        if file_progress == total_files or (
                            (file_progress / total_files * 100) >= step * step_progress
                        ):
                            step_progress = 1 + np.floor(
                                (file_progress / total_files * 100) / step
                            )
                            print(
                                "Progress: {:.2f}% - {} of {} files".format(
                                    file_progress / total_files * 100,
                                    file_progress,
                                    total_files,
                                )
                            )
                            print(f"Collecting: {results_file}")

                    # Fix keys: from class type to str (for saving results in dicts)
                    for iK in collected_results_tmp.keys():
                        collected_results[str(iK)]       = collected_results_tmp[iK]
                        collected_results_fixed[str(iK)] = collected_results_tmp[iK]
                    for iK in errors_tmp.keys():
                        errors[str(iK)]       = errors_tmp[iK]
                        errors_fixed[str(iK)] = errors_tmp[iK]

                    # Save to pickle object
                    if not os.path.isfile(filename) and reuse == True:
                        print(f'Saving {filename}...\n')
                        collected_results_outfile = open(filename,'wb')
                        errors_outfile            = open(errorname,'wb')
                        pickle.dump(collected_results_fixed,collected_results_outfile)
                        pickle.dump(errors_fixed,errors_outfile)
                        collected_results_outfile.close()
                        errors_outfile.close()

                else:
                    print(f'{filename} exists; loading...\n')
                    collected_results_outfile = open(filename,'rb')
                    errors_outfile            = open(errorname,'rb')
                    collected_results_tmp     = pickle.load(collected_results_outfile)
                    errors_tmp                = pickle.load(errors_outfile)
                    collected_results_outfile.close()
                    errors_outfile.close()

                    collected_results.update(collected_results_tmp)
                    errors.update(errors_tmp)
                            
            elif setup.analysis == "concat":
                results_file = utils.generate_results_filename_concat(
                    child,
                    level[-1],
                    setup.gridpoints,
                    setup.ind_test_name,
                    setup.experiment,
                    setup.output_file_pattern,
                    setup.output_folder,
                )
                collect_results_file(key, results_file, collected_results, errors)
    
    return collected_results, errors


def print_errors(errors):
    """Prints the error dictionary according to type"""

    if len(errors) == 0:
        print("No errors were found")
        return

    print("ERRORS\n======")
    for error_type, error_list in errors.items():
        # msg = "{}: {} of {} files ({:.2f}%)".format(
        msg = "{}: {} files".format(
            error_type,
            len(error_list),
            # total_files,
            # len(error_list) / total_files * 100,
        )
        print(msg)
        print("-" * len(msg))
        for file in error_list:
            print(file)


def aggregate_results(collected_results, setup, threshold_dict=False):
    # Configuration
    if threshold_dict:
        thresholds = utils.get_thresholds_dict(setup.thrs_argv) # dict
    else:
        thresholds = setup.thresholds # list
    pc_alphas_filter = [str(a) for a in setup.pc_alphas]

    # Combine results
    aggregated_results = dict()
    var_names_parents = None
    for child, collected_pc_alpha in collected_results.items():
        dict_pc_alpha_parents = dict()
        for pc_alpha, collected in collected_pc_alpha.items():
            dict_threshold_parents = dict()
            dict_num_parents = dict()
            dict_per_parents = dict()
            if pc_alpha not in pc_alphas_filter:
                continue  # Skip this pc alpha
            val_matrix_list = np.array(collected["val_matrix"])
            val_matrix_mean = val_matrix_list.mean(axis=0)

            var_names = collected["var_names"]
            if var_names_parents is None:
                # NOTE: This assumes that the list has only one child, and
                # that the child is last
                var_names_parents = var_names[:-1]
            else:
                assert var_names_parents == var_names[:-1], (
                    "Found different variable names. "
                    f"Expected {var_names_parents}\nFound {var_names[:-1]}"
                )

            parents_matrix = collected["parents"]
            parents_matrix = np.array(parents_matrix)
            parents_percent = parents_matrix.mean(axis=0)[:-1] # All but itself [which is zero]
            thresholds_list = [thresholds[child]] if threshold_dict else thresholds
            for threshold in thresholds_list:
                # Threshold based on the own output's inputs-distrubution
                if setup.pdf:
                    parents_percent = [[0.00001,i][i>0.00001] for i in parents_percent]
                    parents_percent_boxcox, _ = stats.boxcox(parents_percent) # to normal distribution
                    ks = stats.kstest(
                        parents_percent_boxcox, 
                        stats.norm.name, 
                        stats.norm.fit(parents_percent_boxcox), 
                        len(parents_percent_boxcox))[1]   # return p-value
                    if 'phq-3.' in child or 'phq-7.' in child:
                        parents = []
                    else:
                        zscores = stats.zscore(parents_percent_boxcox)
                        n_sided = 1 # 1: one-tail; two-tail
                        z_crit  = stats.norm.ppf(1-threshold/n_sided)
                        parents = [i for i in range(len(parents_percent_boxcox)) if zscores[i] > -z_crit]
                    if ks < 0.05: # Normal distribution, KS's p-val should be > 0.05
                        print(f'Caution! For threshold ({threshold}): {child} not normaly distributed')
                else:
                    parents_filtered = parents_percent >= threshold
                    parents = [
                        i for i in range(len(parents_filtered)) if parents_filtered[i]
                    ]
                threshold_nm = 'optimized' if threshold_dict else str(threshold)
                dict_threshold_parents[threshold_nm] = parents
                dict_num_parents[threshold_nm]  = len(parents)
                dict_per_parents[threshold_nm]  = len(parents) * 100. / len(var_names_parents)
            
            dict_pc_alpha_parents[pc_alpha] = {
                "parents": dict_threshold_parents,
                "num_parents": dict_num_parents,
                "per_parents": dict_per_parents,
                "val_matrix": val_matrix_mean,
                "parents_percent": parents_percent,
                "var_names": var_names,
            }

        aggregated_results[child] = dict_pc_alpha_parents
    return aggregated_results, var_names_parents


def aggregate_results_for_numparents(collected_results, setup, thresholds_dict=False, random=False):
    # Configuration
    if thresholds_dict: thresholds_dict = utils.get_thresholds_dict(setup.thrs_argv)
    
    pc_alphas_filter = [str(a) for a in setup.pc_alphas]

    # Combine results
    aggregated_results = dict()
    var_names_parents = None
    for child, collected_pc_alpha in collected_results.items():
        dict_pc_alpha_parents = dict()
        for pc_alpha, collected in collected_pc_alpha.items():
            dict_threshold_parents = dict()
            dict_num_parents = dict()
            dict_per_parents = dict()
            if pc_alpha not in pc_alphas_filter:
                continue  # Skip this pc alpha
            val_matrix_list = np.array(collected["val_matrix"])
            val_matrix_mean = val_matrix_list.mean(axis=0)

            var_names = collected["var_names"]
            if var_names_parents is None:
                # NOTE: This assumes that the list has only one child, and
                # that the child is last
                var_names_parents = var_names[:-1]
            else:
                assert var_names_parents == var_names[:-1], (
                    "Found different variable names. "
                    f"Expected {var_names_parents}\nFound {var_names[:-1]}"
                )

            parents_matrix = collected["parents"]
            parents_matrix = np.array(parents_matrix)
            parents_percent = parents_matrix.mean(axis=0)[:-1] # All but itself [which is zero]
            
            if isinstance(parents_percent,np.ndarray):
                if random:
                    proba_0 = float(thresholds_dict[child]) if thresholds_dict else float(setup.thresholds[0])
                    parents_percent = np.random.choice([0, 1], size=len(parents_percent), p=[proba_0, 1-proba_0])
                    parents = [i for i in range(len(parents_percent)) if parents_percent[i] > 0]
                else:
                    numparents = utils.get_thresholds_dict(setup.numparents_argv, key_dic='numparents_dict') # dict
                    if int(numparents[child]) > 0:
                        parents_percent_ordered = np.sort(parents_percent)[::-1]
                        parents_threshold = parents_percent_ordered[:int(numparents[child])][-1]
                        parents = [i for i in range(len(parents_percent)) if parents_percent[i] >= parents_threshold]
                    else:
                        parents = []
            else:
                parents = []
            
            threshold_nm = 'optimized' if thresholds_dict else setup.thresholds[0]
            dict_threshold_parents[threshold_nm] = parents
            dict_num_parents[threshold_nm]  = len(parents)
            dict_per_parents[threshold_nm]  = len(parents) * 100. / len(var_names_parents)
            
            dict_pc_alpha_parents[pc_alpha] = {
                "parents": dict_threshold_parents,
                "num_parents": dict_num_parents,
                "per_parents": dict_per_parents,
                "val_matrix": val_matrix_mean,
                "parents_percent": parents_percent,
                "var_names": var_names,
            }

        aggregated_results[child] = dict_pc_alpha_parents
    return aggregated_results, var_names_parents


def print_aggregated_results(var_names_parents, aggregated_results):
    """Print combined results"""

    var_names_np = np.array(var_names_parents)
    for child, dict_pc_alpha_parents in aggregated_results.items():
        print(f"\n{child}")
        for pc_alpha, pc_alpha_results in dict_pc_alpha_parents.items():
            print(f"pc_alpha = {pc_alpha}")
            dict_threshold_parents = pc_alpha_results["parents"]
            dict_num_parents = pc_alpha_results["num_parents"]
            dict_per_parents = pc_alpha_results["per_parents"]
            for threshold, parents in dict_threshold_parents.items():
                print(f"* Threshold {threshold}:\n \
                Total number of parents: {dict_num_parents[threshold]} ({dict_per_parents[threshold]:1.1f} %)\n \
                Parents: {var_names_np[parents]}\n\n")


def build_pc_alpha_plot_matrices(var_names_parents, aggregated_results):
    len_parents = len(var_names_parents)
    len_total = len_parents + len(aggregated_results)

    dict_full = dict()
    for i_child, (child, dict_pc_alpha_parents) in enumerate(
        aggregated_results.items()
    ):
        i_child = i_child + len_parents
        for pc_alpha, pc_alpha_results in dict_pc_alpha_parents.items():
            dict_pc_alpha_full = dict_full.get(pc_alpha, dict())
            dict_full[pc_alpha] = dict_pc_alpha_full

            # Build val_matrix (only from inputs to outputs)
            pc_alpha_val_matrix = pc_alpha_results["val_matrix"]
            full_val_matrix = dict_pc_alpha_full.get(
                "val_matrix", np.zeros((len_total, len_total, 1))
            )
            full_val_matrix[:len_parents, i_child, 0] = pc_alpha_val_matrix[
                :len_parents, -1, 1
            ]
            dict_pc_alpha_full["val_matrix"] = full_val_matrix

            # Build link_width
            parents_percent = pc_alpha_results["parents_percent"]
            link_width = dict_pc_alpha_full.get(
                "link_width", np.zeros((len_total, len_total, 1))
            )
            link_width[:len_parents, i_child, 0] = parents_percent[:len_parents]
            dict_pc_alpha_full["link_width"] = link_width

    return dict_full


class Combination:
    """ """

    def __init__(self, pc_alpha, threshold):
        self.pc_alpha = pc_alpha
        self.threshold = threshold

    def __str__(self):
        return f"a{self.pc_alpha}-t{self.threshold}"

    def __repr__(self):
        return repr(str(self))

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)


def build_plot_matrices(var_names_parents, aggregated_results):

    var_names_children = list()

    dict_combinations = dict()
    len_parents = len(var_names_parents)
    len_total = len_parents + len(aggregated_results)
    for i_child, (child, dict_pc_alpha_parents) in enumerate(
        aggregated_results.items()
    ):
        i_child = i_child + len_parents
        var_names_children.append(child)
        for pc_alpha, pc_alpha_results in dict_pc_alpha_parents.items():
            dict_threshold_parents = pc_alpha_results["parents"]
            for threshold, parents in dict_threshold_parents.items():
                key = Combination(pc_alpha, threshold)
                combination_results = dict_combinations.get(key, dict())

                # Build links
                links = combination_results.get(
                    "links", {i: [] for i in range(len_total)}
                )
                links[i_child] = [(parent, 0) for parent in parents]
                combination_results["links"] = links

                # Store results
                dict_combinations[key] = combination_results

    all_var_names = var_names_parents + var_names_children  # Concatenation
    pc_alpha_plot_matrices = build_pc_alpha_plot_matrices(
        var_names_parents, aggregated_results
    )
    for combination, combination_results in dict_combinations.items():
        combination_results["var_names"] = all_var_names
        val_matrix = pc_alpha_plot_matrices[combination.pc_alpha]["val_matrix"]
        combination_results["val_matrix"] = val_matrix
        link_width = pc_alpha_plot_matrices[combination.pc_alpha]["link_width"]
        combination_results["link_width"] = link_width

    return dict_combinations



def build_links_metrics(setup, aggregated_results):

    pc_alphas  = setup.pc_alphas
    outputs_nm = [var.name for var in setup.list_spcam if var.type == "out"]
    
#     Model_levels = [ 
#         3.64, 7.59, 14.36, 24.61, 38.27, 54.6, 72.01, 87.82, 103.32, 121.55, 142.99, 
#         168.23, 197.91, 232.83, 273.91, 322.24, 379.1, 445.99, 524.69, 609.78, 
#         691.39, 763.4, 820.86, 859.53, 887.02, 912.64, 936.2, 957.49, 976.33, 992.56
#     ]
#     target_levels = Model_levels if not setup.target_levels else setup.target_levels
#     target_levels = [str(round(i,2)) for i in target_levels]
    
    dict_combinations = {}
    
    # Num. of parents (by level in 3D)
    for i_child, (child, dict_pc_alpha_parents) in enumerate(
            aggregated_results.items()
        ):  
        
        for pc_alpha, pc_alpha_results in dict_pc_alpha_parents.items():
            
            Model_levels = [
                float(i.split('-')[-1]) \
                for i in aggregated_results[child][pc_alpha]['var_names'] \
                if 'tbp' in i
            ]
            target_levels = Model_levels if not setup.target_levels else setup.target_levels
            target_levels = [str(i) for i in target_levels]
            
            child_main = child.split('-')[0]
            child_lev = child.split('-')[-1] if child_main in ['phq', 'tphystnd'] else 'None'
            
            if child_main not in dict_combinations: dict_combinations[child_main] = {}
            if pc_alpha not in dict_combinations[child_main]: dict_combinations[child_main][pc_alpha] = {}

            dict_threshold_parents = pc_alpha_results["parents"]
            dict_num_parents       = pc_alpha_results["num_parents"]
            dict_per_parents       = pc_alpha_results["per_parents"]

            thrs = [float(thr) for thr, nPar in dict_num_parents.items()]
            nPar = [nPar       for thr, nPar in dict_num_parents.items()]
            pPar = [pPar       for thr, pPar in dict_per_parents.items()]

            if child_main in ['phq', 'tphystnd'] and child_lev not in target_levels:
                pass
            else:
                to_be_updated = {child_lev:{'thresholds':thrs,'num_parents':nPar,'per_parents':pPar,}}
                dict_combinations[child_main][pc_alpha].update(to_be_updated)
    
    for child_main in dict_combinations.keys():
        for pc_alpha in dict_combinations[child_main].keys():
            
#             if len(dict_combinations[child_main][pc_alpha]) > 1:
            if child_main in ['phq', 'tphystnd']:
                
                nLevs = len(dict_combinations[child_main][pc_alpha].keys())
                nPar_mean = np.zeros([nLevs,len(thrs)])
                pPar_mean = np.zeros([nLevs,len(thrs)])
                count = 0
                for i_child in dict_combinations[child_main][pc_alpha].keys():
                    nPar_mean[count,:] = dict_combinations[child_main][pc_alpha][i_child]['num_parents']
                    pPar_mean[count,:] = dict_combinations[child_main][pc_alpha][i_child]['per_parents'] 
                    count += 1
                nPar_mean, pPar_mean = np.mean(nPar_mean,axis=0), np.mean(pPar_mean,axis=0)
                to_be_updated = {'mean':{'thresholds':thrs,'num_parents':nPar_mean,'per_parents':pPar_mean,}}
                dict_combinations[child_main][pc_alpha].update(to_be_updated)
            
            else:
                dict_combinations[child_main][str(pc_alpha)]['mean'] = \
                dict_combinations[child_main][str(pc_alpha)].pop('None')
    
    dict_combinations['mean'] = {}
    for j, iPC in enumerate(pc_alphas):
        nPar_mean = np.zeros([len(outputs_nm),len(thrs)])
        pPar_mean = np.zeros([len(outputs_nm),len(thrs)])
        for j, iVar in enumerate(outputs_nm):
            nPar_mean[j,:] = dict_combinations[iVar][str(iPC)]['mean']['num_parents']
            pPar_mean[j,:] = dict_combinations[iVar][str(iPC)]['mean']['per_parents']
        nPar_mean, pPar_mean = np.mean(nPar_mean,axis=0), np.mean(pPar_mean,axis=0)
        to_be_updated = {str(iPC):{'thresholds':thrs,'num_parents':nPar_mean,'per_parents':pPar_mean,}}
        dict_combinations['mean'].update(to_be_updated)
    
    return dict_combinations



# Plotting


def recommend_sizes(links):
    """
    
    """
    linked_variables = find_linked_variables(links)
    n_linked_vars = len(linked_variables)
    print(f"n_linked_vars : {n_linked_vars}")
    if n_linked_vars <= 35:
        # Small
        figsize = (16, 16)
        node_size = 0.15
    elif n_linked_vars <= 70:
        # Medium
        figsize = (32, 32)
        node_size = 0.10
    else:
        # Big
        figsize = (48, 48)
        node_size = 0.05

    return figsize, node_size


# def scale_val_matrix(val_matrix):
#     """
#     Scales to values between -1 and 1, keeping zero in the same place
#     """
#     max_val = np.abs(val_matrix).max()
#     return val_matrix/max_val


def scale_link_width(o_link_width, threshold):
    """
    Scales 0.05 (so links are drawn) and 1 range, taking the threshold
    as the minimum
    """
    min_val = threshold
    # min_val = o_link_width[o_link_width >= threshold].min()
    smallest_link = 0.05
    link_width = o_link_width + smallest_link
    link_width -= min_val
    link_width[o_link_width < threshold] = 0
    return link_width / link_width.max()


def plot_aggregated_results(var_names_parents, aggregated_results, setup):
    Path(setup.plots_folder).mkdir(parents=True, exist_ok=True)
    dict_combinations = build_plot_matrices(var_names_parents, aggregated_results)

    for combination, combination_results in dict_combinations.items():
        print(combination)
        plot_filename = "{cfg}_{combination}.png".format(
            cfg=Path(setup.yml_filename).name.rsplit(".")[0], combination=combination
        )
        plot_file = Path(setup.plots_folder, plot_filename)
        if not setup.overwrite_plots and plot_file.is_file():
            print(f"Found file {plot_file}, skipping.")
            continue  # Ignore this result
        links = combination_results["links"]
        figsize, node_size = recommend_sizes(links)

        var_names = combination_results["var_names"]

        val_matrix = combination_results["val_matrix"]
        edges_array = np.array([val_matrix.min(), val_matrix.max()])
        vmin_edges = np.max(np.abs(edges_array)) * -1.0
        vmax_edges = np.max(np.abs(edges_array))
        edge_ticks = (
            vmax_edges - vmin_edges
        ) * 0.2  # So it has the same proportions as default
        link_width = combination_results["link_width"]
        # val_matrix = scale_val_matrix(val_matrix)
        link_width = scale_link_width(link_width, float(combination.threshold))

        plot_links(
            links,
            var_names,
            val_matrix=val_matrix,
            vmin_edges=vmin_edges,
            vmax_edges=vmax_edges,
            edge_ticks=edge_ticks,
            link_width=link_width,
            arrow_linewidth=10,
            save_name=plot_file,
            figsize=figsize,
            node_size=node_size,
            show_colorbar=True,
        )

        
def plot_causal_metrics(
    aggregated_results, 
    setup, 
    save=False,
):
    dict_combinations = build_links_metrics(setup, aggregated_results)
    
    # Filenm format (folder and figure name)
#     save_dir = setup.plots_folder
    save_dir = Path(setup.output_folder+'/'+setup.aggregate_folder+'/'+setup.plots_folder)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    variables = [var.name for var in setup.list_spcam if var.type == "out"]
    variables = '-'.join(variables)
    pc_alphas  = [str(iPC) for iPC in setup.pc_alphas]
    pcs       = '-'.join(pc_alphas)
    lats = [str(iL) for iL in setup.region[0]];  lats = '-'.join(lats)
    lons = [str(iL) for iL in setup.region[-1]]; lons = '-'.join(lons)
#     wgt_format = ['.png','_latwts.png'][setup.area_weighted]
    wgt_format = ['.pdf','_latwts.pdf'][setup.area_weighted]
    filenm = 'causal-links_metrics'+'_'+variables+'_a-'+pcs+'_'+wgt_format
    if save: save = str(save_dir)+'/'+filenm
    
    plot_links_metrics(setup, dict_combinations, save)
    
    
def get_matrix_idx(dict_vars_idx, inverted=False):
    idx_2d = 0; idx_3d = []; levels = []; child_main = []
    for i, key in dict_vars_idx.items():
        child_tmp = key
        if len(str(key).split('-')) == 1:
            idx_2d += 1
        else:
            child_main_tmp = str(key).split('-')[0]
            child_lev_tmp  = int(float(str(key).split('-')[-1]))
            child_lev_tmp  = 50 * round(child_lev_tmp/50) # Round to 50hPa for ticks
            
            if child_main_tmp not in child_main: child_main.append(child_main_tmp)
            levels.append(child_lev_tmp) 
    
    # For boxes separators
    if inverted==True:
        idx_vars = [i for i in range(len(levels))][::int(len(levels)/len(child_main))]
        idx_vars = np.array(idx_vars) + idx_2d -.5
        idx_3d = [i for i in range(len(levels))][::10]           # Get every 10 idx
    else:
        idx_vars = np.array([int(len(levels)/len(child_main)),int(len(levels))]) - .5
        idx_3d = [i for i in range(len(levels))][9::10]          # Get every 10 idx
    # x-Y ticks
    levels = np.array(levels)[idx_3d]                         # Get such levels
    levels = [str(i) for i in levels]                         # Get the ticks_labels
    idx_3d = np.array(idx_3d)+[+.5,idx_2d-.5][inverted==True] # Get the idx including 2D vars
    return child_main, idx_vars, idx_3d, levels


def get_matrix_insets_idx(dict_vars_idx, inverted=False, insets=False):
    idx_2d = 0; idx_3d = []; levels = []; child_main = []
    for i, key in dict_vars_idx.items():
        child_tmp = key
        if len(str(key).split('-')) == 1:
            idx_2d += 1
        else:
            child_main_tmp = str(key).split('-')[0]
            child_lev_tmp  = int(float(str(key).split('-')[-1]))
            child_lev_tmp  = 50 * round(child_lev_tmp/50) # Round to 50hPa for ticks

            if child_main_tmp not in child_main: child_main.append(child_main_tmp)
            levels.append(child_lev_tmp) 

    # For boxes separators
    if inverted==True or insets==True:
        idx_vars = [i for i in range(len(levels))][::int(len(levels)/len(child_main))]
        idx_vars = np.array(idx_vars) + idx_2d
    else:
        idx_vars = np.array([int(len(levels)/len(child_main)),int(len(levels))]) -[.5,0.][insets is True]
        
    # 3D ticks
    if inverted:
        idx_3d = [i for i in range(len(levels))][::10]           # Get every 10 idx
    else:
        idx_3d = [i for i in range(len(levels))][9::10]          # Get every 10 idx
        
    # x-Y ticks
    levels = np.array(levels)[idx_3d]                         # Get such levels
    levels = [str(i) for i in levels]                         # Get the ticks_labels
    
    if insets:
        idx_3d = np.array(idx_3d)-idx_2d+1. # Get the idx including 2D vars
    else:
        idx_3d = np.array(idx_3d)+[+.5,idx_2d][inverted==True] # Get the idx including 2D vars
    
    return child_main, idx_vars, idx_3d, levels


def get_matrix_2d_idx(dict_vars_idx):
    idx_2d = []; child_main = []
    for i, key in dict_vars_idx.items():
        if len(str(key).split('-')) == 1:
            child_main.append(str(key))
            idx_2d.append(i)
    return child_main, idx_2d


def plot_matrix_results(
    var_names_parents, 
    aggregated_results, 
    setup, 
    values='percentage',
    insets=False,
    threshold_dict=False,
    num_parents=False,
    random=False,
    save=False,
    masking=False,
    cmap=False,
):
    
    pltPath = Path(setup.output_folder+'/'+setup.aggregate_folder+'/'+setup.plots_folder)
    pltPath.mkdir(parents=True, exist_ok=True)

    var_names_parents_inv = var_names_parents[::-1]
    dict_inputs_idxs_inv  = {i:var_names_parents_inv[i] for i in range(len(var_names_parents_inv))}
    dict_outputs_idxs = {i:key for i, key in enumerate(aggregated_results.keys())}
    
    in_vars, in_box_idx, in_ticks, in_ticks_labs = get_matrix_insets_idx(dict_inputs_idxs_inv, inverted=True)
    out_vars, out_box_idx, out_ticks, out_ticks_labs = get_matrix_insets_idx(dict_outputs_idxs,insets=insets)

    # Using only two thresholds if more were given (min-max)
    if threshold_dict:
        thrs_labs  = 'optimized'
        thresholds = [thrs_labs]
    elif len(setup.thresholds) > 1:
        thresholds = [setup.thresholds[0],setup.thresholds[-1]]
        thrs_labs  = str(setup.thresholds[0])+'-'+str(setup.thresholds[-1])
    else:
        thresholds = setup.thresholds
        thrs_labs  = str(setup.thresholds[0])

    for iAlpha in setup.pc_alphas:
        iAlpha      = str(iAlpha)
        var_to_plot = np.ma.zeros([len(dict_outputs_idxs),len(dict_inputs_idxs_inv)])
        nparents    = np.ma.zeros([len(dict_outputs_idxs)])
        mask        = {}
        mask[str(thresholds[-1])] = np.ma.zeros([len(dict_outputs_idxs),len(dict_inputs_idxs_inv)])
        for i, output in dict_outputs_idxs.items():
            for j, jThrs in enumerate(thresholds):
                jThrs = str(jThrs)
                nparents[i] = aggregated_results[output][iAlpha]['num_parents'][jThrs]
                parents_tmp = aggregated_results[output][iAlpha]['parents'][jThrs]
                if values == 'percentage':
                    values_tmp  = aggregated_results[output][iAlpha]['parents_percent']
                    vmin = 0.; vmax = .9; cmap=[cmap,'Reds'][cmap is False]; extend='max'; cbar_label='ratio'
                elif values == 'val_matrix':
                    values_tmp  = aggregated_results[output][iAlpha]['val_matrix'][:,-1][:-1,1]
                    vmin = -.7; vmax = .7; cmap=[cmap,'bwr'][cmap is False]; extend='both'; cbar_label='r'
                if j == 0:
                    var_to_plot[i,:] = np.ma.masked_equal(
                        [[0.,val][i in parents_tmp] for i, val in enumerate(values_tmp)][::-1],
                        0.
                    )
                if j > 0 or len(thresholds) == 1:
                    mask[jThrs][i,:] = np.ma.masked_equal(
                        [[0.,1.][i in parents_tmp] for i, val in enumerate(values_tmp)][::-1],
                        1.
                    )
        
        # Normalization [-1,1]
        if values == 'val_matrix':
            var_to_plot = preprocessing.normalize(var_to_plot)
        
        if insets:
            out_vars_2d, out_vars_2d_ticks = get_matrix_2d_idx(dict_outputs_idxs)
            fig, ax = plot_matrix_insets(
                iAlpha,
                var_to_plot,
                in_vars,
                in_box_idx,
                in_ticks, 
                in_ticks_labs,
                out_vars,
                out_box_idx,
                out_ticks, 
                out_ticks_labs,
                out_vars_2d, 
                out_vars_2d_ticks,
                extend,
                cbar_label,
                dict_outputs_idxs=dict_outputs_idxs,
                mask=[False,mask][masking!=False],
                num_parents=[False,nparents][num_parents!=False],
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
        else:
            fig, ax = plot_matrix(
                iAlpha,
                var_to_plot,
                in_vars,
                in_box_idx,
                in_ticks, 
                in_ticks_labs,
                out_vars,
                out_box_idx,
                out_ticks, 
                out_ticks_labs,
                extend,
                cbar_label,
                mask=[False,mask][masking!=False],
                num_parents=[False,nparents][num_parents!=False],
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
        
        if save:
            fignm = str(pltPath)+'/'+f"matrix_pcalpha-{iAlpha}_{values}_thrs-{thrs_labs}"
            if num_parents:
                fignm = fignm+'_parnm'
            if random:
                fignm = fignm+'_random-links'
            if insets:
                fignm = fignm+'_insets'
            if masking:
                fignm = fignm+'.png'
            else:
                fignm = fignm+'_no-mask.png'
            print(f"Saving figure: {fignm}")
            fig.savefig(
                fignm, dpi=3000., format='png', metadata=None,
                bbox_inches='tight', pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None,
           )