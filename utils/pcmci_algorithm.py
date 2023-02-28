import numpy as np
import utils.links as links
from utils.constants import TAU_MIN, TAU_MAX
import tigramite
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI


def run_pc_stable(pcmci, selected_links, list_pc_alpha):
    pc_alpha_results = dict()
    for pc_alpha in list_pc_alpha:
        try:
            links = pcmci.run_pc_stable(
                tau_min=TAU_MIN,
                tau_max=TAU_MAX,
                selected_links=selected_links,
                pc_alpha=pc_alpha,
            )
            # "result_parents" will now be a dictionary with only the
            # causal parents for each variable like selected_links, but
            # only with the actual links

            # link_matrix = pcmci.return_significant_links(
            #     pq_matrix=pcmci.p_matrix,
            #     val_matrix=pcmci.val_matrix,
            #     alpha_level=pc_alpha
            # )['link_matrix']

            results = {
                "links": links,
                "p_matrix": pcmci.p_matrix,
                "val_matrix": pcmci.val_matrix,
                # "link_matrix" : link_matrix,
                "var_names": pcmci.var_names,
            }
        except ValueError as e:
            print(e)
            results = {}
        pc_alpha_results[str(pc_alpha)] = results

    return pc_alpha_results


def run_pc_stable_2(pcmci, list_pc_alpha):
    """
    NOTE: Undeveloped code
    """
    pc_alpha_results = dict()
    for pc_alpha in list_pc_alpha:
        # A way to run this in parallel
        results_parents = dict()
        for j in children:
            results_parents[j] = pcmci_list.run_pc_stable_single(
                j,
                tau_min=TAU_MIN,
                tau_max=TAU_MAX,
                selected_links=selected_links[j],
                pc_alpha=pc_alpha,
            )
        pc_alpha_results[str(pc_alpha)] = results_parents


def run_pc_stable_3(pcmci, list_pc_alpha):
    """
    NOTE: Undeveloped code
    """
    # A way to run this in parallel
    results_parents = dict()
    for j in children:
        results_parents[j] = pcmci.run_pc_stable_single(
            j,
            tau_min=TAU_MIN,
            tau_max=TAU_MAX,
            selected_links=selected_links[j],
            pc_alpha=list_pc_alpha,
        )
    # For each parent you may get a different pc_alpha


def find_links(list_var_data, list_pc_alpha, cond_ind_test, verbosity=0):
    spcam_data, var_names, parents, children = links.prepare_tigramite_data(
        list_var_data
    )

    selected_links = links.select_links(TAU_MIN, TAU_MAX, parents, children)

    # Initialize dataframe object, specify time axis and variable names
    dataframe = pp.DataFrame(
        spcam_data, datatime=np.arange(len(spcam_data)), var_names=var_names
    )

    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=verbosity)

    return run_pc_stable(pcmci, selected_links, list_pc_alpha)


def run_pearsonr(
    dataframe, 
    cond_ind_test,
    list_pc_alpha, 
    TAU_MIN, 
    TAU_MAX, 
    parents, 
    children
):

    parents = set(parents)
    children = set(children)
    pc_alpha_results = dict()

    for pc_alpha in list_pc_alpha:
        try:
            
            links = dict()
            p_matrix = np.zeros([len(dataframe.values[0]),len(dataframe.values[0]),2])
            r_matrix = np.zeros(p_matrix.shape)
            
            # Set the default as all combinations of the selected variables
            for var in [*parents, *children]:
                if var in children:
                    # Children can be caused only by parents and by themselves
                    links_tmp = []
                    for parent in [*parents, *children]:
                        
                        r_matrix[parent,-1,-1], p_matrix[parent,-1,-1] = cond_ind_test(
                            dataframe.values[:,parent],
                            dataframe.values[:,var]
                        )
                        
                        if p_matrix[parent,-1,-1] <= pc_alpha and parent != var:
                            [links_tmp.append((parent, -lag)) for lag in range(TAU_MIN, TAU_MAX + 1)]
                    links[var] = links_tmp
                else:
                    links[var] = []

            results = {
                "links": links,
                "p_matrix": p_matrix,
                "val_matrix": r_matrix,
                "var_names": dataframe.var_names,
            }
    
        except ValueError as e:
            print(e)
            results = {}
        
        pc_alpha_results[str(pc_alpha)] = results

    return pc_alpha_results


def pearsonr(list_var_data, list_pc_alpha, cond_ind_test, verbosity=0):

    spcam_data, var_names, parents, children = links.prepare_tigramite_data(
        list_var_data
    )

    # Initialize dataframe object, specify time axis and variable names
    dataframe = pp.DataFrame(
        spcam_data, datatime=np.arange(len(spcam_data)), var_names=var_names
    )
    
    return run_pearsonr(dataframe, cond_ind_test, list_pc_alpha, TAU_MIN, TAU_MAX, parents, children)