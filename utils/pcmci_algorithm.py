import numpy                          as np
import utils.links                    as links
from utils.constants              import TAU_MIN, TAU_MAX, SIGNIFICANCE
import tigramite
from tigramite import data_processing as pp
from tigramite.pcmci              import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb


def run_pc_stable(pcmci, selected_links, list_pc_alpha):
    pc_alpha_results = dict()
    for pc_alpha in list_pc_alpha:
        try:
            links = pcmci.run_pc_stable(
                    tau_min = TAU_MIN,
                    tau_max = TAU_MAX,
                    selected_links = selected_links,
                    pc_alpha = pc_alpha
            )
            # "result_parents" will now be a dictionary with only the causal
            # parents for each variable like selected_links, but only with the
            # actual links

            link_matrix = pcmci.return_significant_links(
                    pq_matrix=pcmci.p_matrix,
                    val_matrix=pcmci.val_matrix,
                    alpha_level=pc_alpha
            )['link_matrix']

            results = {
                "links"       : links,
                "p_matrix"    : pcmci.p_matrix,
                "val_matrix"  : pcmci.val_matrix,
                "link_matrix" : link_matrix,
                "var_names"   : pcmci.var_names
            }
        except ValueError as e:
            print(e)
            results = {}
        pc_alpha_results[str(pc_alpha)] = results
        
    return pc_alpha_results


def run_pc_stable_2(pcmci, list_pc_alpha):
    pc_alpha_results = dict()
    for pc_alpha in list_pc_alpha:
        # A way to run this in parallel
        results_parents = dict()
        for j in children:
            results_parents[j] = pcmci_list.run_pc_stable_single(
                    j,
                    tau_min = TAU_MIN,
                    tau_max = TAU_MAX,
                    selected_links = selected_links[j],
                    pc_alpha = pc_alpha
            )
        pc_alpha_results[str(pc_alpha)] = results_parents


def run_pc_stable_3(pcmci, list_pc_alpha):
    # A way to run this in parallel
    results_parents = dict()
    for j in children:
        results_parents[j] = pcmci.run_pc_stable_single(
                j,
                tau_min = TAU_MIN,
                tau_max = TAU_MAX,
                selected_links = selected_links[j],
                pc_alpha = list_pc_alpha
        )
    # For each parent you may get a different pc_alpha        
        

def find_links(list_var_data, list_pc_alpha, verbosity = 0):
    spcam_data, var_names, parents, children = links.prepare_tigramite_data(
            list_var_data)
    
    selected_links = links.select_links(TAU_MIN, TAU_MAX, parents, children)
    
    # Initialize dataframe object, specify time axis and variable names
    dataframe = pp.DataFrame(
            spcam_data, 
            datatime = np.arange(len(spcam_data)),
            var_names= var_names
    )

    parcorr = ParCorr(significance = SIGNIFICANCE)
    pcmci = PCMCI(
            dataframe=dataframe, 
            cond_ind_test=parcorr,
            verbosity = verbosity
    )
    
    return run_pc_stable(pcmci, selected_links, list_pc_alpha)