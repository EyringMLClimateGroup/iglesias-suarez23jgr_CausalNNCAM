import numpy as np


def prepare_tigramite_data(list_var_data):
    T = len(list_var_data[0].data)
    N = len(list_var_data)

    spcam_data = np.zeros([T, N])
    var_names = list()
    parents = list()
    children = list()
    for i, var_data in enumerate(list_var_data):
        var = var_data.variable
        if var.type == "in":  # Parent
            parents.append(i)
        elif var.type == "out":  # Child
            children.append(i)

        var_names.append(var_data.name)

        spcam_data[:, i] = var_data.data
    return spcam_data, var_names, parents, children


def select_links(tau_min, tau_max, parents, children):
    """
    This function selects the causal links that will be tested by
    PCMCI. The links are selected such that per each variable in
    `children` all `parents` are stablished as causes, and no other
    causal relationships exist.
    
    Assumes `parents` and `children` are disjoint sets, and that all
    variables are included in the union of both sets.
    
    Parameters
    ----------
    tau_min : int
        Minimum time lag to test. Note that zero-lags are undirected.
    tau_max : int
        Maximum time lag. Must be larger or equal to tau_min.
    parents : set of int
        List of variables that will be assigned as a parent link.
        Assumed to be disjoint with children
    children : set of int
        List of variables that will be assigned a link from a parent.
        Assumed to be disjoint with parents
    Returns
    -------
    selected_links: dict
        Dictionary of selected links for Tigramite
        
    """

    parents = set(parents)
    children = set(children)

    selected_links = dict()
    # Set the default as all combinations of the selected variables
    for var in [*children, *parents]:
        if var in children:
            # Children can be caused only by parents and by themselves
            selected_links[var] = [
                (parent, -lag)
                for parent in parents
                for lag in range(tau_min, tau_max + 1)
            ]
        else:
            selected_links[var] = []

    return selected_links
