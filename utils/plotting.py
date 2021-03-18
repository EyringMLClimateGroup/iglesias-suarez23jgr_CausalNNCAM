import matplotlib.pyplot as plt
from tigramite import plotting as tp
import numpy as np


def find_linked_variables(links):
    linked_variables = set() # Avoids duplicates and sorts when list
    for child, parents_list in links.items():
        if len(parents_list) > 0:
            linked_variables.add(child)
            for parent in parents_list:
                linked_variables.add(parent[0])
    return list(linked_variables)


def build_link_matrix(links):
    size_matrix = len(links)
    min_lag = 0
    for link in links.values():
        for parent, lag in link:
            if lag < min_lag:
                min_lag = lag
    
    link_matrix = np.array(
            [[[False] * abs(min_lag-1)] * size_matrix] * size_matrix)
    for child, link in links.items():
        for parent, lag in link:
            link_matrix[parent][child][abs(lag)] = True
    return link_matrix


def plot_links(links,
               var_names,
               val_matrix = None,
               vmin_edges = -1,
               vmax_edges = 1,
               edge_ticks = 0.4,
               link_width = None,
               arrow_linewidth = 5,
               save_name = None,
               figsize = (16, 16),
               node_size = 0.15,
               show_colorbar = False
              ):
    """
    This function is copied from the basic tutorial, but it may not be
    generalizable: It had issues with output from pcmciplus
    """
    
    var_names = np.array(var_names)
    
    linked_variables = find_linked_variables(links)
    
    link_matrix = build_link_matrix(links)
    
    if len(linked_variables) != 0:
        # Raise an exception if there is a different number of links
        # on the reduced link_matrix
        filtered_link_matrix = link_matrix[linked_variables][:,linked_variables]
        assert((link_matrix == True).sum()
               == (filtered_link_matrix == True).sum())
        var_names = var_names[linked_variables]
        if val_matrix is not None:
            val_matrix = val_matrix[linked_variables][:,linked_variables]
        if link_width is not None:
            link_width = link_width[linked_variables][:,linked_variables]
    else:
        filtered_link_matrix = link_matrix
    
    tp.plot_graph(
        figsize = figsize,
        val_matrix = val_matrix,
        vmin_edges = vmin_edges,
        vmax_edges = vmax_edges,
        edge_ticks = edge_ticks,
        link_matrix = filtered_link_matrix,
        var_names = var_names,
        # Not MCI
#         link_colorbar_label = 'cross-MCI',
#         node_colorbar_label = 'auto-MCI',
        link_colorbar_label = 'cross',
        node_colorbar_label = 'auto',
        link_width = link_width,
        arrow_linewidth = arrow_linewidth,
        node_size = node_size,
        save_name = save_name,
        show_colorbar = show_colorbar
    ); plt.show()
