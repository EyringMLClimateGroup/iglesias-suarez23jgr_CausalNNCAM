import matplotlib.pyplot as plt
from tigramite import plotting as tp
import numpy as np


def plot_links(results,
               save_name = None,
               figsize = (16, 16),
               node_size = 0.15
              ):
    """
    This function is copied from the basic tutorial, but it may not be
    generalizable: It had issues with output from pcmciplus
    """

    link_matrix = results["link_matrix"]
    var_names = np.array(results["var_names"])
#     val_matrix = results['val_matrix']
    
    links = results["links"]
    linked_variables = set() # Avoids duplicates and sorts when list
    for child, parents_list in links.items():
        if len(parents_list) > 0:
            linked_variables.add(child)
            for parent in parents_list:
                linked_variables.add(parent[0])
    if len(linked_variables) != 0:
        linked_variables = list(linked_variables)
        # Raise an exception if there is a different number of links
        # on the reduced link_matrix
        assert((link_matrix == True).sum()
               == (link_matrix[linked_variables][:,linked_variables] == True).sum())
    else:
        linked_variables = range(len(links))
    
    tp.plot_graph(
        figsize = figsize,
#         val_matrix = val_matrix[linked_variables][:,linked_variables],
        link_matrix = link_matrix[linked_variables][:,linked_variables],
        var_names = var_names[linked_variables],
        link_colorbar_label = 'cross-MCI',
        node_colorbar_label = 'auto-MCI',
        arrow_linewidth = 5,
        node_size = node_size,
        save_name = save_name,
        show_colorbar = False
    ); plt.show()
