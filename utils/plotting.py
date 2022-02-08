import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from tigramite import plotting as tp
import numpy as np
from pathlib import Path


def find_linked_variables(links):
    linked_variables = set()  # Avoids duplicates and sorts when list
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

    link_matrix = np.array([[[False] * abs(min_lag - 1)] * size_matrix] * size_matrix)
    for child, link in links.items():
        for parent, lag in link:
            link_matrix[parent][child][abs(lag)] = True
    return link_matrix


def plot_links(
    links,
    var_names,
    val_matrix=None,
    vmin_edges=-1,
    vmax_edges=1,
    edge_ticks=0.4,
    link_width=None,
    arrow_linewidth=5,
    save_name=None,
    figsize=(16, 16),
    node_size=0.15,
    show_colorbar=False,
):
    """
    This function is copied from the basic tutorial, but it may not be
    generalizable: It had issues with output from pcmciplus
    """

    var_names = np.array(var_names)

    linked_variables = find_linked_variables(links)
    linked_variables.sort() # set() does not work with climate_invariant
    
    link_matrix = build_link_matrix(links)

    if len(linked_variables) != 0:
        # Raise an exception if there is a different number of links
        # on the reduced link_matrix
        filtered_link_matrix = link_matrix[linked_variables][:, linked_variables]
        assert (link_matrix == True).sum() == (filtered_link_matrix == True).sum()
        var_names = var_names[linked_variables]
        if val_matrix is not None:
            val_matrix = val_matrix[linked_variables][:, linked_variables]
        if link_width is not None:
            link_width = link_width[linked_variables][:, linked_variables]
    else:
        filtered_link_matrix = link_matrix

    tp.plot_graph(
        figsize=figsize,
        val_matrix=val_matrix,
        vmin_edges=vmin_edges,
        vmax_edges=vmax_edges,
        edge_ticks=edge_ticks,
        link_matrix=filtered_link_matrix,
        var_names=var_names,
        # Not MCI
        # link_colorbar_label = 'cross-MCI',
        # node_colorbar_label = 'auto-MCI',
        link_colorbar_label="cross",
        node_colorbar_label="auto",
        cmap_edges="RdBu_r",  #'Reds', # 'winter', #'RdBu_r',
        cmap_nodes="OrRd",
        link_width=link_width,
        arrow_linewidth=arrow_linewidth,
        node_size=node_size,
        save_name=save_name,
        show_colorbar=show_colorbar,
    )
    plt.show()

    
def plot_links_metrics(
    setup,
    dict_combinations,
    save=False,
    figsize=(6.4, 4.8),
    node_size=0.15,
    **kwargs
):
    
    pc_alphas  = [str(a) for a in setup.pc_alphas]
    thresholds = np.array(setup.thresholds)
    outputs_nm = [var.name for var in setup.list_spcam if var.type == "out"]
    
#     fig = plt.figure()
    fig, ax = plt.subplots(1, figsize=figsize)
    '''
    print(plt.style.available)
    ['Solarize_Light2', '_classic_test_patch', 'bmh', 'classic', 'dark_background', 
     'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn', 'seaborn-bright', 
     'seaborn-colorblind', 'seaborn-dark', 'seaborn-dark-palette', 'seaborn-darkgrid',
     'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel',
     'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid',
     'tableau-colorblind10']
    '''
#     plt.style.use('default')
#     plt.style.use('tableau-colorblind10') # Not really clear 
    plt.style.use('seaborn-pastel')
    
    linestyles = ['dotted','dashed','dashdot']
    colors     = ['blue','orange','red','purple','brown','pink','gray','olive','cyan']

    for i, iVar in enumerate(outputs_nm):
        for j, jPC in enumerate(pc_alphas):
            
            if len(dict_combinations[iVar][jPC]) > 1:
                for k, kLev in enumerate(dict_combinations[iVar][jPC].keys()):
                    if kLev != 'mean':
                        ax.plot(
                            thresholds,
                            dict_combinations[iVar][jPC][str(kLev)]['num_parents'],
                            linewidth=.2,
#                             linestyle='-',
#                             color='k',
                            linestyle=linestyles[j],
                            color=colors[i],
                            alpha=.8,
                        )
                        
            ax.plot(
                thresholds,
                dict_combinations[iVar][jPC]['mean']['num_parents'],
                linewidth=2.,
                linestyle=linestyles[j],
                color=colors[i],
                alpha=.8,
                label=iVar+' (\u03B1 '+str(jPC)+')',
            )
    
    for j, jPC in enumerate(pc_alphas):
        ax.plot(
            thresholds,
            dict_combinations['mean'][jPC]['num_parents'],
            linewidth=5.,
            linestyle=linestyles[j],
            color='k',
            alpha=.8,
            label='pc-alpha (\u03B1 '+str(jPC)+')',
        )
    

    
#     plt.xlim(thresholds[0],thresholds[-1])
    plt.ylim(0,100)
    
    plt.xlabel('Threshols (ratio)')
    plt.ylabel('Num. Causal links')
    
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_locator(MultipleLocator(.10))
    ax.xaxis.set_minor_locator(MultipleLocator(.05))
#     ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    
#     plt.legend(loc=0)
    plt.legend(ncol=3,bbox_to_anchor=(1.05, -.2))
    
    if save:
        sPath = save.split('/')[0]
        sName = save.split('/')[-1]
        Path(sPath).mkdir(parents=True, exist_ok=True)
        fig.savefig(
            save, dpi='figure', format=None, metadata=None,
            bbox_inches=None, pad_inches=0.1,
            facecolor='auto', edgecolor='auto',
            backend=None, **kwargs
       )
    
    plt.show()

    
def plot_matrix(
    pc_alpha,
    matrix,
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
    mask=False,
    **kwargs
):
    
    vars_labs_dict = {
        'tbp':'T (hPa)',
        'qbp':'Q (hPa)',
        'vbp':'V (hPa)',
        'tphystnd':'dT/dt (hPa)',
        'phq':'dQ/dt (hPa)',
    }
    
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 1, figsize=(12, 5))
    
    # Mask?
    if mask is not False:
        X, Y = np.meshgrid(np.arange(0,len(matrix[0]),1), np.arange(0,len(matrix),1))
        jThrs = list(mask.keys())[0]
        cs = axes.contourf(
            X,
            Y,
            mask[jThrs],
            colors='none',
            hatches='.',
            extend='both',
        )

    I  = axes.imshow(matrix,**kwargs)
    cbar = plt.colorbar(I, extend=extend)
    cbar.set_label(cbar_label)
    axes.set_xticks(in_ticks); axes.set_xticklabels(in_ticks_labs)
    axes.set_yticks(out_ticks); axes.set_yticklabels(out_ticks_labs)
    axes.vlines(in_box_idx, ymin=-.5, ymax=len(matrix), color='k')
    axes.hlines(out_box_idx, xmin=-.5, xmax=len(matrix[0]), color='k')
    
    axes.set_xlim(xmax=len(matrix[0])-.5)
    axes.set_ylim(ymin=len(matrix)-.5)
    
    trans = axes.get_xaxis_transform()
    xy_coor = [(-15., .68),(-15., .20)]
    for i, iVar in enumerate(out_vars):
        axes.annotate(vars_labs_dict[iVar], xy=xy_coor[i], xycoords=trans, rotation=90)
    axes.annotate('out-2Ds', xy=(-20., .02), xycoords=trans, rotation=0)
    xy_coor = [(12., -.15),(42., -.15),(72., -.15)]
    for i, iVar in enumerate(in_vars):
        axes.annotate(vars_labs_dict[iVar], xy=xy_coor[i], xycoords=trans, rotation=0)
    axes.annotate('in-2Ds', xy=(.6, -.2), xycoords=trans, rotation=90)
    
    fig.suptitle(pc_alpha)

    return fig, axes