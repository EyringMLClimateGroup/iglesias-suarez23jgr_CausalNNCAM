import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tigramite import plotting as tp
import numpy as np
import numpy.ma as ma
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
#     plt.legend(ncol=2,fontsize='medium')
    
    if save:
        sPath = save.split('/')[0]
        sName = save.split('/')[-1]
        Path(sPath).mkdir(parents=True, exist_ok=True)
        fig.savefig(
            save, dpi=1000, format=None, metadata=None,
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
    num_parents=False,
    **kwargs
):

    vars_labs_dict = {
        'tbp':'T (hPa)',
        'qbp':'q (hPa)',
        'vbp':'V (hPa)',
        'tphystnd':r'$\Delta$T$\mathregular{_{phy}}$ (hPa)',
        'phq':'$\Delta$q$\mathregular{_{phy}}$ (hPa)',
        # 'tphystnd':'dT/dt (hPa)',
        # 'phq':'dq/dt (hPa)',
    }
    
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib as mpl
    
    # mpl.rcParams['font.size']      = 12
    mpl.rcParams['axes.labelsize'] = 'large'
    
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
            # extend=extend,
        )

    I  = axes.imshow(matrix,**kwargs)
    cbar = plt.colorbar(I, ax=axes, extend=extend)
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
        axes.annotate(vars_labs_dict[iVar], xy=xy_coor[i], xycoords=trans, rotation=90,fontsize='large')
    axes.annotate('out-2Ds', xy=(-20., .02), xycoords=trans, rotation=0,fontsize='large')
    xy_coor = [(12., -.15),(42., -.15),(72., -.15)]
    for i, iVar in enumerate(in_vars):
        axes.annotate(vars_labs_dict[iVar], xy=xy_coor[i], xycoords=trans, rotation=0,fontsize='large')
    axes.annotate('in-2Ds', xy=(.6, -.2), xycoords=trans, rotation=90,fontsize='large')
    
    if isinstance(num_parents, np.ndarray):
        divider = make_axes_locatable(axes)
        axy = divider.append_axes("right", size="20%", pad=.5, sharey=axes)
        axy.plot(
            num_parents,
            np.arange(0.,len(num_parents),1),
            color='darkred',
            alpha=.8,
            linewidth=3.,
        )
        axy.set_xticks([0,50,100])
        axy.xaxis.set_tick_params(labelright=False)
        axy.yaxis.set_tick_params(labelleft=False)
        axy.set_xlabel('Num. Inputs')
        axy.get_yaxis().set_visible(False)
        axy.spines['top'].set_visible(False)
        axy.spines['right'].set_visible(False)
        axy.spines['bottom'].set_visible(True)
#         axy.spines['left'].set_visible(False)
        axy.set_xlim(-1.,100.)
    
    
    # fig.suptitle(pc_alpha)
    
    return fig, axes


def plot_matrix_insets(
    pc_alpha,
    raw_matrix,
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
    dict_outputs_idxs=False,
    mask=False,
    num_parents=False,
    inset_var='phq',
    **kwargs
):

    vars_labs_dict = {
        'tbp':'T (hPa)',
        'qbp':'q (hPa)',
        'vbp':'V (hPa)',
        'tphystnd':r'$\Delta$T$\mathregular{_{phy}}$ (hPa)',
        'phq':'$\Delta$q$\mathregular{_{phy}}$ (hPa)',
        'fsns':'$Q\mathregular{_{sw}^{srf}}$',
        'flns':'$Q\mathregular{_{lw}^{srf}}$',
        'fsnt':'$Q\mathregular{_{sw}^{top}}$',
        'flnt':'$Q\mathregular{_{lw}^{top}}$',
        'prect':'$P$',
    }
    
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib as mpl
    
    # mpl.rcParams['font.size']      = 12
    mpl.rcParams['axes.labelsize'] = 'large'
    
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
            # extend=extend,
        )

    matrix    = ma.zeros([raw_matrix.shape[0], raw_matrix.shape[-1]], dtype="d")
    matrix[:] = raw_matrix
    
    extent = (0, matrix.shape[-1], 0, matrix.shape[0])
        
    I  = axes.imshow(matrix,extent=extent,origin="upper",**kwargs)
    cbar = plt.colorbar(I, ax=axes, extend=extend)
    cbar.set_label(cbar_label)
    axes.set_xticks(in_ticks); axes.set_xticklabels(in_ticks_labs)
    axes.set_yticks(out_ticks[::-1]); axes.set_yticklabels(out_ticks_labs)
    axes.vlines(in_box_idx, ymin=-.5, ymax=len(matrix), color='k')
    axes.hlines(out_box_idx, xmin=-.5, xmax=len(matrix[0]), color='k')
    
    axes.set_xlim(xmin=0,xmax=len(matrix[0])-.5)
    axes.set_ylim(ymin=0)
    
    trans = axes.get_xaxis_transform()
    xy_coor = [(-15., .68),(-15., .20)]
    for i, iVar in enumerate(out_vars):
        axes.annotate(vars_labs_dict[iVar], xy=xy_coor[i], xycoords=trans, rotation=90,fontsize='large')
    axes.annotate('out-2Ds', xy=(-20., .02), xycoords=trans, rotation=0,fontsize='large')
    xy_coor = [(12., -.15),(42., -.15),(72., -.15)]
    for i, iVar in enumerate(in_vars):
        axes.annotate(vars_labs_dict[iVar], xy=xy_coor[i], xycoords=trans, rotation=0,fontsize='large')
    axes.annotate('in-2Ds', xy=(.6, -.2), xycoords=trans, rotation=90,fontsize='large')
    
    if isinstance(num_parents, np.ndarray):
        divider = make_axes_locatable(axes)
        axy = divider.append_axes("right", size="20%", pad=.5, sharey=axes)
        axy.plot(
            num_parents,
            np.arange(0.,len(num_parents),1),
            color='darkred',
            alpha=.8,
            linewidth=3.,
        )
        axy.set_xticks([0,50,100])
        axy.xaxis.set_tick_params(labelright=False)
        axy.yaxis.set_tick_params(labelleft=False)
        axy.set_xlabel('Num. Inputs')
        axy.get_yaxis().set_visible(False)
        axy.spines['top'].set_visible(False)
        axy.spines['right'].set_visible(False)
        axy.spines['bottom'].set_visible(True)
#         axy.spines['left'].set_visible(False)
        axy.set_xlim(-1.,100.)
    
    ## 3D inset
    ax3ins = zoomed_inset_axes(axes, 2.,
                              bbox_to_anchor=(1.015, 1.87),
                              bbox_transform=axes.transAxes)
    ax3ins.imshow(matrix,extent=extent,origin="upper",**kwargs)
    ax3ins.set_xticks(in_ticks); ax3ins.set_xticklabels(in_ticks_labs)
    ax3ins.set_yticks(out_ticks[::-1]); ax3ins.set_yticklabels(out_ticks_labs)
    if inset_var == 'phq':
        # sub region of the original image
        x1, x2, y1, y2 = 34, 55., 5., 26.
        ax3ins.set_xlim(x1, x2)
        ax3ins.set_ylim(y1, y2)
        trans = ax3ins.get_xaxis_transform()
        ax3ins.annotate(vars_labs_dict['phq'], xy=(28.,.4), xycoords=trans, rotation=90,fontsize='large')
        ax3ins.annotate(vars_labs_dict['qbp'], xy=(42.,-.18), xycoords=trans, rotation=0,fontsize='large')
    ax3ins.set_aspect(1.)
    mark_inset(axes, ax3ins, loc1=3, loc2=4, linewidth=3, ec='k', fc='none',linestyle='--',alpha=.7)
    # mark_inset(axes, ax3ins, loc1=4, loc2=1, fc="none", ec="white",linewidth=2.)

    ## 2D inset
    ax2ins = zoomed_inset_axes(axes, 4.5,
                              bbox_to_anchor=(2.774, -.25),
                              bbox_transform=axes.transAxes)
    ax2ins.imshow(matrix,extent=extent,origin="upper",**kwargs)
    out_vars_2d_ticks = [i+.5 for i in range(len(out_ticks))]
    ax2ins.set_yticks(out_vars_2d_ticks[::-1])#; ax2ins.set_yticklabels(out_vars_2d)
    trans = ax2ins.get_xaxis_transform()
    xy_coor = [(-8., .87),(-8., .66),(-8., .45),(-8., .24),(-7., .04)]
    for i, iVar in enumerate(out_vars_2d):
        ax2ins.annotate(vars_labs_dict[iVar], xy=xy_coor[i], xycoords=trans, rotation=0,fontsize='large')
    ax2ins.tick_params(bottom=False,labelbottom=False,labelleft=False)
    ax2ins.vlines(in_box_idx, ymin=-.5, ymax=len(matrix), color='k')
    y1, y2 = 0., 5.
    ax2ins.set_ylim(y1, y2)
    ax2ins.set_aspect(4.52)
    mark_inset(axes, ax2ins, loc1=2, loc2=1, linewidth=3, ec='k', fc='none',linestyle='--',alpha=.7)
    
    # fig.suptitle(pc_alpha)
    
    return fig, axes