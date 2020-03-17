
import os
import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from matplotlib import collections
import matplotlib.cbook as cb
from matplotlib.colors import colorConverter, Colormap
from matplotlib.patches import FancyArrowPatch, Circle, ArrowStyle
import matplotlib.text as mpl_text
import networkx as nx

import qmla.get_growth_rule

__all__ = [
    'plot_tree_multiple_instances', 
    'plot_qmla_single_instance_tree'
]


def plot_tree_multiple_instances(
    results_csv,
    all_bayes_csv,
    latex_mapping_file,
    avg_type='medians',
    growth_generator=None,
    entropy=None,
    inf_gain=None,
    save_to_file=None
):
    try:
        # qmd_res = pd.DataFrame.from_csv(
        qmd_res = pd.read_csv(
            results_csv,
            index_col='LatexName'
        )
    except ValueError:
        print(
            "Latex Name not in results CSV keys.",
            "There aren't enough data for a tree of multiple QMD."
            "This may be because this run was for QHL rather than QMD."
        )
        raise

    mods = list(qmd_res.index)
    winning_count = {}
    for mod in mods:
        winning_count[mod] = mods.count(mod)

    cumulative_qmla_tree_plot(
        cumulative_csv=all_bayes_csv,
        wins_per_mod=winning_count,
        latex_mapping_file=latex_mapping_file,
        growth_generator=growth_generator,
        only_adjacent_branches=True,
        avg=avg_type,
        entropy=entropy,
        inf_gain=inf_gain,
        save_to_file=save_to_file
    )


def get_averages_from_combined_results(
    all_bayes_csv,
    growth_generator=None
):
    import csv
    import pandas
    cumulative_bayes = pandas.read_csv(all_bayes_csv)
    names = list(cumulative_bayes.keys())

    count_bayes = {}
    mod_names = list(cumulative_bayes.keys())

    for mod in mod_names:
        count_bayes[mod] = {}
        model_results = cumulative_bayes[mod]
        for comp_mod in mod_names:
            try:
                num_bayes = model_results[comp_mod].count()
            except BaseException:
                num_bayes = 0
            count_bayes[mod][comp_mod] = num_bayes

    piv = pandas.pivot_table(
        cumulative_bayes,
        index='ModelName',
        values=names,
        aggfunc=[np.mean, np.median]
    )

    means = piv['mean']
    medians = piv['median']

    b = means.apply(lambda x: x.dropna().to_dict(), axis=1)
    means_dict = b.to_dict()

    c = medians.apply(lambda x: x.dropna().to_dict(), axis=1)
    medians_dict = c.to_dict()

    return means_dict, medians_dict, count_bayes


def cumulative_qmla_tree_plot(
    cumulative_csv,
    wins_per_mod,
    latex_mapping_file,
    avg='means',
    only_adjacent_branches=True,
    growth_generator=None,
    directed=True,
    entropy=None,
    inf_gain=None,
    save_to_file=None
):
    import networkx as nx
    import copy
    import csv
    means, medians, counts = get_averages_from_combined_results(
        cumulative_csv,
        growth_generator=growth_generator
    )
    if avg == 'means':
        # print("[cumulative_qmla_tree_plot] USING MEANS")
        # print(means)
        bayes_factors = means  # medians
    elif avg == 'medians':
        # print("[cumulative_qmla_tree_plot] USING MEDIANS")
        # print(medians)
        bayes_factors = medians

    print("[cumulative_qmla_tree_plot] COUNTS", counts)

    max_bayes_factor = max([max(bayes_factors[k].values())
                            for k in bayes_factors.keys()])
    growth_class = qmla.get_growth_rule.get_growth_generator_class(
        growth_generation_rule=growth_generator
    )
    true_model = growth_class.true_model_latex()

    term_branches = growth_class.name_branch_map(
        latex_mapping_file=latex_mapping_file,
    )

    modlist = csv.DictReader(open(cumulative_csv)).fieldnames
    if 'ModelName' in modlist:
        modlist.remove('ModelName')

    pair_freqs = {}
    for c in list(counts.keys()):
        for k in list(counts[c].keys()):
            this_edge = (c, k)
            pair_freqs[this_edge] = counts[c][k]

    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    positions = {}
    branch_x_filled = {}
    branch_mod_count = {}

#    max_branch_id = qmd.branch_highest_id # TODO: get this number without access to QMD class instance
    # max_branch_id = 9 # TODO: this is hardcoded - is there an alternative?
    max_branch_id = max(list(term_branches.values())) + 1
    max_mod_id = len(modlist)
    for i in range(max_branch_id + 1):
        branch_x_filled[i] = 0
        branch_mod_count[i] = 0

    colour_by_node_name, colour_by_count = (
        colour_dicts_from_win_count(
            wins_per_mod,
            latex_mapping_file=latex_mapping_file,
            growth_generator=growth_generator,
            min_colour_value=0.4
        )
    )
    min_colour = min(list(colour_by_node_name.values()))

    for m in modlist:
        branch = term_branches[m]
        branch_mod_count[branch] += 1
        G.add_node(m)
        G.nodes[m]['label'] = str(m)

        if m == true_model:
            G.nodes[m]['relation_to_true_model'] = 'true'
        else:
            G.nodes[m]['relation_to_true_model'] = 'none'

        try:
            G.nodes[m]['status'] = colour_by_node_name[m]
            G.nodes[m]['wins'] = wins_per_mod[m]
            G.nodes[m]['info'] = wins_per_mod[m]

        except BaseException:
            G.nodes[m]['wins'] = 0
            G.nodes[m]['status'] = min_colour
            G.nodes[m]['info'] = 0

    print("[cumulative_qmla_tree_plot] nodes added.")
    max_num_mods_any_branch = max(list(branch_mod_count.values()))
    # get the cordinates to display this model's node at

    for m in modlist:

        branch = term_branches[m]
        num_mods_this_branch = branch_mod_count[branch]
        pos_list = available_position_list(
            num_mods_this_branch,
            max_num_mods_any_branch
        )
        branch_filled_so_far = branch_x_filled[branch]
        branch_x_filled[branch] += 1

        x_pos = pos_list[branch_filled_so_far]
        y_pos = branch
        positions[m] = (x_pos, y_pos)
        G.node[m]['pos'] = (x_pos, y_pos)

    print("[cumulative_qmla_tree_plot] node positions added.")
    sorted_positions = sorted(positions.values(), key=lambda x: (x[1], x[0]))
    mod_id = 0
    model_ids_names = {}
    pos_keys = list(positions.keys())
    for p in sorted_positions:
        mod_id += 1
        idx = list(positions.values()).index(p)
        corresponding_model = pos_keys[idx]
        G.node[corresponding_model]['mod_id'] = mod_id
        model_ids_names[mod_id] = G.node[m]['label']

    edges = []
    edge_frequencies = []
    max_frequency = max(list(pair_freqs.values()))

    try:
        low = int(np.percentile(range(0, max_frequency), q=20))
        mid = int(np.percentile(range(0, max_frequency), q=60))
        high = max_frequency
    except BaseException:
        low = max_frequency
        mid = max_frequency
        high = max_frequency

    # frequency_markers = list(np.linspace(0, max_frequency, 4, dtype=int))
    print("[cumulative_qmla_tree_plot] setting edges.")
    print("[cumulative_qmla_tree_plot] modelist:", modlist)
    # only_adjacent_branches = False

    even_arrow_width = True
    # setting the thickness if the arrows
    for a in modlist:
        remaining_modlist = modlist[modlist.index(a) + 1:]
        for b in remaining_modlist:
            is_adj = global_adjacent_branch_test(
                a,
                b,
                term_branches
            )
            if is_adj or not only_adjacent_branches:
                if a != b:
                    pairing = (a, b)
                    try:
                        # frequency = pair_freqs[pairing]/max_frequency
                        frequency = pair_freqs[pairing]
                        if frequency < low:
                            frequency = 1  # thin
                        elif low <= frequency < mid:
                            frequency = 10  # medium
                        else:
                            frequency = 50  # thick
                    except BaseException:
                        print("couldn't assign frequency")
                        frequency = 1

                    edges.append(pairing)
                    print("pair {} freq {}".format(pairing, frequency))
                    edge_frequencies.append(frequency)

                    vs = [a, b]

                    try:
                        bf = bayes_factors[a][b]
                    except BaseException:
                        bf = 0

                    if bf != 0:
                        if bf < 1:  # ie model b has won
                            bf = float(1 / bf)
                            weight = np.log10(bf)
                            winner = b
                            loser = a
                        else:
                            weight = np.log10(bf)
                            winner = a
                            loser = b

                        # thisweight = np.log10(bayes_factors[a][b])
                        try:
                            print(
                                "\n\t pair {},  \
                                \n\t BF[a,b]:{} \
                                \n\t BF[b,a]:{}\
                                \n\t weight:{} \
                                \n\t bf:{} \
                                \n\t freq:{} \
                                \n\t pair freq {}".format(
                                    pairing,
                                    str(bayes_factors[a][b]),
                                    str(bayes_factors[b][a]),
                                    str(weight),
                                    str(bf),
                                    frequency,
                                    pair_freqs[pairing]
                                )
                            )
                            G.add_edge(
                                loser,
                                winner,
                                weight=weight,
                                winner=winner,
                                loser=loser,
                                # flipped=flipped,
                                adj=is_adj,
                                freq=frequency
                            )
                        except BaseException:
                            print(
                                "[plotQMD - cumulative_qmla_tree_plot] failed to add edge", pairing
                            )
                            raise

                    elif bf == 0:
                        weight = 0
            else:
                print("not adding edge {}/{}".format(a, b))

    print("[cumulative_qmla_tree_plot] edges added.")
    print("edge freqs:", edge_frequencies)
    max_freq = max(edge_frequencies)
    # print("freq markers:", frequency_markers)
    print("[plotQMD]max freq:", max_freq)

    # try:
    #     freq_scale = 10/max_freq
    # except:
    #     freq_scale = 0

    freq_scale = 1
    edge_f = [i * freq_scale for i in edge_frequencies]

    arr = np.linspace(0, 50, 100).reshape((10, 10))
    cmap = plt.get_cmap('viridis')
    cmap = plt.cm.Blues
    # cmap = plt.cm.rainbow
    new_cmap = truncate_colormap(cmap, 0.35, 1.0)
    # new_cmap = cmap

    plot_qmla_tree(
        G,
        n_cmap=plt.cm.pink_r,
        e_cmap=new_cmap,
        # e_cmap = plt.cm.Blues,
        # e_cmap = plt.cm.Paired,
        # e_cmap = plt.cm.rainbow,
        nonadj_alpha=0.0,
        e_alphas=[],
        # widthscale=10.5,
        widthscale=3,
        label_padding=0.4,
        pathstyle="curve",
        arrow_size=None,
        entropy=None,
        inf_gain=None
    )

    if save_to_file is not None:
        plt.savefig(save_to_file, bbox_inches='tight')
    return G, edges, edge_f

def plot_qmla_tree(
    G,
    n_cmap,
    e_cmap,
    e_alphas=[], nonadj_alpha=0.1,
    label_padding=0.4,
    arrow_size=0.02, widthscale=1.0,
    entropy=None, inf_gain=None,
    pathstyle="straight",
    id_labels=True,
    save_to_file=None
):
    plt.clf()
    plt.figure(figsize=(6, 11))

    directed = nx.is_directed(G)

    if int(nx.__version__[0]) >= 2:
        list_of_edges = list(G.edges(data=True))

    edge_tuples = tuple(G.edges())

    positions = dict(zip(G.nodes(), tuple([prop['pos'] for
                                           (n, prop) in G.nodes(data=True)]))
                     )
    n_colours = tuple(
        [
            n_cmap(prop['status']) for (n, prop) in G.nodes(data=True)
        ]
    )

    label_positions = []
    if id_labels is True:
        labels = dict(
            zip(
                G.nodes(),
                tuple([prop['mod_id'] for (n, prop) in G.nodes(data=True)])
                # tuple(  [n for (n,prop) in G.nodes(data=True)]  )
            )
        )
        for key in positions.keys():
            label_positions.append(
                tuple(np.array(positions[key]) - np.array([0., 0.])
                      )
            )
    else:
        labels = dict(
            zip(
                G.nodes(),
                tuple([prop['label'] for (n, prop) in G.nodes(data=True)])
            )
        )
        for key in positions.keys():
            label_positions.append(tuple(np.array(positions[key]) - np.array([0., label_padding]))
                                   )

    label_positions = dict(
        zip(positions.keys(), tuple(label_positions))
    )

    if len(e_alphas) == 0:
        for idx in range(len(edge_tuples)):
            e_alphas.append(
                0.8 if list_of_edges[idx][2]["adj"]
                else nonadj_alpha
            )
    weights = tuple(
        [prop['weight'] for (u, v, prop) in list_of_edges]
    )

    nx.draw_networkx_labels(
        G,
        label_positions,
        labels,
        font_color='black',
        font_weight='bold'
    )

    plt.tight_layout()
    plt.gca().invert_yaxis()  # so branch 0 on top
    plt.gca().get_xaxis().set_visible(False)
    plt.ylabel('Branch')

    xmin = min(np.array(list(label_positions.values()))[:, 0])
    xmax = max(np.array(list(label_positions.values()))[:, 0])
    plt.xlim(xmin - 0.8, xmax + 0.8)

    nodes = list(G.nodes)
    distinct_status = []

    model_ids_names = {}

    labels = []
    handles = []
    for n in nodes:
        model_ids_names[G.nodes[n]['mod_id']] = G.nodes[n]['label']
        # only for status not yet represented in legend
        stat = G.nodes[n]['status']
        if stat not in distinct_status:
            distinct_status.append(stat)
            # node colour encodes either number wins or branch champion
            info = str(G.nodes[n]['wins'])
            col = tuple(n_cmap(G.nodes[n]['status']))
            handles.append(mpatches.Patch(color=col))
            labels.append(info)
    labels, handles = zip(
        *sorted(zip(labels, handles), key=lambda t: int(t[0])))
    lgd_handles = []

    # if 'Branch Champion' in labels:
    #     legend_title='Champion Type'
    # else:
    #     legend_title='# QMD wins'
    # plt.legend(handles, labels, title=legend_title)
    if 'Branch Champion' in labels:
        legend_title = 'Champion Type'
    else:
        legend_title = '# QMD wins'

    legend_num_wins = plt.legend(
        handles,
        labels,
        title=legend_title,
        # mode="expand",
        ncol=min(6, len(handles)),
        loc='lower center'
    )

    mod_id_handles = list(sorted(list(model_ids_names.keys())))
    mod_id_labels = [model_ids_names[k] for k in mod_id_handles]

    mod_id_labels, mod_id_handles = zip(
        *sorted(
            zip(
                mod_id_labels,
                mod_id_handles
            ),
            key=lambda t: t[0]
        )
    )

    model_handles = []
    model_labels = []
    handler_map = {}

    # mod_handle = textHandleModelID(
    #     "ID",
    #     "black"
    # )
    # model_handles.append(mod_handle)
    # handler_map[mod_handle] = textObjectHandler()

    for mid in mod_id_handles:
        mod_str = model_ids_names[mid]
        num_wins = G.nodes[mod_str]['wins']
        relation_to_true_model = G.nodes[mod_str]['relation_to_true_model']

        mod_lab = "({}) \t {}".format(
            num_wins,
            str(mod_str)
        )

        model_labels.append(mod_lab)
        mod_colour = "black"  # if true/champ can change colour
        # mod_colour =n_cmap(G.nodes[mod_str]['status'])
        mod_handle = textHandleModelID(
            mid,
            relation_to_true_model,
            num_wins,
            mod_colour
        )
        model_handles.append(mod_handle)
        handler_map[mod_handle] = textObjectHandler()

    model_labels, model_handles = zip(
        *sorted(
            zip(
                model_labels,
                model_handles
            ),
            key=lambda t: int(t[1].model_id)
        )
    )

    model_legend_title = str("ID    (Wins)     Model")

    node_boundary_colours = []

    # for n in G.nodes:
    #     if G[n]['relation_to_true_model'] == 'true':
    #         node_boundary_colours.append('green')
    #     else:
    #         node_boundary_colours.append('black')

    nx.draw_networkx_nodes(
        G,
        with_labels=True,  # labels=labels,
        pos=positions,
        k=1.5,  # node spacing
        width=None,
        alpha=0.5,
        node_size=700,  # node_shape='8',
        node_color=n_colours,
        edgecolors=node_boundary_colours,
    )

    edges_for_cmap = draw_networkx_arrows(
        G,
        edgelist=edge_tuples,
        pos=positions,
        arrows=True,
        arrowstyle='->',
        width=arrow_size,
        widthscale=widthscale,
        pathstyle=pathstyle,
        alphas=e_alphas,
        edge_color=weights,
        edge_cmap=e_cmap,
        edge_vmin=None,  # 0.8,
        edge_vmax=None,  # 0.85
    )

    plt.legend(
        model_handles,
        model_labels,
        bbox_to_anchor=(0.5, 1.0, 1, 0),
        # bbox_to_anchor=(1.1, 1.05),
        handler_map=handler_map,
        loc=1,
        title=model_legend_title
    )._legend_box.align = 'left'

    plt.gca().add_artist(legend_num_wins)
    plt.colorbar(
        edges_for_cmap,
        orientation="horizontal",
        pad=0,
        label=r'$\log_{10}$ Bayes factor'
    )

    plot_title = str(
        "Quantum Model Development Tree"
    )
    plt.title(plot_title)

    if save_to_file is not None:
        plt.savefig(save_to_file, bbox_inches='tight')


class textHandleModelID(object):
    def __init__(
        self,
        model_id,
        relation_to_true_model,
        num_wins,
        color
    ):

        self.model_id = model_id
        self.num_wins = num_wins
        self.relation_to_true_model = relation_to_true_model
        if num_wins < 0:
            self.my_text = str(
                "{} ({}) ".format(model_id, num_wins)
            )
        else:
            self.my_text = str("{}".format(model_id))

        if relation_to_true_model == 'true':
            self.my_color = 'green'
            self.text_weight = 'bold'
        else:
            self.my_color = color
            self.text_weight = 'normal'


class textObjectHandler(object):
    def legend_artist(
        self,
        legend,
        orig_handle,
        fontsize,
        handlebox
    ):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpl_text.Text(
            x=0, y=0,
            text=orig_handle.my_text,
            color=orig_handle.my_color,
            fontweight=orig_handle.text_weight,
            verticalalignment=u'baseline',
            horizontalalignment=u'left',
            multialignment=None,
            fontproperties=None,
            #             rotation=45,
            linespacing=None,
            rotation_mode=None
        )
        handlebox.add_artist(patch)
        return patch


def colour_dicts_from_win_count(
    winning_count,
    latex_mapping_file,
    growth_generator=None,
    min_colour_value=0.1
):
    growth_class = qmla.get_growth_rule.get_growth_generator_class(
        growth_generation_rule=growth_generator
    )

    max_wins = max(list(winning_count.values()))
    min_wins = min(list(winning_count.values()))
    min_col = min_colour_value
    max_col = 0.9

    win_count_vals = list(winning_count.values())
    win_count_vals.append(0)
    distinct_win_vals = sorted(list(set(win_count_vals)))
    num_colours = len(distinct_win_vals)
    col_space = np.linspace(min_col, max_col, num_colours)
    colour_by_win_count = {}
    colour_by_node_name = {}

    all_models = growth_class.name_branch_map(
        latex_mapping_file=latex_mapping_file,
    ).keys()

    # print("colour dict function. all_models:\n", all_models)

    for k in range(num_colours):
        colour_by_win_count[k] = col_space[k]

    for k in all_models:
        try:
            num_wins = winning_count[k]
        except BaseException:
            num_wins = 0
        idx = distinct_win_vals.index(num_wins)
        colour_by_node_name[k] = colour_by_win_count[idx]
    return colour_by_node_name, colour_by_win_count




def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def draw_networkx_arrows(
    G,
    pos,
    edgelist=None,
    nodedim=0.,
    width=0.02,  # width=0.02, 1.0
    widthscale=1.0,
    edge_color='k',
    style='solid',
    alphas=1.,
    edge_cmap=None,
    edge_vmin=None,
    edge_vmax=None,
    ax=None,
    label=[None],
    pathstyle='straight',
    **kwds
):
    if ax is None:
        ax = plt.gca()

    if edgelist is None:
        edgelist = G.edges()

    if width is None:
        try:
            widthlist = np.array(
                list(
                    [(widthscale * prop['freq'])
                     for (u, v, prop) in G.edges(data=True)]
                )
            )
            widthlist = widthscale * widthlist / np.max(widthlist)
            # widthlist = [(a+widthscale*0.1) for a in widthlist] ## this was
            # giving colour to non-existent edges
        except BaseException:
            #            widthlist = widthscale*0.02
            widthlist = widthscale

    else:
        widthlist = width

    if not edgelist or len(edgelist) == 0:  # no edges!
        return None

    if len(alphas) < len(edgelist):
        alphas = np.repeat(alphas, len(edgelist))

    # set edge positions
    edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])

    if not cb.iterable(widthlist):
        lw = (widthlist,)
    else:
        lw = widthlist

    if (
        # not cb.is_string_like(edge_color)
        type(edge_color) != str
        and cb.iterable(edge_color)
        and len(edge_color) == len(edge_pos)
    ):
        if np.alltrue(
            [type(c) == str for c in edge_color]
        ):
            # (should check ALL elements)
            # list of color letters such as ['k','r','k',...]
            edge_colors = tuple([colorConverter.to_rgba(c)
                                 for c in edge_color])
        elif np.alltrue(
            # [not cb.is_string_like(c) for c in edge_color]
            [type(c) != str for c in edge_color]
        ):
            # If color specs are given as (rgb) or (rgba) tuples, we're OK
            if np.alltrue(
                [cb.iterable(c) and len(c) in (3, 4) for c in edge_color]
            ):
                edge_colors = tuple(edge_color)
            else:
                # numbers (which are going to be mapped with a colormap)
                edge_colors = None
        else:
            raise ValueError('edge_color must consist of \
                either color names or numbers'
                             )
    else:
        if (
            # cb.is_string_like(edge_color)
            type(edge_color) == str
            or len(edge_color) == 1
        ):
            edge_colors = (colorConverter.to_rgba(edge_color), )
        else:
            raise ValueError('edge_color must be a single color or \
            list of exactly m colors where m is the number or edges'
                             )

    edge_collection = collections.LineCollection(
        edge_pos,
        colors=edge_colors,
        linewidths=lw
    )
    edge_collection.set_zorder(1)  # edges go behind nodes

    # ax.add_collection(edge_collection)

    if edge_colors is None:
        if edge_cmap is not None:
            assert(isinstance(edge_cmap, Colormap))
        edge_collection.set_array(np.asarray(edge_color))
        edge_collection.set_cmap(edge_cmap)
        if edge_vmin is not None or edge_vmax is not None:
            edge_collection.set_clim(edge_vmin, edge_vmax)
        else:
            edge_collection.autoscale()

    max_bayes_value = max(edge_collection.get_clim())
    edge_collection.set_clim(0.5, max_bayes_value)
    # for i in range(len(edgelist)):
    #     print(edgelist[i], ":", edge_color[i])

    for n in G:
        c = Circle(pos[n], radius=0.02, alpha=0.5)
        ax.add_patch(c)
        G.node[n]['patch'] = c
        x, y = pos[n]
    seen = {}

    # Rescale all weights between 0,1 so cmap can find the appropriate RGB
    # value.
    offset = 0.7
    norm_edge_color = edge_color / max_bayes_value

    # print("all color cmap values", norm_edge_color)

    if G.is_directed():
        seen = {}
        for idx in range(len(edgelist)):
            if not cb.iterable(widthlist):
                lw = widthlist
            else:
                lw = widthlist[idx]

            arrow_colour = edge_cmap(norm_edge_color[idx])

            if pathstyle is "straight":
                (src, dst) = edge_pos[idx]
                x1, y1 = src
                x2, y2 = dst
                delta = 0.2
                theta = np.arctan((y2 - y1) / (x2 - x1))
                # print(theta)
                if x1 == x2:
                    dx = x2 - x1
                    dy = y2 - y1 - np.sign(y2 - y1) * delta
                elif y1 == y2:
                    dx = x2 - x1 - np.sign(x2 - x1) * delta
                    dy = y2 - y1
                else:
                    dx = x2 - x1 - \
                        np.sign(x2 - x1) * np.abs(np.cos(theta)
                                                  * delta)   # x offset
                    dy = y2 - y1 - \
                        np.sign(y2 - y1) * np.abs(np.sin(theta)
                                                  * delta)   # y offset

                thislabel = None if len(label) < len(edgelist) else label[idx]

                ax.arrow(
                    x1, y1, dx, dy,
                    facecolor=arrow_colour,
                    alpha=alphas[idx],
                    linewidth=0,
                    antialiased=True,
                    width=lw,
                    head_width=5 * lw,
                    overhang=-5 * 0.02 / lw,
                    length_includes_head=True,
                    label=thislabel, zorder=1
                )

            elif pathstyle is "curve":

                (u, v) = edgelist[idx]
                # (u,v,prop) = prop['weight'] for  in list_of_edges
                # flipped = G.edge[(u,v)]

                winner = G.edges[(u, v)]['winner']
                loser = G.edges[(u, v)]['loser']

                n1 = G.node[winner]['patch']
                n2 = G.node[loser]['patch']

                # n1=G.node[loser]['patch']
                # n2=G.node[winner]['patch']

                # n1=G.node[u]['patch']
                # n2=G.node[v]['patch']

                rad = 0.1

                if (u, v) in seen:
                    rad = seen.get((u, v))
                    rad = (rad + np.sign(rad) * 0.1) * -1
                alpha = 0.5

                kwargs = {
                    # 'head_width': 5*lw,
                    'facecolor': arrow_colour[0:3] + (alphas[idx],),
                    'edgecolor': (0, 0, 0, 0.)
                    # 'overhang':-5*0.02/lw,
                    # 'length_includes_head': True,
                    # capstyle='projecting',
                }

                # Can be accepted by fancy arrow patch to alter arrows
                arrow_style = ArrowStyle.Wedge(
                    tail_width=lw,
                    shrink_factor=0.4
                )

                # arrow_style = mpatches.ArrowStyle.Curve(
                # )

                e = FancyArrowPatch(
                    n1.center,
                    n2.center,
                    patchA=n1,
                    patchB=n2,
                    arrowstyle=arrow_style,
                    # arrowstyle='simple',
                    # arrowstyle='curveb',
                    connectionstyle='arc3,rad=%s' % rad,
                    mutation_scale=5.0,
                    # alpha=0.5,
                    lw=lw,  # AROUND 10 TO BE FEASIBLE
                    **kwargs
                )
                seen[(u, v)] = rad
                ax.add_patch(e)

    # print("rad", rad)
    # print("Node coordinates", n1, n2)
    # print("arrowcolor", arrow_colour)

    # update view
    minx = np.amin(np.ravel(edge_pos[:, :, 0]))
    maxx = np.amax(np.ravel(edge_pos[:, :, 0]))
    miny = np.amin(np.ravel(edge_pos[:, :, 1]))
    maxy = np.amax(np.ravel(edge_pos[:, :, 1]))

    w = maxx - minx
    h = maxy - miny
    padx, pady = 0.05 * w, 0.05 * h
    corners = (minx - padx, miny - pady), (maxx + padx, maxy + pady)
    ax.update_datalim(corners)
    ax.autoscale_view()

    return edge_collection

def adjacent_branch_test(qmd, mod1, mod2):
    mod_a = qmd.get_model_storage_instance_by_id(mod1).Name
    mod_b = qmd.get_model_storage_instance_by_id(mod2).Name
    br_a = qmd.get_model_data_by_field(name=mod_a, field='branch_id')
    br_b = qmd.get_model_data_by_field(name=mod_b, field='branch_id')

    diff = br_a - br_b
    if diff in [-1, 0, 1]:
        return True
    else:
        return False


def global_adjacent_branch_test(a, b, term_branches):
    branch_a = int(term_branches[a])
    branch_b = int(term_branches[b])

    available_branches = sorted(list(set(term_branches.values())))
    branch_a_idx = available_branches.index(branch_a)
    branch_b_idx = available_branches.index(branch_b)

    # closeness conditions
    c1 = (branch_a_idx == branch_b_idx)
    c2 = (branch_a_idx == branch_b_idx + 1)
    c3 = (branch_a_idx == branch_b_idx - 1)

    if (
        c1 == True
        or c2 == True
        or c3 == True
    ):
        return True
    else:
        return False


def available_position_list(max_this_branch, max_any_branch):
    # Used to get a list of positions to place nodes centrally
    N = 2 * max_any_branch - 1
    all_nums = list(range(N))
    evens = [a for a in all_nums if a % 2 == 0]
    odds = [a for a in all_nums if a % 2 != 0]

    diff = max_any_branch - max_this_branch
    if diff % 2 == 0:
        all_positions = evens
        even_odd = 'even'
    else:
        all_positions = odds
        even_odd = 'odd'

    if diff > 1:
        if even_odd == 'even':
            to_cut = int(diff / 2)
            available_positions = all_positions[to_cut:-to_cut]
        else:
            to_cut = int((diff) / 2)
            available_positions = all_positions[to_cut:-to_cut]
    else:
        available_positions = all_positions

    return available_positions

#######################
# single QMLA instance tree
#######################


def qmdclassTOnxobj(
    qmd,
    modlist=None,
    directed=True,
    only_adjacent_branches=True
):

    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    positions = {}
    branch_x_filled = {}
    branch_mod_count = {}

    max_branch_id = qmd.branch_highest_id
    max_mod_id = qmd.highest_model_id
    if modlist is None:
        modlist = range(max_mod_id)
    for i in range(max_branch_id + 1):
        branch_x_filled[i] = 0
        branch_mod_count[i] = 0

    for i in modlist:
        mod = qmd.get_model_storage_instance_by_id(i)
        name = mod.model_name
        branch = qmd.get_model_data_by_field(name=name, field='branch_id')
        branch_mod_count[branch] += 1
        latex_term = mod.model_name_latex

        G.add_node(i)
        G.node[i]['label'] = latex_term
        G.node[i]['status'] = 0.2
        G.node[i]['info'] = 'Non-winner'

    # Set x-coordinate for each node based on how many nodes
    # are on that branch (y-coordinate)
    most_models_per_branch = max(branch_mod_count.values())
    for i in modlist:
        mod = qmd.get_model_storage_instance_by_id(i)
        name = mod.model_name
        branch = qmd.get_model_data_by_field(name=name, field='branch_id')
        num_models_this_branch = branch_mod_count[branch]
        pos_list = available_position_list(
            num_models_this_branch,
            most_models_per_branch
        )
        branch_filled_so_far = branch_x_filled[branch]
        branch_x_filled[branch] += 1

        x_pos = pos_list[branch_filled_so_far]
        y_pos = branch
        positions[i] = (x_pos, y_pos)
        G.node[i]['pos'] = (x_pos, y_pos)

    # set node colour based on whether that model won a branch
    for b in list(qmd.branch_champions.values()):
        if b in modlist:
            G.node[b]['status'] = 0.45
            G.node[b]['info'] = 'Branch Champion'

    G.node[qmd.champion_model_id]['status'] = 0.9
    G.node[qmd.champion_model_id]['info'] = 'Overall Champion'

    edges = []
    for a in modlist:
        for b in modlist:
            is_adj = adjacent_branch_test(qmd, a, b)
            if is_adj or not only_adjacent_branches:
                if a != b:
                    unique_pair = database_framework.unique_model_pair_identifier(a, b)
                    if ((unique_pair not in edges)
                        and (unique_pair in qmd.bayes_factor_pair_computed)
                        ):
                        edges.append(unique_pair)
                        vs = [int(stringa) for stringa
                              in unique_pair.split(',')
                              ]

                        thisweight = np.log10(
                            qmd.all_bayes_factors[float(vs[0])][float(vs[1])][-1]
                        )

                        if thisweight < 0:
                            # flip negative valued edges and move
                            # them to positive
                            thisweight = - thisweight
                            flipped = True
                            G.add_edge(vs[1], vs[0],
                                       weight=thisweight, flipped=flipped,
                                       winner=b,
                                       loser=a,
                                       adj=is_adj
                                       )
                        else:
                            flipped = False
                            G.add_edge(vs[0], vs[1],
                                       weight=thisweight, flipped=flipped,
                                       winner=a,
                                       loser=b,
                                       adj=is_adj
                                       )
    return G


def plot_qmla_single_instance_tree(
    qmd,
    save_to_file=None,
    only_adjacent_branches=True,
    id_labels=True,
    modlist=None
):

    G = qmdclassTOnxobj(
        qmd,
        only_adjacent_branches=only_adjacent_branches,
        modlist=modlist)

    arr = np.linspace(0, 50, 100).reshape((10, 10))
    cmap = plt.get_cmap('viridis')
    new_cmap = truncate_colormap(cmap, 0.35, 1.0)

    plotTreeDiagram(
        G,
        n_cmap=plt.cm.pink_r,
        e_cmap=new_cmap,
        arrow_size=0.02,
        # arrow_size = 8.0,
        nonadj_alpha=0.1, e_alphas=[],
        label_padding=0.4, pathstyle="curve",
        id_labels=id_labels, save_to_file=save_to_file)
