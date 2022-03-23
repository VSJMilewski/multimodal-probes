import argparse
import logging
import os
from typing import (
    Callable,
    Dict,
    Final,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import matplotlib.patches as patches  # type: ignore
import numpy as np
from pytorch_lightning.callbacks import ProgressBar
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    ProgressType,
    StyleType,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

logger = logging.getLogger(__name__)

# DISTRIBUTIONS COPIED FROM Hewitt and Manning's Structural Probes project
PTB_TRAIN_EMPIRICAL_POS_DISTRIBUTION: Final = [
    0.00003789361998,
    0.00006105083219,
    0.0001021022538,
    0.0001494692788,
    0.0001768368932,
    0.0002463085299,
    0.0003894622053,
    0.0004747228503,
    0.0009083942789,
    0.001437852358,
    0.001448378364,
    0.001860997781,
    0.00204941328,
    0.002255722989,
    0.002487295111,
    0.002802022677,
    0.002813601283,
    0.003408320597,
    0.004519866783,
    0.005023009848,
    0.00728294324,
    0.007465043136,
    0.007759771291,
    0.008849212865,
    0.009158677428,
    0.01031864324,
    0.01314803353,
    0.01562690784,
    0.01835314328,
    0.02107727351,
    0.02281195923,
    0.02353299061,
    0.02520662549,
    0.02782865347,
    0.03146117799,
    0.03259903919,
    0.03849149709,
    0.04155456471,
    0.05129006724,
    0.06300445882,
    0.06443704817,
    0.08614693462,
    0.09627716236,
    0.1037379951,
    0.1399274548,
]
PTB_DEV_EMPIRICAL_DEPTH_DICT: Final = {
    14: 0.00009970835307,
    13: 0.000373906324,
    12: 0.0007228855597,
    11: 0.001395916943,
    10: 0.003938479946,
    9: 0.007702470274,
    8: 0.01570406561,
    7: 0.02921454745,
    0: 0.04237605005,
    6: 0.05309469801,
    5: 0.08729466311,
    4: 0.1302440362,
    3: 0.183563078,
    1: 0.2192088142,
    2: 0.22506668,
}
PTB_DEV_EMPIRICAL_DEPTH_keys: Final = list(sorted(PTB_DEV_EMPIRICAL_DEPTH_DICT.keys()))
PTB_DEV_EMPIRICAL_DEPTH_DISTRIBUTION: Final = [
    PTB_DEV_EMPIRICAL_DEPTH_DICT[x] for x in PTB_DEV_EMPIRICAL_DEPTH_keys
]
PTB_TRAIN_EMPIRICAL_DEP_SEQ_LEN_dict: Final = {
    -44: 7.690651244347372e-06,
    -3: 0.047819370772888475,
    -2: 0.1088534777124927,
    -1: 0.277384211752194,
    4: 0.035580248649741374,
    1: 0.17205854563192982,
    3: 0.06036172428795556,
    2: 0.09961151224571411,
    -4: 0.02238199244997781,
    15: 0.003433326448369362,
    6: 0.01574166443271559,
    7: 0.011697480542652352,
    8: 0.009206808203947281,
    11: 0.00579765237377444,
    -13: 0.0016556873464616411,
    -11: 0.002414864490725075,
    5: 0.022290803299509117,
    -8: 0.004191404928169318,
    19: 0.0021665663219790025,
    -7: 0.005423007791728375,
    -5: 0.012027079881695811,
    9: 0.00793565341970301,
    22: 0.0015447222356503435,
    -10: 0.0029543087422928688,
    -19: 0.0007163292301877837,
    -6: 0.00748410232521347,
    12: 0.004976950019556227,
    35: 0.0003317966679704152,
    13: 0.004389164531595393,
    18: 0.002396187194845945,
    -9: 0.0034783716913719684,
    28: 0.0008723395840016876,
    43: 0.00011865576205564516,
    -17: 0.0009151874980773372,
    -12: 0.0020545025467042263,
    26: 0.0009964886683747236,
    25: 0.0011404137130903674,
    -23: 0.0003471779704591099,
    -26: 0.00023731152411129032,
    20: 0.001866630923449455,
    34: 0.00038343389775389035,
    10: 0.006666695964385693,
    36: 0.0002955407406756347,
    -22: 0.00042518314736606183,
    -15: 0.0012920294090503583,
    -21: 0.0005306549358599686,
    16: 0.0030652738531041666,
    17: 0.0026005387850528898,
    -16: 0.001105256450259065,
    14: 0.003947501417277158,
    23: 0.001423869144667742,
    -20: 0.0005767988433260529,
    21: 0.0017677511217364173,
    32: 0.00048780702178431896,
    38: 0.0002647781356982452,
    37: 0.0002450021753556377,
    50: 4.834123639304062e-05,
    46: 6.042654549130078e-05,
    31: 0.0005910814813512694,
    -14: 0.0015601035381390383,
    27: 0.0009470487675182048,
    45: 0.00010107713063999403,
    24: 0.0012953254024407929,
    42: 0.00013623439347129629,
    29: 0.000745993170701695,
    40: 0.00020654891913390083,
    41: 0.00013953038686173087,
    47: 5.49332231739098e-05,
    30: 0.0006273374086460499,
    -18: 0.0008174063608277777,
    56: 1.7578631415651135e-05,
    -35: 4.1749249612171444e-05,
    -27: 0.0001658983339852076,
    39: 0.00019885826788955345,
    33: 0.0004647350680512769,
    -31: 8.789315707825567e-05,
    57: 2.1973289269563917e-05,
    61: 1.867729587912933e-05,
    -30: 0.00011975442651912336,
    44: 8.239983476086469e-05,
    -24: 0.00028455409604085275,
    -29: 0.000106570452957385,
    -25: 0.0002614821423078106,
    65: 8.789315707825568e-06,
    49: 4.834123639304062e-05,
    51: 3.186126944086768e-05,
    62: 1.0986644634781959e-05,
    90: 1.098664463478196e-06,
    -36: 3.405859836782407e-05,
    -28: 0.00013953038686173087,
    -38: 2.1973289269563917e-05,
    -33: 6.921586119912634e-05,
    52: 2.3071953733042113e-05,
    55: 1.867729587912933e-05,
    72: 4.394657853912784e-06,
    73: 3.295993390434588e-06,
    77: 2.197328926956392e-06,
    85: 1.098664463478196e-06,
    48: 5.603188763738799e-05,
    68: 5.493322317390979e-06,
    -32: 6.482120334521356e-05,
    -40: 1.4282638025216547e-05,
    53: 2.417061819652031e-05,
    54: 2.5269282659998507e-05,
    100: 1.098664463478196e-06,
    -34: 6.372253888173536e-05,
    -39: 2.3071953733042113e-05,
    -48: 3.295993390434588e-06,
    -37: 2.3071953733042113e-05,
    -67: 1.098664463478196e-06,
    -64: 2.197328926956392e-06,
    -63: 1.098664463478196e-06,
    -59: 1.098664463478196e-06,
    -41: 9.887980171303763e-06,
    58: 1.2085309098260154e-05,
    -47: 3.295993390434588e-06,
    59: 9.887980171303763e-06,
    60: 9.887980171303763e-06,
    63: 1.0986644634781959e-05,
    67: 3.295993390434588e-06,
    79: 3.295993390434588e-06,
    64: 6.591986780869176e-06,
    69: 2.197328926956392e-06,
    -43: 5.493322317390979e-06,
    80: 1.098664463478196e-06,
    81: 1.098664463478196e-06,
    -58: 1.098664463478196e-06,
    -56: 1.098664463478196e-06,
    -42: 5.493322317390979e-06,
    -49: 1.098664463478196e-06,
    74: 4.394657853912784e-06,
    75: 3.295993390434588e-06,
    117: 1.098664463478196e-06,
    -62: 1.098664463478196e-06,
    76: 1.098664463478196e-06,
    78: 2.197328926956392e-06,
    -53: 2.197328926956392e-06,
    -65: 1.098664463478196e-06,
    -61: 1.098664463478196e-06,
    127: 1.098664463478196e-06,
    -45: 4.394657853912784e-06,
    -46: 1.098664463478196e-06,
    -50: 1.098664463478196e-06,
    -77: 1.098664463478196e-06,
    -74: 1.098664463478196e-06,
    70: 2.197328926956392e-06,
    66: 1.098664463478196e-06,
    -55: 1.098664463478196e-06,
    -54: 2.197328926956392e-06,
    -66: 1.098664463478196e-06,
    71: 2.197328926956392e-06,
    83: 1.098664463478196e-06,
    87: 1.098664463478196e-06,
    86: 1.098664463478196e-06,
}
PTB_TRAIN_EMPIRICAL_DEP_SEQ_LEN_dists: Final = list(
    sorted(PTB_TRAIN_EMPIRICAL_DEP_SEQ_LEN_dict.keys())
)
PTB_TRAIN_EMPIRICAL_DEP_SEQ_LEN_probs: Final = [
    PTB_TRAIN_EMPIRICAL_DEP_SEQ_LEN_dict[x]
    for x in PTB_TRAIN_EMPIRICAL_DEP_SEQ_LEN_dists
]


class Observation(NamedTuple):
    """Observation of a single sentence, with all info from the conllx File"""

    index: Tuple[str]  # type: ignore
    sentence: Tuple[str]
    lemma_sentence: Tuple[str]
    upos_sentence: Tuple[str]
    xpos_sentence: Tuple[str]
    morph: Tuple[str]
    head_indices: Tuple[str]
    governance_relations: Tuple[str]
    secondary_relations: Tuple[str]
    extra_info: Tuple[str]
    pp_entity_id: Tuple[str] = ("",)
    pp_gold_governance: Tuple[str] = ("",)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class UnionFind:
    """
    Naive UnionFind implementation for (slow) Prim's MST algorithm
    Used to compute minimum spanning trees for distance matrices
    """

    def __init__(self, n: int):
        """
        initialize the list, with for each item in the sequence its parent.
        Args:
            n: The number of parents, equal to the length of the sequence
        """
        self.parents = list(range(n))

    def union(self, i: int, j: int) -> None:
        """
        Update the parent of i, to be j
        Args:
            i: the index of the item to update
            j: the index of the new parent for item i
        """
        if self.find(i) != self.find(
            j
        ):  # make sure both indices don't have the same parent
            i_parent = self.find(i)
            self.parents[i_parent] = j

    def find(self, i: int):
        """
        find the direct parent connected to i
        Args:
            i: the index of the item for which to find the parent
        Returns: the id of the parent of item i
        """
        # initialize the parent of i as itself
        i_parent = i
        while True:
            # loop over the stored parent ids sequence,
            # until we find one that is actually set correct.
            if i_parent != self.parents[i_parent]:
                i_parent = self.parents[i_parent]
            else:
                break
        return i_parent


def dist_matrix_to_edges(matrix, poses=None):
    """
    Constructs a minimum spanning tree from the pairwise weights in matrix;
    returns the edges.

    Never lets punctuation-tagged words be part of the tree.
    """
    # map each tuple of indices to a distance
    pairs_to_distances = {}
    uf = UnionFind(len(matrix))
    for i_index, line in enumerate(matrix):
        for j_index, dist in enumerate(line):
            # Skip all the punctuations, not part of dependency tree
            if poses is not None and poses[i_index] in [
                "''",
                ",",
                ".",
                ":",
                "``",
                "-LRB-",
                "-RRB-",
            ]:
                continue
            if poses is not None and poses[j_index] in [
                "''",
                ",",
                ".",
                ":",
                "``",
                "-LRB-",
                "-RRB-",
            ]:
                continue
            pairs_to_distances[(i_index, j_index)] = dist
    edges = []
    # loop over the sorted distances, so we start at the root
    for (i_index, j_index), distance in sorted(
        pairs_to_distances.items(), key=lambda x: x[1]
    ):
        if uf.find(i_index) != uf.find(j_index):
            uf.union(i_index, j_index)
            edges.append((i_index, j_index))
    return edges


def find_child_edges(test_edge, edges):
    check_edges = edges.copy()
    child_edges = []
    test_id = [test_edge[1]]
    added_idxs = True
    while added_idxs:
        added_idxs = []
        new_test_ids = []
        for idx, e in enumerate(check_edges):
            if e[0] in test_id:
                child_edges.append(e)
                added_idxs.append(idx)
                new_test_ids.append(e[1])
        if added_idxs:
            check_edges = np.delete(check_edges, added_idxs, axis=0)
            test_id = new_test_ids


def sentence_to_graph(
    words: List[str],
    words_phraseids: Optional[List[str]] = None,
    tikz: bool = False,
    gold_edges: Optional[List[Tuple[int, int]]] = None,
    prediction_edges: Optional[List[Tuple[int, int]]] = None,
    entid2color: Dict[str, str] = None,
    depths=None,
    ax=None,
):
    """Turns edge sets on word (nodes) into tikz dependency LaTeX."""
    if entid2color is None:
        entid2color = {}
    # generate tikz string
    if tikz:
        assert (
            gold_edges is not None or prediction_edges is not None
        ), "atleast one of the list of edges must be filled"
        tikz_string = """\\begin{dependency}[hide label, edge unit distance=.5ex]\\begin{deptext}[column sep=0.05cm]"""
        tikz_string += (
            "\\& ".join([x.replace("$", "\$").replace("&", "+") for x in words])
            + " \\\\"
            + "\n"
        )
        tikz_string += "\\end{deptext}" + "\n"
        if gold_edges is not None:
            for i_index, j_index in gold_edges:
                tikz_string += "\\depedge{{{}}}{{{}}}{{{}}}\n".format(
                    i_index + 1, j_index + 1, "."
                )
        if prediction_edges is not None:
            for i_index, j_index in prediction_edges:
                tikz_string += "\\depedge[edge style={{red!60!}}, edge below]{{{}}}{{{}}}{{{}}}\n".format(
                    i_index + 1, j_index + 1, "."
                )
        tikz_string += "\\end{dependency}\n"
        return tikz_string
    else:
        assert (
            words_phraseids is not None
        ), "when not doing tikz, the 'words_phraseids' should be set"
        assert (
            gold_edges is not None and ax is not None
        ), "the networkx method is only implemented for gold standard edges"
        assert (
            depths is not None
        ), "expecting to receive the depths, when not creating the tikz figure"
        ax.axis("off")
        stepsize = 4
        points = np.array([[i, 0] for i in range(0, len(words) * stepsize, stepsize)])
        # edges = np.array(gold_edges)
        edges = gold_edges.copy()
        x = points[:, 0].flatten()
        y = points[:, 1].flatten()
        ax.scatter(x, y, marker="o")
        ax.set_ylim([-15, 9])
        phrid2id_: Dict[str, int] = {
            phr: i for i, phr in enumerate(set(words_phraseids))
        }
        phrid2id_["_"] = -100
        entid2color["_"] = "black"

        # FIX ALL THE AVAILABLE TEXT PRINTING
        for i, point in enumerate(points):
            if words_phraseids[i] in entid2color:
                col = entid2color[words_phraseids[i]]
            else:
                col = "black"
            offset = 1.5
            ax.text(
                point[0],
                point[1] - offset,
                int(depths[i]),
                color="black",
                fontweight="bold",
                rotation=0,
                verticalalignment="top",
                horizontalalignment="center",
            )
            offset = 6
            ax.text(
                point[0],
                point[1] - offset,
                phrid2id_[words_phraseids[i]],
                color=col,
                rotation=0,
                verticalalignment="top",
                horizontalalignment="center",
            )
            offset = 8
            ax.text(
                point[0],
                point[1] - offset,
                words[i],
                color=col,
                rotation=45,
                verticalalignment="top",
                horizontalalignment="right",
            )

        def add_edges_arcs(
            edges: List[Tuple[int, int]],
            color="black",
            linewidth=1,
            linestyle=None,
        ) -> None:
            for edge in edges:
                max_e = points[np.max(edge), 0]
                min_e = points[np.min(edge), 0]
                diff = max_e - min_e
                arc = patches.Arc(
                    (min_e + diff / 2, 0),
                    width=diff,
                    height=(diff) / 2,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    theta2=180.0,
                )
                ax.add_patch(arc)

        # DRAW ALL THE EDGES
        add_edges_arcs(edges)
        visual_edges, phrase2depth, phrase_head_idx = _text_edges2visual_edges(
            edges, depths, words_phraseids
        )
        add_edges_arcs(visual_edges, color="red", linewidth=2, linestyle=(0, (1, 2)))
        for phrase_id, idx in phrase_head_idx.items():
            offset = 3.5
            ax.text(
                points[idx][0],
                points[idx][1] - offset,
                int(phrase2depth[phrase_id]),
                color="red",
                fontweight="bold",
                rotation=0,
                verticalalignment="top",
                horizontalalignment="center",
            )

        return ax


def print_tikz(self, prediction_edges, gold_edges, words, split_name):
    """Turns edge sets on word (nodes) into tikz dependency LaTeX."""
    with open(os.path.join(self.reporting_root, split_name + ".tikz"), "a") as fout:
        string = """\\begin{dependency}[hide label, edge unit distance=.5ex]\\begin{deptext}[column sep=0.05cm]"""
        string += (
            "\\& ".join([x.replace("$", "\$").replace("&", "+") for x in words])
            + " \\\\"
            + "\n"
        )
        string += "\\end{deptext}" + "\n"
        for i_index, j_index in gold_edges:
            string += "\\depedge{{{}}}{{{}}}{{{}}}\n".format(
                i_index + 1, j_index + 1, "."
            )
        for i_index, j_index in prediction_edges:
            string += "\\depedge[edge style={{red!60!}}, edge below]{{{}}}{{{}}}{{{}}}\n".format(
                i_index + 1, j_index + 1, "."
            )
        string += "\\end{dependency}\n"
        fout.write("\n\n")
        fout.write(string)


def get_nopunct_argmin(prediction, words, poses):
    """
    Gets the argmin of predictions, but filters out all punctuation-POS-tagged words
    """
    puncts = ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]
    original_argmin = np.argmin(prediction)
    for i in range(len(words)):
        argmin = np.argmin(prediction)
        if poses[argmin] not in puncts:
            return argmin
        else:
            prediction[argmin] = 9000
    return original_argmin


class MyProgress(ProgressBar):
    def __init__(self, disable_val_bar: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.disable_val_bar = disable_val_bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if self.disable_val_bar:
            bar.disable = True
        return bar


def get_all_subclasses(cls):
    all_subclasses = []
    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))
    return all_subclasses


def check_volta_layer(config, task: str, layer_idx: int, mm_bert_layer: int):
    mm_vision_layer = "Visual" in task
    if layer_idx in ["text_baseline", "vision_baseline", "vision_mapped_baseline"]:
        pass
    elif config is not None:
        if mm_bert_layer:
            if mm_vision_layer:
                layer_idx = config.bert_layer2ff_sublayer[str(layer_idx)]
                if layer_idx not in config.v_ff_sublayers:
                    logger.warning(
                        "given mapped layer_idx {} not a vision-feedforward sublayers, "
                        "skipping this setting".format(layer_idx)
                    )
                    raise ValueError(
                        "given mapped layer_idx {} not a vision-feedforward sublayers, "
                        "skipping this setting".format(layer_idx)
                    )
        else:
            if mm_vision_layer:
                if layer_idx not in config.v_ff_sublayers:
                    logger.warning(
                        "given layer_idx {} not a vision-feedforward sublayers, "
                        "skipping this setting".format(layer_idx)
                    )
                    raise ValueError(
                        "given layer_idx {} not a vision-feedforward sublayers, "
                        "skipping this setting".format(layer_idx)
                    )
            else:
                if layer_idx not in config.t_ff_sublayers:
                    logger.warning(
                        "given layer_idx {} not a text-feedforward sublayers, "
                        "skipping this setting".format(layer_idx)
                    )
                    raise ValueError(
                        "given layer_idx {} not a text-feedforward sublayers, "
                        "skipping this setting".format(layer_idx)
                    )


progress_columns: List["ProgressColumn"] = [
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn(
        "[progress.percentage]{task.percentage:>3.0f}% "
        "[{task.completed}/{task.total}]"
    ),
    TimeElapsedColumn(),
    TextColumn("<"),
    TimeRemainingColumn(),
]


# AN EXACT COPy FROM THE RICH TRACK, BUT EXTENDED SO MORE PROGRESS INFORMATION IS PRINTED
def track(
    sequence: Union[Sequence[ProgressType], Iterable[ProgressType]],
    description: str = "Working...",
    total: Optional[float] = None,
    auto_refresh: bool = True,
    console: Optional[Console] = None,
    transient: bool = False,
    get_time: Optional[Callable[[], float]] = None,
    refresh_per_second: float = 10,
    style: StyleType = "bar.back",
    complete_style: StyleType = "bar.complete",
    finished_style: StyleType = "bar.finished",
    pulse_style: StyleType = "bar.pulse",
    update_period: float = 0.1,
    disable: bool = False,
) -> Iterable[ProgressType]:
    """Track progress by iterating over a sequence.
    Args:
        sequence (Iterable[ProgressType]): A sequence (must support "len")
                                           you wish to iterate over.
        description (str, optional): Description of task show next to progress bar.
                                     Defaults to "Working".
        total: (float, optional): Total number of steps. Default is len(sequence).
        auto_refresh (bool, optional): Automatic refresh, disable to force a refresh
                                       after each iteration. Default is True.
        transient: (bool, optional): Clear the progress on exit. Defaults to False.
        console (Console, optional): Console to write to.
                                     Default creates internal Console instance.
        refresh_per_second (float): Number of times per second to refresh
                                    the progress information. Defaults to 10.
        style (StyleType, optional): Style for the bar background.
                                     Defaults to "bar.back".
        complete_style (StyleType, optional): Style for the completed bar.
                                              Defaults to "bar.complete".
        finished_style (StyleType, optional): Style for a finished bar.
                                              Defaults to "bar.done".
        pulse_style (StyleType, optional): Style for pulsing bars.
                                           Defaults to "bar.pulse".
        update_period (float, optional): Minimum time (in seconds)
                                         between calls to update(). Defaults to 0.1.
        disable (bool, optional): Disable display of progress.
    Returns:
        Iterable[ProgressType]: An iterable of the values in the sequence.
    """

    columns: List["ProgressColumn"] = (
        [TextColumn("{task.description}")] if description else []
    )
    columns.extend(
        (
            BarColumn(
                style=style,
                complete_style=complete_style,
                finished_style=finished_style,
                pulse_style=pulse_style,
            ),
            TextColumn("{task.percentage:>3.0f}% [{task.completed}/{task.total}]"),
            TimeElapsedColumn(),
            TextColumn("<"),
            TimeRemainingColumn(),
        )
    )
    progress = Progress(
        *columns,
        auto_refresh=auto_refresh,
        console=console,
        transient=transient,
        get_time=get_time,
        refresh_per_second=refresh_per_second or 10,
        disable=disable,
    )

    with progress:
        yield from progress.track(
            sequence, total=total, description=description, update_period=update_period
        )


def _text_edges2visual_edges(
    text_edges: List[Tuple[int, int]],
    text_depths: Union[List[int], np.ndarray],
    words_phraseids: List[str],
) -> Tuple[List[Tuple[int, int]], Dict[str, int], Dict[str, int]]:
    """
    Args:
        text_edges: the edges of the dependency tree for the sentence in a np matrix.
                    Each item is an edge with [source, target]
        text_depths: The depths for each text token in the sentence
        words_phraseids: for each word in the sentence,
                         the id of the phrase it belongs to. '_' if none.
    Returns: the edges of the dependency tree projected on visual regions
    """
    visual_edges = []

    # For each phrase, check what the text idx is of each phrase
    phrase2txt_index: Dict[str, int] = {}
    # Also, based on the text idx we can map the phrase to an initial depth
    phrase2depth: Dict[str, int] = {}

    # loop over unique phrase ids in the sentence
    for phr_id in set(words_phraseids):
        if phr_id == "_":  # word does not belong to a phrase
            continue
        # Collections with for current phrase id all its head_idx and all its depths
        phrase_idxs: List[int] = []
        phrase_depths: List[int] = []
        # Find all words that belong to the same phrase
        for idx, id_ in enumerate(words_phraseids):
            if phr_id == id_:
                phrase_idxs.append(idx)
                phrase_depths.append(text_depths[idx])
        # if all depths are equal, we use the final token from the phrase as head index
        if all(phrase_depths[0] == x for x in phrase_depths):
            phr_min_d_idx = len(phrase_depths) - 1
        # not all depths are equal, we use the one with lowest depth
        else:
            phr_min_d_idx = int(np.argmin(phrase_depths))
        phrase2txt_index[phr_id] = phrase_idxs[phr_min_d_idx]
        phrase2depth[phr_id] = phrase_depths[phr_min_d_idx]
    # Find the root node for the text
    text_root_idx = np.where(text_depths == 0)[0][0]
    phrase2depth["0"] = 0  # the full image is always the root for visual tree
    if text_root_idx in phrase2txt_index.values():
        # There is a phrase that is the root, thus to have not two elements
        # map to same text index, we map the visual root to prepended token
        # in front of the sentence
        phrase2txt_index["0"] = -1
    else:
        # this is as expected. The text root is not a phrase
        # and we map the visual root to the text root
        phrase2txt_index["0"] = text_root_idx
    # reverse mapping from index to phrase_id
    idx2phr = {v: k for k, v in phrase2txt_index.items()}

    # loop over all the phrases based on there depth, lowest to highest
    for sort_phr_id in sorted(phrase2depth, key=phrase2depth.get):  # type: ignore
        sort_txt_idx = phrase2txt_index[sort_phr_id]
        # this is our own manual assigned root node. since it is the root,
        # it has no parent, so continue
        if sort_phr_id == "0":
            continue
        # the current node is the root node in the text tree
        elif sort_txt_idx == text_root_idx:
            # there is no parent for a root node. A phrase shouldn't be the root?
            # In the visual tree it should be the child of the full image root node
            new_edge = (phrase2txt_index["0"], sort_txt_idx)
            visual_edges.append(new_edge)
            phrase2depth[sort_phr_id] = phrase2depth["0"] + 1
        else:
            # loop over all textual edges to find and construct the correct
            # visual edge given the current phrase_id
            for edge in text_edges:
                # if the current phrase_idx is not the child node of the
                # current text edge, we don't need this edge.
                if sort_txt_idx != edge[1]:
                    continue
                # the current node is the child of the root node, or its
                # text parent node, is also a visual node
                elif edge[0] == text_root_idx or edge[0] in phrase2txt_index.values():
                    # we append the textual edge as a valid visual edge
                    visual_edges.append(edge)
                    # we update the depth to be equal to
                    # "the depth of the parent node plus 1"
                    phrase2depth[idx2phr[edge[1]]] = phrase2depth[idx2phr[edge[0]]] + 1
                # we found an edge where this node is the child, but the parent
                # is not a visual node we need to move up through the textual edges
                # from the current one, to find the visual parent.
                else:
                    # start chain by finding the direct parent edge from current edge
                    parent_edge_chain = [_find_parent_edge(edge, text_edges)]
                    if parent_edge_chain[0] is None:
                        # No head could be find, so we leave this phrase dangling
                        logger.debug(
                            "no head found. Left phrase dangling "
                            "MAYBE IF IT IS NOT A TREE, WE CAN FURTHER CHECK THIS?"
                        )
                        continue
                    while True:
                        dangling_skip = False
                        # the highest node in the current parent edge chain is
                        # a visual node, or it is the visual root
                        if (
                            parent_edge_chain[-1][0] == text_root_idx
                            or parent_edge_chain[-1][0] in phrase2txt_index.values()
                        ):
                            break
                        # no visual parent found, keep moving up
                        pe = _find_parent_edge(parent_edge_chain[-1], text_edges)
                        if pe is None:
                            # No head could be found higher, leave this phrase dangling
                            logger.debug(
                                "no head found higher. Left phrase dangling! "
                                "MAYBE IF IT IS NOT A TREE, WE CAN FURTHER CHECK THIS?"
                            )
                            dangling_skip = True
                            break
                        parent_edge_chain.append(pe)
                    if dangling_skip:
                        continue
                    # update the new depth, based on steps shifted up in the tree
                    # (length of parent edge chain)
                    parent_id = parent_edge_chain[-1][0]
                    # update depth as parent_depth + 1
                    phrase2depth[sort_phr_id] = phrase2depth[idx2phr[parent_id]] + 1
                    # add the edge from the parent to current node
                    new_edge = (parent_id, edge[1])
                    visual_edges.append(new_edge)
    assert (
        sum(value == 0 for value in phrase2depth.values()) == 1
    ), "ERROR: multiple roots in visual tree."
    return visual_edges, phrase2depth, phrase2txt_index


def _find_parent_edge(test_edge, edges):
    for e in edges:
        if e[1] == test_edge[0]:
            return e
    return None
