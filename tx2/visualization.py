"""Helper functions for constructing visualizations."""

from IPython.display import display, clear_output
import itertools
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud, STOPWORDS
from typing import Dict, List

from tx2 import calc, utils
import tx2.wrapper


# if foreground_color is None, it will automatically decide white or black by color
def get_nice_html_label(text: str, color: str, foreground_color: str = None) -> str:
    """Get a nice looking colored background label.

    :param text: The text to display in the label.
    :param color: The background color as a hex string ('#XXXXXX') for the label
    :param foreground_color: Leave as None for an automatic black/white foreground color
        determination from :meth:`contrasting_text_color`.
    :return: An HTML string for the label.
    """
    # determine if foreground should be white or black
    if foreground_color is None:
        foreground_color = contrasting_text_color(color)

    label = f"<span style='background-color: {color}; color: {foreground_color}; padding: 3px; border-radius: 5px;'>{text}</span>"
    return label


# https://stackoverflow.com/questions/1855884/determine-font-color-based-on-background-color
def contrasting_text_color(hex_str: str) -> str:
    """Get a contrasting foreground text color for specified background hex color

    :param hext_str: A hex string color ('#XXXXXX') for which to determine a black-or-white
        foreground color.
    :return: '#FFF' or '#000'.
    """
    r, g, b = (hex_str[1:3], hex_str[3:5], hex_str[5:])

    luminance = (int(r, 16) * 0.299 + int(g, 16) * 0.587 + int(b, 16) * 0.114) / 255

    if luminance > 0.5:
        return "#000"
    else:
        return "#FFF"


def render_html_text(text, transformer_wrapper: tx2.wrapper.Wrapper) -> str:
    """Get a text-salience highlighted HTML paragraph.

    :param text: The text to run salience on and render.
    :param transformer_wrapper: The :class:`tx2.wrapper.Wrapper` instance.
    :return: An HTML string with span-styled-highlights on each relevant word.
    """
    deltas = calc.salience_map(
        transformer_wrapper.soft_classify, text[:512], transformer_wrapper.encodings
    )
    normalised_d = calc.normalize_salience_map(deltas)

    html = "<p>"
    for word in text.split(" "):
        if word in normalised_d:
            html += f"<span style='background-color: rgba(255, 100, 0.0, {normalised_d[word]});'>{word}</span> "
        else:
            html += word + " "
    html += "<p>"
    return html


_cached_wordclouds = {}


def prepare_wordclouds(
    clusters: Dict[str, List[int]], test_df: pd.DataFrame, input_col_name: str
):
    """Pre-render the wordcloud for each cluster, this makes switching the main wordcloud figure faster.

    :param clusters: Dictionary of clusters where the values are the lists of dataframe indices for the entries in each cluster.
    :param test_df: The dataframe to draw the indices from.
    :param input_col_name: The name of the column containing the text.
    """
    for cluster in clusters:
        _cached_wordclouds[cluster] = gen_wordcloud(
            clusters[cluster], test_df, input_col_name
        )


def gen_wordcloud(indices: List[int], df: pd.DataFrame, input_col_name: str):
    """Creates and returns a wordcloud image that can be rendered with :code:`plt.imshow`.

    :param indices: The list of indices in the dataframe to draw text from to create the wordcloud.
    :param df: The dataframe to draw the indices from.
    :param input_col_name: The name of the column containing the text.
    """
    stopwords = set(STOPWORDS)
    stopwords.update(["via", "this"])
    text = " ".join([text for text in df[input_col_name].iloc[indices]])
    cloud = WordCloud(
        stopwords=set(STOPWORDS), background_color="white", width=800, height=400
    ).generate(text)
    return cloud


def plot_big_wordcloud(index: int, clusters: Dict[str, List[int]]):
    """Render the word cloud that the currently selected point is in.

    :param index: The index of the point to find the cluster of.
    :param clusters: The dictionary of clusters where the values are the lists of indices of entries in that cluster.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    cluster = utils.which_cluster(index, clusters)
    ax.imshow(_cached_wordclouds[cluster], interpolation="bilinear")
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.set_title(str(cluster))
    fig.tight_layout()
    return fig


def plot_passed_wordcloud(cloud, name):
    """Render the given word cloud.

    :param cloud: The word cloud to render.
    :param name: The title to render with the word cloud.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(cloud, interpolation="bilinear")
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.set_title(name)
    fig.tight_layout()
    return fig


def plot_wordclouds(dashboard):
    """Render the grid of all wordclouds.

    :param dashboard: The current dashboard, needed in order to grab the cluster data.
    """
    num_cols = 4
    num_rows = math.ceil(len(dashboard.transformer_wrapper.clusters) / num_cols)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(8, num_rows * 1.5))

    for index, cluster in enumerate(dashboard.transformer_wrapper.clusters):
        ax_x = int(index / num_cols)
        ax_y = index % num_cols
        axs[ax_x][ax_y].imshow(_cached_wordclouds[cluster], interpolation="bilinear")
        axs[ax_x][ax_y].xaxis.set_ticks([])
        axs[ax_x][ax_y].yaxis.set_ticks([])
        axs[ax_x][ax_y].xaxis.set_ticklabels([])
        axs[ax_x][ax_y].yaxis.set_ticklabels([])
        axs[ax_x][ax_y].set_title(str(cluster))
        axs[ax_x][ax_y].set_axis_off()

    if len(dashboard.transformer_wrapper.clusters) % num_cols != 0:
        for i in range(
            len(dashboard.transformer_wrapper.clusters) % num_cols, num_cols
        ):
            axs[-1][i].set_axis_off()

    fig.tight_layout()

    return fig


def plot_metrics(pred_y: List[int], target_y: List[int], encodings: Dict[str, int]):
    """Get colored dataframes with macro and micro scores for the given predictions on an aggregate level and per class.

    :param pred_y: Predicted labels.
    :param target_y: Actual labels.
    :param encodings: Dictionary of string label -> numeric label.
    :return: The per-class metrics dataframe and the aggregate metrics dataframe.
    """
    temp_dict = {"pred": pred_y, "target": target_y}
    temp_df = pd.DataFrame.from_dict(temp_dict)
    per_class, macros, micros = calc.prediction_scores(
        temp_df, "target", "pred", encodings
    )

    per_df_rows = []
    for metric in "precision", "recall", "f1":
        row = {"metric": metric}
        for key in per_class:
            row[utils.get_cat_by_index(key, encodings)] = per_class[key][metric]

        per_df_rows.append(row)

    per_df = pd.DataFrame(per_df_rows).style.background_gradient(
        cmap="RdYlGn", vmax=1.0, vmin=0.0
    )

    aggregate_rows = []
    for metric in "precision", "recall", "f1":
        aggregate_rows.append(
            {
                "metric": metric,
                "macro": macros[metric],
                "micro": micros[metric],
            }
        )

    agg_df = pd.DataFrame(aggregate_rows).style.background_gradient(
        cmap="RdYlGn", vmax=1.0, vmin=0.0
    )

    return per_df, agg_df


def plot_confusion_matrix(
    pred_y: List[int], target_y: List[int], encodings: Dict[str, int], figsize=(8, 8)
):
    """Get the confusion matrix for given predictions.

    :param pred_y: Predicted labels.
    :param target_y: Actual labels.
    :param encodings: Dictionary of string label -> numeric label.
    :param figsize: the size with which to
    """
    labels = []
    encoded = []
    for i in range(len(encodings.keys())):
        for key, value in encodings.items():
            if value == i:
                labels.append(key)
                encoded.append(value)
                break
    cm = confusion_matrix(target_y, pred_y, labels=list(encodings.values()))

    acc = np.trace(cm) / float(np.sum(cm))
    miss = 1 - acc

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(cm, interpolation="nearest", cmap="Blues")

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, rotation=90)
    #     for tick in ax.get_xticklabels():
    #         tick.set_rotation(45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels)
    ax.set_xlabel(
        "Predicted label\naccuracy={:0.4f}; misclassified={:0.4f}".format(acc, miss)
    )
    ax.set_ylabel("True label")
    ax.grid(False)

    text_fg_threshold = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(
            j,
            i,
            "{:,}".format(cm[i, j]),
            horizontalalignment="center",
            color="white" if cm[i, j] > text_fg_threshold else "black",
        )

    return fig


def _get_scatter_points_from_embeddings(
    dashboard, embeddings: List[List[int]], df: pd.DataFrame, label_col_name: str
):
    """DOES NOT DISPLAY GRAPH, just a helper for splitting out the UMAP embeddings."""

    colors = []
    for index, row in df.iterrows():
        if label_col_name is None:
            colors.append(dashboard.colors[0])
        else:
            colors.append(dashboard.colors[row[label_col_name]])

    x = []
    y = []
    for result in embeddings:
        x.append(result[0])
        y.append(result[1])

    return np.array(x), np.array(y), np.array(colors)


@utils.debounce(1.0)
def plot_embedding_projections(text, dashboard, prediction=None):
    dashboard.html_graph_status.value = (
        "<p>" + tx2.visualization.get_nice_html_label("Graphing...", "#FF0000") + "</p>"
    )
    fig, ax = plt.subplots(figsize=(10, 8))

    # render all test data projections
    if not dashboard.chk_focus_errors.value:
        testing_x, testing_y, testing_c = _get_scatter_points_from_embeddings(
            dashboard,
            dashboard.transformer_wrapper.projections_testing,
            dashboard.transformer_wrapper.test_df,
            dashboard.transformer_wrapper.target_col_name,
        )
        ax.scatter(x=testing_x, y=testing_y, c=testing_c, alpha=0.8, s=dashboard.point_size)
    else:
        # render incorrect test data
        df_incorrect = dashboard.transformer_wrapper.test_df[
            dashboard.transformer_wrapper.test_df[
                dashboard.transformer_wrapper.target_col_name
            ]
            != dashboard.transformer_wrapper.test_df.predicted_classes
        ]
        incorrect_indices = list(df_incorrect.index)
        incorrect_projections = [
            dashboard.transformer_wrapper.projections_testing[i]
            for i in range(len(dashboard.transformer_wrapper.projections_testing))
            if i in incorrect_indices
        ]
        incorrect_x, incorrect_y, incorrect_c = _get_scatter_points_from_embeddings(
            dashboard,
            incorrect_projections,
            df_incorrect,
            dashboard.transformer_wrapper.target_col_name,
        )
        ax.scatter(x=incorrect_x, y=incorrect_y, c=incorrect_c, alpha=0.8, s=dashboard.point_size)
        testing_x, testing_y, testing_c = incorrect_x, incorrect_y, incorrect_c

        # render correct test data
        df_correct = dashboard.transformer_wrapper.test_df[
            dashboard.transformer_wrapper.test_df[
                dashboard.transformer_wrapper.target_col_name
            ]
            == dashboard.transformer_wrapper.test_df.predicted_classes
        ]
        correct_indices = list(df_correct.index)
        correct_projections = [
            dashboard.transformer_wrapper.projections_testing[i]
            for i in range(len(dashboard.transformer_wrapper.projections_testing))
            if i in correct_indices
        ]
        correct_x, correct_y, correct_c = _get_scatter_points_from_embeddings(
            dashboard,
            correct_projections,
            df_correct,
            dashboard.transformer_wrapper.target_col_name,
        )
        ax.scatter(x=correct_x, y=correct_y, c=correct_c, alpha=0.1, s=dashboard.unfocused_point_size)

    # render training data projections
    if dashboard.chk_show_train.value:
        training_x, training_y, training_c = _get_scatter_points_from_embeddings(
            dashboard,
            dashboard.transformer_wrapper.projections_training,
            dashboard.transformer_wrapper.train_df,
            dashboard.transformer_wrapper.target_col_name,
        )
        ax.scatter(x=training_x, y=training_y, c=training_c, alpha=0.1, s=dashboard.unfocused_point_size)

    # render highlighted data points
    if len(dashboard.highlight_indices) > 0:
        ax.scatter(
            x=testing_x[dashboard.highlight_indices],
            y=testing_y[dashboard.highlight_indices],
            c=testing_c[dashboard.highlight_indices],
            s=dashboard.highlighted_point_size,
            edgecolors="red",
            linewidth=2,
        )

    # if text differs from the selected index, render new point
    if (
        dashboard.prior_reference_point is None
        or dashboard.prior_reference_text != text
    ):
        text_projection = dashboard.transformer_wrapper.project([text])[0]
        if prediction is None:
            classification = dashboard.transformer_wrapper.classify([text])[0]
            pred_color = dashboard.colors[classification]
        else:
            pred_color = dashboard.colors[prediction]
        ax.scatter(
            x=text_projection[0],
            y=text_projection[1],
            s=dashboard.highlighted_point_size,
            c=pred_color,
            edgecolors="black",
            linewidth=2,
        )

    # render the original reference point and arrow if applicable
    if dashboard.prior_reference_point is not None:
        ax.scatter(
            x=dashboard.prior_reference_point[0][0],
            y=dashboard.prior_reference_point[0][1],
            s=dashboard.highlighted_point_size,
            c=dashboard.prior_reference_point[1],
            edgecolors="black",
            linewidth=2,
        )

        # arrow!
        if dashboard.prior_reference_text != text:
            x_dist = text_projection[0] - dashboard.prior_reference_point[0][0]
            y_dist = text_projection[1] - dashboard.prior_reference_point[0][1]
            #             if abs(x_dist) + abs(y_dist) > .5:
            width = 0.15
            head_width = 0.65
            ax.arrow(
                dashboard.prior_reference_point[0][0],
                dashboard.prior_reference_point[0][1],
                x_dist,
                y_dist,
                width=width,
                length_includes_head=True,
                head_width=head_width,
                facecolor="black",
                edgecolor="white",
            )

    # show visual cluster labels
    if dashboard.chk_show_cluster_labels.value:
        for x, y, label in dashboard.transformer_wrapper.cluster_labels:
            ax.text(
                float(x),
                float(y),
                label,
                bbox={"facecolor": "white", "alpha": 0.5, "pad": 2},
            )

    # display legend
    legend_elements = []
    for i in range(0, len(dashboard.transformer_wrapper.encodings)):
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=list(dashboard.transformer_wrapper.encodings.keys())[i],
                markerfacecolor=dashboard.colors[i],
                markersize=10,
            )
        )
    ax.legend(handles=legend_elements)

    # output
    fig.tight_layout()
    with dashboard.out_projection_scatter:
        clear_output(wait=True)
        dashboard.current_figures["umap"] = fig
        display(fig)
    dashboard.html_graph_status.value = (
        "<p>" + tx2.visualization.get_nice_html_label("Ready!", "#008000") + "</p>"
    )


def plot_clusters(clusters, cluster_values):
    """Plot highest word values for each cluster."""
    num_cols = 4

    num_rows = math.ceil(len(clusters) / num_cols)

    fig, axs = plt.subplots(
        num_rows, num_cols, figsize=(10, num_rows * 2), squeeze=False
    )

    for index, cluster in enumerate(clusters):
        ax_x = int(index / num_cols)
        ax_y = index % num_cols

        ax = axs[ax_x][ax_y]
        freq = cluster_values[cluster][:10]
        y = [item[1] for item in freq]
        y_labels = [item[0][:20] for item in freq]
        y_pos = np.arange(len(y_labels))
        y_pos = y_pos * -1

        ax.barh(y_pos, y)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels)
        axs[ax_x][ax_y].set_title(str(cluster))

    fig.tight_layout()
    return fig


def plot_clusters_stacked(clusters, cluster_words_classified, encodings, colors):
    """Plot highest word values for each cluster, colored according to entry classification"""
    num_cols = 4

    num_rows = math.ceil(len(clusters) / num_cols)

    fig, axs = plt.subplots(
        num_rows, num_cols, figsize=(10, num_rows * 2), squeeze=False
    )

    for index, cluster in enumerate(clusters):
        ax_x = int(index / num_cols)
        ax_y = index % num_cols

        ax = axs[ax_x][ax_y]
        #         print(cluster_words_classified)
        #         print(cluster_words_classified.keys())
        #         print(cluster)
        words = list(cluster_words_classified[cluster].keys())[:10]
        y_labels = words
        y_pos = np.arange(len(y_labels))

        starts = {}  # starting x pos for each word

        for word in words:
            starts[word] = 0

        for classification in encodings.values():

            y = []
            widths = []
            left = []

            freqs = {}
            for index, word in enumerate(words):
                freqs[word] = cluster_words_classified[cluster][word]

                y.append(index)
                left.append(starts[word])
                classification = str(classification)
                width = int(freqs[word][classification])
                widths.append(width)
                starts[word] += width

            ax.barh(y, widths, left=left, color=colors[int(classification)])

        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels)
        axs[ax_x][ax_y].set_title(str(cluster))

    fig.tight_layout()
    return fig
