"""Class and code for rendering the ipywidgets dashboard."""
from datetime import datetime
import functools
import os
import random

import ipywidgets as widgets
from ipywidgets import HBox, Layout, VBox, Tab
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import pandas as pd

from tx2 import utils, visualization, wrapper


class Dashboard:
    """Class that handles the setup, visualizations, and customization of the ipywidgets
    dashboard."""

    def __init__(
        self,
        transformer_wrapper: wrapper.Wrapper,
        show_umap=True,
        show_salience=True,
        show_word_count=True,
        show_cluster_salience=True,
        show_cluster_sample_btns=True,
        show_wordclouds=False,
        show_scoring=True,
        point_size=5,
        unfocused_point_size=20,
        highlighted_point_size=75,
    ):
        """Constructor.

        :param transformer_wrapper: the :class:`tx2.wrapper.Wrapper` instance/interface into the transformer.
        :param show_umap: Show the UMAP projection/embedding scatter plot.
        :param show_salience: Show the current entry salience map "heatmap".
        :param show_word_count: Show the per-cluster top word frequency graphs.
        :param show_cluster_salience: Show the top aggregate-salience "important" words per cluster graphs.
        :param show_cluster_sample_btns: Show the sampling buttons for each cluster.
        :param show_wordclouds: Show the wordclouds for each cluster.
        :param show_scoring: Show aggregate scoring metrics, confusion matrices, etc.
        :param point_size: Size to render points in the UMAP plot.
        :param unfocused_point_size: Size to render unfocused background points in the UMAP plot.
        :param selected_point_size: Size to render highlighted and currently selected points in the UMAP plot.
        """

        self.transformer_wrapper = transformer_wrapper
        """The :class:`tx2.wrapper.Wrapper` instance/interface into the transformer."""

        # for handling manual changes from dataframe entries
        self.prior_reference_point = None
        self.prior_reference_text = None
        self.highlight_indices = []

        # keep track of all currently rendered figures for saving purposes
        self.current_figures = {}

        # display options
        self.show_umap = show_umap
        """Show the UMAP projection/embedding scatter plot."""
        self.show_salience = show_salience
        """Show the current entry salience map "heatmap"."""
        self.show_word_count = show_word_count
        """Show the per-cluster top word frequency graphs."""
        self.show_cluster_salience = show_cluster_salience
        """Show the top aggregate-salience "important" words per cluster graphs."""
        self.show_cluster_sample_btns = show_cluster_sample_btns
        """Show the sampling buttons for each cluster."""
        self.show_wordclouds = show_wordclouds
        """Show the wordclouds for each cluster."""
        self.show_scoring = show_scoring
        """Show aggregate scoring metrics, confusion matrices, etc."""

        # colors (rgb codes for category20)
        self.colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
            "#aec7e8",
            "#ffbb78",
            "#98df8a",
            "#ff9896",
            "#c5b0d5",
            "#c49c94",
            "#f7b6d2",
            "#c7c7c7",
            "#dbdb8d",
            "#9edae5",
        ]
        """The hex RGB colors to use in the scatter plot and word frequency graphs to represent each possible
        category. By default this is set to the 'Category20' palette. The number of clusters cannot exceed the
        number of colors in this array. (Currently it's automatically limited to the first 20 clusters.)"""

        self.point_size = point_size
        """Size to render points in the UMAP plot."""
        self.unfocused_point_size = unfocused_point_size
        """Size to render unfocused background points in the UMAP plot."""
        self.highlighted_point_size = highlighted_point_size
        """Size to render highlighted and currently selected points in the UMAP plot."""

        self._initialize_widgets()
        self._create_cluster_sample_buttons()
        self._populate_dropdown_indices()
        self._create_section_layouts()
        self._attach_event_handlers()

    def _initialize_widgets(self):

        # labels
        self.lbl_wordcloud = widgets.HTML(value="<h3>Cluster Word Clouds</h3>")
        """Large HTML label for the large cluster word cloud. ("Cluster Word Clouds")"""
        self.lbl_projection_graph = widgets.HTML(value="<h3>UMAP Embedding Graph</h3>")
        """Large HTML label for the UMAP embedding graph. ("UMAP Embedding Graph")"""
        self.lbl_entry_text = widgets.HTML(value="<h3>Entry Text</h3>")
        """Large HTML label for the entry text box. ("Entry Text")"""
        self.lbl_salience = widgets.HTML(value="<h3>Word Salience Map</h3>")
        """Large HTML label for text salience map. ("Word Salience Map")"""
        self.lbl_cluster_words = widgets.HTML(value="<h3>Visual Cluster Words</h3>")
        """Large HTML label for cluster bar graph visualizations. ("Visual Cluster Words")"""
        self.lbl_class_labels = widgets.HTML(value="<p><b>Model Classification</b></p>")
        """HTML label for the predicted model output. ("Model Classification")"""
        self.lbl_status_labels = widgets.HTML(value="<p><b>Status</b></p>")
        """HTML label for the status indicators section. ("Status")"""
        self.lbl_graph_controls = widgets.HTML(value="<p><b>Graph Controls</b></p>")
        """HTML label for the graph controls. ("Graph Controls")"""
        self.lbl_search = widgets.HTML(value="<p><b>Keyword Search</b></p>")
        """HTML label for the term search box. ("Keyword Search")"""
        self.html_freq_words = widgets.HTML(
            value="<p><b>Frequent Words</b></p><p style='font-size: 10px'>Which words are most common in each cluster, x-axis denoting word count from all entries in the cluster, and color denoting the ratio of occurrence for each class in that cluster.</p>"
        )
        """HTML label and description for the per-cluster frequent word bar graphs. ("Frequent Words")"""
        self.html_important_words = widgets.HTML(
            value="<p><b>Important Words</b></p><p style='font-size: 10px'>Which words cause the largest change in predictions throughout the cluster, x-axis denoting the relative total change caused by word removal from all entries in the cluster.</p>"
        )
        """HTML label and description for the per-cluster important word bar graphs. ("Important Words")"""
        self.html_wordcloud_explanation = widgets.HTML(
            value="<p style='font-size: 10px'>If a keyword search is entered above, the main wordcloud will display word frequency from the currently highlighted points.</p>"
        )
        """HTML description label for main word cloud."""
        self.lbl_sample_btns = widgets.HTML(value="<p><b>Sampling Buttons</b></p>")
        """HTML label for cluster sampling buttons. ("Sampling Buttons")"""

        self.drop_text_picker = widgets.Dropdown(
            options=[],
            description="Selected datapoint index",
            style={"description_width": "initial"},
        )
        """Dropdown menu with all test dataframe row indices."""
        self.text_entry = widgets.Textarea(layout={"height": "200px", "width": "95%"})
        """Textbox containing the currently selected row input text."""
        self.text_search_terms = widgets.Text(layout={"width": "175px"})
        """Search term input box."""
        self.btn_search_sample = widgets.Button(
            description="Sample from highlighted", layout={"width": "175px"}
        )
        """Sample from highlighted button, this randomly selects one of the currently highlighted points."""
        self.btn_misclass_sample = widgets.Button(
            description="Sample misclassified", layout={"width": "175px"}
        )
        """Sample from misclassified button, this randomly selects one of the points the classifier missed."""
        self.btn_savefigs = widgets.Button(
            description="Save all figures", layout={"width": "175px"}
        )
        """Button to save a copy of all currently rendered figures to the cache folder."""

        self.html_text_render = widgets.HTML()
        """HTML for salience map."""
        self.html_target = widgets.HTML()
        """HTML for target class indicator."""
        self.html_prediction = widgets.HTML()
        """HTML for predicted class indicator."""
        self.html_status = widgets.HTML(
            value="<p>"
            + visualization.get_nice_html_label("Ready!", "#008000")
            + "</p>"
        )
        """HTML for status indicator."""
        self.html_graph_status = widgets.HTML(
            value="<p>"
            + visualization.get_nice_html_label("Ready!", "#008000")
            + "</p>"
        )
        """HTML for graphing status indicator."""
        self.chk_show_train = widgets.Checkbox(
            value=False,
            description="Show training data",
            indent=False,
            layout=Layout(width="175px"),
        )
        """Checkbox for whether to display training points in graph as well."""
        self.chk_show_cluster_labels = widgets.Checkbox(
            value=False,
            description="Visual cluster nums",
            indent=False,
            layout=Layout(width="175px"),
        )
        """Checkbox to display the cluster labels."""
        self.chk_focus_errors = widgets.Checkbox(
            value=False,
            description="Focus misclassifications",
            indent=False,
            layout=Layout(width="175px"),
        )
        self.out_projection_scatter = widgets.Output()
        """Output widget for UMAP embedding graph."""
        self.out_wordcloud_big = widgets.Output()
        """Output widget for large word cloud."""
        self.out_wordcloud_set = widgets.Output()
        """Output widget for collection of word clouds."""
        self.out_cluster_word_frequency = widgets.Output()
        """Output widget for the per-cluster frequency word bar graphs."""
        self.out_cluster_word_attention = widgets.Output()
        """Output widget for the per-cluster attention word bar graphs."""
        self.out_confusion_matrix = widgets.Output()
        """Output widget for confusion matrix."""
        self.out_aggregate_metrics = widgets.Output()
        """Output widget for aggregate confusion matrix metrics."""
        self.out_perclass_metric = widgets.Output()
        """Output widget for per-class confusion matrix metrics."""

        self.current_cluster_lbl = widgets.HTML()

    def _create_cluster_sample_buttons(self):
        cluster_buttons_collection = []
        for index, cluster in enumerate(self.transformer_wrapper.clusters):
            btn = widgets.Button(
                description=f"[{str(cluster)}] ({str(len(self.transformer_wrapper.clusters[cluster]))})",
                layout=Layout(width="200px"),
            )
            btn.on_click(
                functools.partial(self.on_cluster_button_clicked, cluster_name=cluster)
            )
            cluster_buttons_collection.append(btn)
        self.cluster_buttons = VBox(
            cluster_buttons_collection, layout=Layout(width="210px")
        )

    def _populate_dropdown_indices(self):
        text_options = []

        for i, text in enumerate(self.transformer_wrapper.test_texts):
            text_options.append((i, [i, text]))

        self.drop_text_picker.options = text_options
        self.drop_text_picker.index = 0

    def _create_section_layouts(self):
        # ------------
        # PROJECTIONS SIDEBAR
        # ------------
        self.status_box = VBox(
            [self.lbl_status_labels, self.html_status, self.html_graph_status]
        )
        """Status section, containing :param:`lbl_status_labels`, :param:`html_status`, :param:`html_graph_status`."""

        self.labels_box = VBox(
            [self.lbl_class_labels, self.html_target, self.html_prediction]
        )

        self.projection_graph_controls_box = VBox(
            [
                self.lbl_graph_controls,
                self.chk_show_train,
                self.chk_show_cluster_labels,
                self.chk_focus_errors,
                self.btn_misclass_sample,
            ]
        )

        self.keyword_search_box = VBox(
            [self.lbl_search, self.text_search_terms, self.btn_search_sample]
        )

        # actual sidebar
        self.projection_graph_sidebar = VBox(
            [
                self.status_box,
                self.labels_box,
                self.projection_graph_controls_box,
                self.keyword_search_box,
            ]
        )

        # ------------
        # PROJECTIONS GRAPH
        # ------------

        # Projection layout
        self.projection_layout = VBox(
            [
                self.lbl_projection_graph,
                HBox([self.out_projection_scatter, self.projection_graph_sidebar]),
            ]
        )

        # ------------
        # OVERALL CONTROLLS
        # ------------

        self.controls_layout = HBox([self.drop_text_picker, self.btn_savefigs])

        # ------------
        # TEXT ENTRY/SALIENCE
        # ------------

        # Text salience and textbox for free text entry
        self.manual_text_entry_and_salience_layout = HBox(
            [
                VBox(
                    [self.lbl_entry_text, self.text_entry], layout=Layout(width="40%")
                ),
                VBox(
                    [self.lbl_salience, self.html_text_render],
                    layout=Layout(width="60%"),
                ),
            ]
        )

        # ------------
        # TABS
        # ------------

        self.tabs = Tab()
        tab_children = {}

        # cluster stuff
        self.cluster_freq_group = VBox(
            [self.html_freq_words, self.out_cluster_word_frequency]
        )
        self.cluster_salience_group = VBox(
            [self.html_important_words, self.out_cluster_word_attention]
        )
        self.word_cloud_group = VBox(
            [
                self.lbl_wordcloud,
                self.html_wordcloud_explanation,
                self.out_wordcloud_big,
                self.out_wordcloud_set,
            ]
        )
        self.scoring_group = VBox(
            [
                HBox([self.out_confusion_matrix, self.out_aggregate_metrics]),
                self.out_perclass_metric,
            ]
        )

        # this can be included in both the word graphs and word clouds tabs
        self.sampling_group = VBox(
            [self.lbl_sample_btns, self.cluster_buttons], layout=Layout(width="20%")
        )

        bar_cluster_sections = [self.lbl_cluster_words]
        if self.show_word_count:
            bar_cluster_sections.append(self.cluster_freq_group)
        if self.show_cluster_salience:
            bar_cluster_sections.append(self.cluster_salience_group)

        if len(bar_cluster_sections) > 0:
            tab_children["Cluster words"] = HBox(
                [
                    VBox(bar_cluster_sections, layout=Layout(width="80%")),
                    self.sampling_group,
                ]
            )

        # TODO: condition on show_cluster_sample_btns
        if self.show_wordclouds:
            tab_children["Word clouds"] = HBox(
                [
                    VBox([self.word_cloud_group], layout=Layout(width="80%")),
                    self.sampling_group,
                ]
            )

        if self.show_scoring:
            tab_children["Scoring"] = self.scoring_group

        # set tabs and fix titles
        self.tabs.children = list(tab_children.values())
        for index, key in enumerate(tab_children.keys()):
            self.tabs.set_title(index, key)

        # ------------
        # COMBINE VISIBLE
        # ------------

        # Combine all visible groups and sections
        visible_sections = []

        if self.show_umap:
            visible_sections.append(self.projection_layout)

        visible_sections.append(self.controls_layout)

        if self.show_salience:
            visible_sections.append(self.manual_text_entry_and_salience_layout)

        visible_sections.append(self.tabs)

        self.dashboard_layout = VBox(visible_sections)

    def _attach_event_handlers(self):
        self.drop_text_picker.observe(self.on_text_picker_change, names="value")
        self.text_entry.observe(self.on_text_area_change, names="value")
        self.chk_show_train.observe(self.on_change_show_train, names="value")
        self.chk_show_cluster_labels.observe(self.on_change_show_words, names="value")
        self.chk_focus_errors.observe(self.on_change_focus_errors, names="value")
        self.text_search_terms.observe(self.on_search_term_change, names="value")
        self.btn_search_sample.on_click(self.on_sample_button_clicked)
        self.btn_misclass_sample.on_click(self.on_sample_misclass_button_clicked)
        self.btn_savefigs.on_click(self.on_savefigs_button_clicked)

    def render(self):
        """Return combined layout widget"""
        if self.show_wordclouds:
            visualization.prepare_wordclouds(
                self.transformer_wrapper.clusters, self.transformer_wrapper.test_texts
            )
            with self.out_wordcloud_set:
                clear_output(wait=True)
                fig_wordcloud_grid = visualization.plot_wordclouds(self)
                self.current_figures["wordcloud_grid"] = fig_wordcloud_grid
                display(fig_wordcloud_grid)

        plt.ioff()

        with self.out_cluster_word_frequency:
            clear_output(wait=True)
            fig = visualization.plot_clusters_stacked(
                self.transformer_wrapper.clusters,
                self.transformer_wrapper.cluster_class_word_sets,
                self.transformer_wrapper.encodings,
                self.colors,
            )
            self.current_figures["cluster_word_frequency"] = fig
            display(fig)

        with self.out_cluster_word_attention:
            clear_output(wait=True)
            fig = visualization.plot_clusters(
                self.transformer_wrapper.clusters,
                self.transformer_wrapper.cluster_profiles,
            )
            self.current_figures["cluster_word_salience"] = fig
            display(fig)

        with self.out_confusion_matrix:
            clear_output(wait=True)
            fig = visualization.plot_confusion_matrix(
                self.transformer_wrapper.predictions,
                self.transformer_wrapper.test_labels,
                self.transformer_wrapper.encodings,
            )
            self.current_figures["confusion_matrix"] = fig
            display(fig)

        display_per_df, display_agg_df = visualization.plot_metrics(
            self.transformer_wrapper.predictions,
            self.transformer_wrapper.test_labels,
            self.transformer_wrapper.encodings,
        )

        with self.out_aggregate_metrics:
            clear_output(wait=True)
            display(display_agg_df)

        with self.out_perclass_metric:
            clear_output(wait=True)
            display(display_per_df)

        display(self.dashboard_layout)
        self.on_text_picker_change(None)

    # EVENT HANDLERS

    def on_text_picker_change(self, change):
        self.html_status.value = (
            "<p>"
            + visualization.get_nice_html_label("Computing...", "#FF0000")
            + "</p>"
        )
        index = self.drop_text_picker.value[0]
        class_num = self.transformer_wrapper.test_labels[index]
        label = utils.get_cat_by_index(class_num, self.transformer_wrapper.encodings)
        self.html_target.value = (
            "<p>Target: "
            + visualization.get_nice_html_label(label, self.colors[class_num])
            + "</p>"
        )

        new_text = self.transformer_wrapper.test_texts[index]
        # below: THIS IS TECHNICALLY INCORRECT, shows the color of correct class instead of
        # predicted, but still need index. Leaving this here and then just modifying color in
        # on_text_area_change
        self.prior_reference_point = (
            self.transformer_wrapper.projections_testing[index],
            self.colors[class_num],
        )
        self.prior_reference_text = new_text

        self.text_entry.value = new_text
        if self.show_wordclouds and len(self.highlight_indices) == 0:
            with self.out_wordcloud_big:
                clear_output(wait=True)
                fig = visualization.plot_big_wordcloud(
                    int(index), self.transformer_wrapper.clusters
                )
                self.current_figures["active_wordcloud"] = fig
                display(fig)

    def on_text_area_change(self, change):
        self.html_status.value = (
            "<p>"
            + visualization.get_nice_html_label("Computing...", "#FF0000")
            + "</p>"
        )
        new_text = self.text_entry.value
        self.html_text_render.value = visualization.render_html_text(
            new_text, self.transformer_wrapper
        )

        prediction = self.transformer_wrapper.classify([new_text])[0]
        prediction_label = utils.get_cat_by_index(
            prediction, self.transformer_wrapper.encodings
        )
        self.html_prediction.value = (
            "<p>Predicted: "
            + visualization.get_nice_html_label(
                prediction_label, self.colors[prediction]
            )
            + "</p>"
        )
        # here we fix the color from above (on_text_picker_change)
        if new_text == self.prior_reference_text:
            self.prior_reference_point = (
                self.prior_reference_point[0],
                self.colors[prediction],
            )

        self.html_status.value = (
            "<p>" + visualization.get_nice_html_label("Ready!", "#008000") + "</p>"
        )
        self.html_graph_status.value = (
            "<p>" + visualization.get_nice_html_label("Graph stale", "#FFA500") + "</p>"
        )
        visualization.plot_embedding_projections(new_text, self, prediction)

    @utils.debounce(0.5)
    def on_search_term_change(self, change):
        if self.text_search_terms.value != "":
            self.highlight_indices = self.transformer_wrapper.search_test_df(
                self.text_search_terms.value
            )
        else:
            self.highlight_indices = []

        if len(self.highlight_indices) > 0:
            if self.show_wordclouds:
                with self.out_wordcloud_big:
                    clear_output(wait=True)
                    fig = visualization.plot_passed_wordcloud(
                        visualization.gen_wordcloud(
                            self.transformer_wrapper.test_texts[self.highlight_indices]
                        ),
                        "custom:" + self.text_search_terms.value,
                    )
                    self.current_figures["active_wordcloud"] = fig
                    display(fig)

        visualization.plot_embedding_projections(self.text_entry.value, self)

    def on_sample_button_clicked(self, change):
        result_index = random.choice(self.highlight_indices)
        self.drop_text_picker.index = result_index

    def on_sample_misclass_button_clicked(self, change):
        temp_df = pd.DataFrame.from_dict(
            {
                "predicted": self.transformer_wrapper.predictions,
                "target": self.transformer_wrapper.test_labels,
            }
        )

        indices = temp_df[temp_df.target != temp_df.predicted].index
        result_index = random.choice(list(indices))
        self.drop_text_picker.index = result_index

    # TODO still not actually sure if cluster_name is technically a string or not, double check this
    def on_cluster_button_clicked(self, change, cluster_name: str):
        result_index = random.choice(self.transformer_wrapper.clusters[cluster_name])
        self.drop_text_picker.index = result_index

    def on_change_show_train(self, change):
        visualization.plot_embedding_projections(self.text_entry.value, self)

    def on_change_show_words(self, change):
        visualization.plot_embedding_projections(self.text_entry.value, self)

    def on_change_focus_errors(self, change):
        visualization.plot_embedding_projections(self.text_entry.value, self)

    def on_savefigs_button_clicked(self, change):
        self.html_status.value = (
            "<p>" + visualization.get_nice_html_label("Saving...", "#FF0000") + "</p>"
        )

        # make directory in cache for current dump
        folder_name = datetime.now().strftime("%Y-%m-%d")
        count = 0
        for filename in os.listdir(self.transformer_wrapper.cache_path):
            if filename.startswith(folder_name):
                count += 1
        folder_name += "-" + str(count)
        folder = self.transformer_wrapper.cache_path + "/" + folder_name
        os.makedirs(folder)

        for key, value in self.current_figures.items():
            value.savefig(f"{folder}/{key}.png", format="png", transparent=False)

        self.html_status.value = (
            "<p>" + visualization.get_nice_html_label("Ready!", "#008000") + "</p>"
        )
