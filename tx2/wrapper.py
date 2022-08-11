"""The wrapper class around a transformer and its functionality."""

import logging
import os
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
import umap

# TODO: not crazy about this, but library agnosticism later
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from tx2 import calc, dataset, utils
from tx2.cache import check, read, write


class Wrapper:
    """A wrapper or interface class between a transformer and the dashboard.

    This class handles running all of the calculations for the data needed by
    the front-end visualizations.
    """

    # TODO: option to not do aggregate word salience (longest running step by far)

    def __init__(
        self,
        train_texts: Union[np.ndarray, pd.Series],
        train_labels: Union[np.ndarray, pd.Series],
        test_texts: Union[np.ndarray, pd.Series],
        test_labels: Union[np.ndarray, pd.Series],
        encodings: Dict[str, int],
        classifier=None,
        language_model=None,
        tokenizer=None,
        device=None,
        cache_path="data",
        overwrite=False,
    ):
        """Constructor.

        :param train_texts: A set of text entries that were used during the model's training process.
        :param train_labels: The set of class labels for train_texts.
        :param test_texts: The set of text entries that the model hadn't seen during training.
        :param test_labels: The set of class labels for test_texts.
        :param encodings: A dictionary associating class label names with integer values.
        :param classifier: A class/network containing a language model and classification head.
            **Running this variable as a function by default should send the passed inputs through
            the entire network and return the argmaxed classification index (reverse encoding)**.
            Note that **this argument is not required**, if the user intends to manually specify
            classification functions.
        :param language_model: A `huggingface transformer model
            <https://huggingface.co/transformers/main_classes/model.html>`_, if a custom network
            class is being used and has a layer representing the output of just the language model,
            pass it here. Note that **this argument is not required**, if the user intends to
            manually specify classification functions.
        :param tokenizer: A `huggingface tokenizer
            <https://huggingface.co/transformers/main_classes/tokenizer.html>`_. Note that **this
            argument is not required**, if the user intends to manually specify encode and
            classification functions.
        :param device: Set the device for pytorch to place tensors on, pass either "cpu", 
            "cuda", or "mps". This variable is used by the default embedding function. 
            If unspecified, "cuda" or "mps" will be used if GPU is found, otherwise it defaults to "cpu".
        :param cache_path: The directory path to cache intermediate outputs from the
            :meth:`tx2.wrapper.Wrapper.prepare` function. This allows the wrapper to precompute
            needed values for the dashboard to reduce render time and allow rerunning all wrapper
            code without needing to recompute. Note that every wrapper/dashboard instance is expected
            to have a unique cache path, otherwise filenames will conflict. You will need to set
            this if you intend to use more than one dashboard.
        :param overwrite: Whether to ignore the cache and overwrite previous results or not.
        """

        # special override functions
        self.embedding_function = None
        """A function to take a single set of inputs and return embedded versions - a sequence
        representation from the language model. This variable points to a sensible default function
        based on a language model layer being specified in the constructor. If classifier or language
        model were not specified to the constructor, **this variable must be assigned to a custom
        function definition.**

        .. admonition:: Example

            Below is a simplified example of creating a customized embed function.
            :code:`my_custom_embedding_function` will be used by the wrapper, and will be
            called with an array of pre-encoded inputs for a single entry, and is expected
            to return an array. (TODO: 1d or 2d?)

            .. code-block:: python

                def my_custom_embedding_function(inputs):
                    return np.mean(my_transformer(inputs['input_id'], inputs['attention_mask'])[0])

                wrapper = Wrapper(...)
                wrapper.embedding_function = my_custom_embedding_function
        """
        self.classification_function = None
        """A function to take a single set of inputs and return the index of the predicted class."""
        self.soft_classification_function = None
        """A function to take a single set of inputs and return the (non arg-maxed) output layer of
        the network."""
        self.encode_function = None
        """A function to take a single text entry and return an encoded version of it. The default
        function will utilize the tokenizer given in the constructor if available."""

        self.train_texts = train_texts
        """Collection of all text entries used during models training process."""
        self.train_labels = train_labels
        """Collection of all class labels used during models training process."""
        self.test_texts = test_texts
        """Collection of all text entries used during models testing process."""
        self.test_labels = test_labels
        """Collection of all class labels used during models testing process."""

        # convert any bad indices
        if type(train_texts) == pd.Series:
            self.train_texts = self.train_texts.reset_index(drop=True)
        if type(train_labels) == pd.Series:
            self.train_labels = self.train_labels.reset_index(drop=True)
        if type(test_texts) == pd.Series:
            self.test_texts = self.test_texts.reset_index(drop=True)
        if type(test_labels) == pd.Series:
            self.test_labels = self.test_labels.reset_index(drop=True)

        self.encodings = encodings
        """A dictionary associating class label names with integer values.

        example:

        .. code-block::

            {
                "label1": 0,
                "label2": 1,
            }
        """

        self.classifier = classifier
        """A class containing the entire network, which can be called as a function taking the
        encoded input and returning the output classification."""
        self.language_model = language_model
        """A variable containing only the huggingface language model portion of the network."""
        self.tokenizer = tokenizer
        """The huggingface tokenizer to use for encoding text input."""
        self.device = device
        """Set the device for pytorch to place tensors on, pass either "cpu", "cuda", or "mps". This
        variable is used by the default embedding function. If unspecified and a GPU is found,
        "cuda" or "mps" will be used, otherwise it defaults to "cpu"."""
        if self.device is None:
            self.device = utils.get_device()

        self.cache_path = cache_path
        """The directory path to cache pre-calculated values."""
        self.overwrite = overwrite
        """Whether to ignore cached calculations and overwrite them or not."""

        # paths for caching specific items
        self.predictions_path = f"{self.cache_path}/predictions.json"
        self.embeddings_training_path = f"{self.cache_path}/embedding_training.json"
        self.embeddings_testing_path = f"{self.cache_path}/embedding_testing.json"

        self.projector_path = f"{self.cache_path}/projector.pkl.gz"
        self.projections_training_path = f"{self.cache_path}/projections_training.json"
        self.projections_testing_path = f"{self.cache_path}/projections_testing.json"

        self.salience_maps_path = f"{self.cache_path}/salience.pkl.gz"

        self.clusters_path = f"{self.cache_path}/clusters.json"
        self.cluster_profiles_path = f"{self.cache_path}/cluster_profiles.pkl.gz"
        self.cluster_labels_path = f"{self.cache_path}/cluster_labels.json"
        self.cluster_words_path = f"{self.cache_path}/cluster_words.json"
        self.cluster_class_words_path = f"{self.cache_path}/cluster_class_words.json"

        # if following the pre-determined logical flow for model creation etc. use these simplified
        # default functions that directly call the correct parts of the model
        if classifier is not None:
            self.classification_function = self._default_classification_function
            self.soft_classification_function = (
                self._default_soft_classification_function
            )
            classifier.eval()  # TODO: may need to move this out, shouldn't be strictly dependent on using torch?

        if language_model is not None:
            self.embedding_function = self._default_embedding_function

        if tokenizer is not None:
            self.encode_function = self._default_encoding_function

        # other defaults
        self.max_clusters = 20
        """Maximum number of clusters to retain. Note that this cannot exceed the number of colors in the dashboard."""
        self.batch_size = 2
        """The batch size to use in backend dataloader creation."""
        self.max_len = 256
        """The maximum length of each text entry, based on the expected input size of the transformer."""

        # NOTE: only relevant if using default encode function
        self.encoder_options = dict(
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
        )
        """The default options to pass to the tokenizer's :code:`encode_plus()` function. See
        `huggingface documentation <https://huggingface.co/transformers/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.encode_plus>`_."""

        # flags
        self.projection_model_ready = False
        self.salience_computed = False

        # Precomputed data store (just here for documentation purposes)
        self.predictions = None
        """The predicted class for each entry in :code:`test_texts`, as returned by
        :meth:`tx2.wrapper.Wrapper.classify`."""
        self.embeddings_training = None
        """Precomputed embeddings for each entry in :code:`train_texts`, as returned by
        :meth:`tx2.wrapper.Wrapper.embed`."""
        self.embeddings_testing = None
        """Precomputed embeddings for each entry in :code:`test_texts`, as returned by
        :meth:`tx2.wrapper.Wrapper.embed`."""
        self.projector = None
        """The trained UMAP projector. See `umap-learn documentation
        <https://umap-learn.readthedocs.io/en/latest/>`_."""
        self.projections_training = None
        """The two dimensional projections of :code:`embeddings_training`, for each entry in :code:`train_texts`."""
        self.projections_testing = None
        """The two dimensional projections of :code:`embeddings_testing`, for each entry in :code:`test_texts`."""
        self.salience_maps = None
        """The salience map for each entry in :code:`test_texts` as calculated by :meth:`tx2.calc.salience_map`."""
        self.clusters = None
        """A dictionary of cluster names, each associated with a list of indices of points in that cluster, as
         calculated by :meth:`tx2.calc.cluster_projections`."""
        self.cluster_profiles = None
        """A dictionary of aggregate sorted salience maps for each cluster as calculated by
        :meth:`tx2.calc.aggregate_cluster_salience_maps`."""
        self.cluster_word_freqs = None
        """A dictionary of clusters and sorted top word frequencies for each, as calculated by
        :meth:`tx2.calc.frequent_words_in_cluster`"""
        self.cluster_class_word_sets = None
        """A dictionary of clusters, further divided version of :code:`cluster_word_freqs` that divides each
        word count up into the number of entries of each category containing that word, as calculated by
        :meth:`tx2.calc.frequent_words_by_class_in_cluster`."""

    def _run_predictions(self):
        """ """
        logging.info("Running classifier...")
        self.predictions = self.classify(self.test_texts)
        # self.test_df["predicted_classes"] = self.predictions

        logging.info("Saving predictions...")
        write(self.predictions, self.predictions_path)
        logging.info("Done!")

    def _train_projector(self, **umap_args):
        """Train the UMAP projector and run it on the test embeddings. Caches results."""
        logging.info("Training projector...")

        umap_args = utils.set_defaults(
            umap_args, n_neighbors=30, min_dist=0.5, random_state=42
        )
        logging.debug("UMAP arguments: %s", str(umap_args))

        logging.debug("Running embeddings through umap")
        # TODO
        self.projector = umap.UMAP(**umap_args)
        self.projections_training = self.projector.fit_transform(
            self.embeddings_training
        )

        logging.info("Applying projector to test dataset...")
        self.projections_testing = self.projector.transform(self.embeddings_testing)

        logging.info("Saving projections...")
        write(self.projections_training.tolist(), self.projections_training_path)
        write(self.projections_testing.tolist(), self.projections_testing_path)
        write(self.projector, self.projector_path)
        logging.info("Done!")

        self.projection_model_ready = True

    def _embed_data(self):
        """Run the transformer language model on training and test data."""
        # TODO: is data loader necessary?
        logging.info("Embedding training and testing datasets")

        self.embeddings_training = self.embed(self.train_texts)
        self.embeddings_testing = self.embed(self.test_texts)

        logging.info("Saving embeddings...")
        write(self.embeddings_training, self.embeddings_training_path)
        write(self.embeddings_testing, self.embeddings_testing_path)
        logging.info("Done!")

    # TODO: move out?
    def _determine_cluster_label(self, cluster, cluster_profiles, cluster_name):
        """Determine the center point to render a cluster label at"""
        if type(self.test_texts) == pd.Series:
            projections = self.project(self.test_texts[cluster].reset_index(drop=True))
        else:
            projections = self.project(self.test_texts[cluster])

        x = []
        y = []
        for point in projections:
            x.append(point[0])
            y.append(point[1])

        x_center = np.mean(x)
        y_center = np.mean(y)

        label = str(cluster_name)
        return x_center, y_center, label

    def _compute_all_salience_maps(self):
        """Get a salience map of every test entrypoint and store it. This is one of
        the longest running steps (~1s per entry), and is separate so that it doesn't
        have to be recomputed just because the user wants to try different clusterings
        (which does not impact salience)"""

        logging.info("Computing salience maps...")
        self.salience_maps = []
        for entry in tqdm(self.test_texts, total=len(self.test_texts)):
            deltas = calc.salience_map(
                self.soft_classify, entry[: self.max_len], self.encodings
            )
            self.salience_maps.append(deltas)
        logging.info("Saving salience maps...")
        write(self.salience_maps, self.salience_maps_path)
        logging.info("Done!")

        self.salience_computed = True

    def _compute_visual_clusters(self, clustering_alg, **clustering_args):
        """Find clusters from 2d projections."""
        if not self.salience_computed:
            raise RuntimeError("Salience maps have not been computed")

        if clustering_alg == "DBSCAN":
            clustering_args = utils.set_defaults(clustering_args, eps=1, min_samples=5)

        logging.info("Clustering projections...")
        self.clusters = calc.cluster_projections(
            self.projections_testing, clustering_alg, **clustering_args
        )
        if len(self.clusters) > 20:
            logging.error(
                "More than 20 clusters found - cutting off all but first 20. Try different clustering arguments."
            )
            self.clusters = {
                key: val
                for key, val in self.clusters.items()
                if key in list(self.clusters.keys())[:20]
            }
            # self.clusters = self.clusters[:20]

        self.cluster_profiles = calc.aggregate_cluster_salience_maps(
            self.clusters, self.salience_maps
        )

        logging.info("Saving cluster profiles...")
        write(self.clusters, self.clusters_path)
        write(self.cluster_profiles, self.cluster_profiles_path)
        logging.info("Done!")

        # get the labels and label positions
        self.cluster_labels = []
        for index, cluster in enumerate(self.clusters):
            x, y, label = self._determine_cluster_label(
                self.clusters[cluster], self.cluster_profiles, cluster
            )
            self.cluster_labels.append((x, y, label))
        logging.info("Saving cluster labels...")
        write(self.cluster_labels, self.cluster_labels_path)
        logging.info("Done!")

        # get word freqs
        self.cluster_word_freqs = {}
        self.cluster_class_word_sets = {}
        for cluster in self.clusters:
            freq_words = calc.frequent_words_in_cluster(
                self.test_texts[self.clusters[cluster]]
            )[:10]
            words_by_class = calc.frequent_words_by_class_in_cluster(
                freq_words,
                self.encodings,
                self.test_texts[self.clusters[cluster]],
                self.test_labels[self.clusters[cluster]],
            )
            self.cluster_word_freqs[cluster] = freq_words
            self.cluster_class_word_sets[cluster] = words_by_class

        logging.info("Saving cluster word counts...")
        write(self.cluster_word_freqs, self.cluster_words_path)
        write(self.cluster_class_word_sets, self.cluster_class_words_path)
        logging.info("Done!")

    # TODO
    def _default_encoding_function(self, text):
        encoded = self.tokenizer.encode_plus(text, None, **self.encoder_options)
        return {
            "input_ids": torch.tensor(encoded["input_ids"], device=self.device),
            "attention_mask": torch.tensor(
                encoded["attention_mask"], device=self.device
            ),
        }

    def _default_classification_function(self, inputs):
        output = self.classifier(inputs["input_ids"], inputs["attention_mask"])
        if type(output) != torch.Tensor:
            output = output.logits
        return torch.argmax(output, dim=1)

    def _default_embedding_function(self, inputs):
        return self.language_model(inputs["input_ids"], inputs["attention_mask"])[0][
            :, 0, :
        ]  # [CLS] token embedding

    def _default_soft_classification_function(self, inputs):
        output = self.classifier(inputs["input_ids"], inputs["attention_mask"])
        if type(output) != torch.Tensor:
            output = output.logits
        return output

    def _prepare_input_data(self, texts):
        encoded_data = dataset.EncodedDataset(texts, self)
        loader = DataLoader(
            encoded_data, shuffle=False, num_workers=0, batch_size=self.batch_size
        )
        return loader

    def recompute_visual_clusterings(self, clustering_alg="DBSCAN", clustering_args={}):
        """Re-run the clustering algorithm. Note that this automatically overrides any
        previously cached data for clusters.

        :param clustering_alg: The name of the clustering algorithm to use, a class name
            from sklearn.cluster, see `sklearn's documentation <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster>`_.
            ("DBSCAN", "KMeans", "AffinityPropagation", "Birch", "OPTICS", "AgglomerativeClustering",
            "SpectralClustering", "SpectralBiclustering", "SpectralCoclustering", "MiniBatchKMeans",
            "FeatureAgglomeration", "MeanShift")
        :param clustering_args: Dictionary of arguments to pass into the clustering algorithm on instantiation.
        """
        self._compute_visual_clusters(clustering_alg, **clustering_args)

    def recompute_projections(
        self, umap_args={}, clustering_alg="DBSCAN", clustering_args={}
    ):
        """Re-run both projection training and clustering algorithms. Note that this
        automatically overrides both previously saved projections as well as clustering
        data.

        :param umap_args: Dictionary of arguments to pass into the UMAP model on instantiation.
        :param clustering_alg: The name of the clustering algorithm to use, a class name
            from sklearn.cluster, see `sklearn's documentation <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster>`_.
            ("DBSCAN", "KMeans", "AffinityPropagation", "Birch", "OPTICS", "AgglomerativeClustering",
            "SpectralClustering", "SpectralBiclustering", "SpectralCoclustering", "MiniBatchKMeans",
            "FeatureAgglomeration", "MeanShift")
        :param clustering_args: Dictionary of arguments to pass into clustering algorithm on instantiation.
        """
        self._train_projector(**umap_args)
        self._compute_visual_clusters(clustering_alg, **clustering_args)

    def encode(self, text: str):
        """Encode/tokenize passed text into a format expected by transformer.

        :param text: The text entry to tokenize.
        :return: A tokenized version of the text, by default this calls :code:`encode_plus()` with
            the options specified in :code:`encoder_options`, and returns a dictionary:

        .. code-block::

           {
                "input_ids": [],
                "attention_mask": []
           }
        """
        encodings = self.encode_function(text)
        return encodings

    def soft_classify(self, texts: List[str]) -> List[List[float]]:
        """Get the non-argmaxed final prediction layer outputs of the classification head.

        :param texts: An array of texts to predict on.
        :return: An Nxd array of arrays, N the number of entries to predict on and d the number of
            categories.
        """
        loader = self._prepare_input_data(texts)
        outputs = []
        for data in loader:
            output = self.soft_classification_function(data)
            output = output.to("cpu").detach().tolist()
            outputs.extend(output)
        return outputs

    def classify(self, texts: List[str]) -> List[int]:
        """Predict the category of each passed entry text.

        :param texts: An array of texts to predict on.
        :return: An array of predicted classes, whose labels can be reverse looked up through
            :code:`encodings`.
        """
        loader = self._prepare_input_data(texts)
        outputs = []
        for data in loader:
            output = self.classification_function(data)
            output = output.to("cpu").detach().tolist()
            outputs.extend(output)
        return outputs

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Get a sequence embedding from the language model for each passed text entry.

        :param texts: An array of texts to embed.
        :return: An array of sequence embeddings.
        """
        loader = self._prepare_input_data(texts)
        outputs = []
        for index, data in enumerate(loader):
            output = self.embedding_function(data)
            output = output.to("cpu").detach().tolist()
            outputs.extend(output)
        return outputs

    def project(self, texts: List[str]) -> np.ndarray:
        """Use the wrapper's UMAP model to project passed texts into two dimensions.

        :param texts: An array of texts to embed.
        :return: A Nx2 numpy array, containing a size 2 array of coordinates for each of the N
            input text entries.
        """
        if not self.projection_model_ready:
            raise RuntimeError("Projection model has not yet been trained")

        embeddings = self.embed(texts)
        return self.projector.transform(embeddings)

    def prepare(self, umap_args={}, clustering_alg="DBSCAN", clustering_args={}):
        """Run all necessary precompute step to support the dashboard. **This function must
        be called before using in a dashboard instance.**

        :param umap_args: Dictionary of arguments to pass into the UMAP model on instantiation.
        :param clustering_alg: The name of the clustering algorithm to use, a class name
            from sklearn.cluster, see `sklearn's documentation <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster>`_.
            ("DBSCAN", "KMeans", "AffinityPropagation", "Birch", "OPTICS", "AgglomerativeClustering",
            "SpectralClustering", "SpectralBiclustering", "SpectralCoclustering", "MiniBatchKMeans",
            "FeatureAgglomeration", "MeanShift")
        :param clustering_args: Dictionary of arguments to pass into clustering algorithm on instantiation.
        """
        logging.debug("Training data shape: %s", self.train_texts.shape)
        logging.debug("Testing data shape: %s", self.test_texts.shape)

        # check for cache path
        if not os.path.isdir(self.cache_path):
            logging.info("Cache path not found, creating...")
            os.makedirs(self.cache_path, exist_ok=True)
        else:
            logging.info("Cache path found")

        # check for predictions
        logging.info("Checking for cached predictions...")
        if check(self.predictions_path, self.overwrite):
            self.predictions = read(self.predictions_path)
            # self.test_df["predicted_classes"] = self.predictions
        else:
            self._run_predictions()
        logging.debug("Predictions example:")
        logging.debug("%s", utils.array_elipsis(self.predictions, 10))
        logging.debug("Predictions type: %s", type(self.predictions))
        logging.debug("Predictions size: %s", len(self.predictions))

        # check for embeddings
        logging.info("Checking for cached embeddings...")
        if check(self.embeddings_training_path, self.overwrite) and check(
            self.embeddings_testing_path, self.overwrite
        ):
            self.embeddings_training = read(self.embeddings_training_path)
            self.embeddings_testing = read(self.embeddings_testing_path)
        else:
            self._embed_data()
        logging.debug("Embeddings example:")
        logging.debug("%s", utils.array_elipsis(self.embeddings_training, 4, 2))
        logging.debug("Embeddings type: %s", type(self.embeddings_training))
        logging.debug(
            "Embeddings size (training): (%s, %s)",
            len(self.embeddings_training),
            len(self.embeddings_training[0]),
        )
        logging.debug(
            "Embeddings size (testing): (%s, %s)",
            len(self.embeddings_testing),
            len(self.embeddings_testing[0]),
        )

        logging.info("Checking for cached projections...")
        if (
            check(self.projections_training_path, self.overwrite)
            and check(self.projections_testing_path, self.overwrite)
            and check(self.projector_path, self.overwrite)
        ):
            self.projections_training = read(self.projections_training_path)
            self.projections_testing = read(self.projections_testing_path)
            self.projector = read(self.projector_path)
            self.projection_model_ready = True
        else:
            self._train_projector(**umap_args)
        logging.debug("Projections example:")
        logging.debug("%s", utils.array_elipsis(self.projections_training, 3, 2))
        logging.debug("Projections type: %s", type(self.projections_training))
        logging.debug(
            "Projections size (training): (%s, %s)",
            len(self.projections_training),
            len(self.projections_training[0]),
        )
        logging.debug(
            "Projections size (testing): (%s, %s)",
            len(self.projections_testing),
            len(self.projections_testing[0]),
        )

        logging.info("Checking for cached salience maps...")
        if check(self.salience_maps_path, self.overwrite):
            self.salience_maps = read(self.salience_maps_path)
            self.salience_computed = True
        else:
            self._compute_all_salience_maps()

        logging.info("Checking for cached cluster profiles...")
        if (
            check(self.cluster_profiles_path, self.overwrite)
            and check(self.clusters_path, self.overwrite)
            and check(self.cluster_labels_path, self.overwrite)
            and check(self.cluster_words_path, self.overwrite)
            and check(self.cluster_class_words_path, self.overwrite)
        ):
            self.clusters = read(self.clusters_path)
            self.cluster_profiles = read(self.cluster_profiles_path)
            self.cluster_labels = read(self.cluster_labels_path)
            self.cluster_word_freqs = read(self.cluster_words_path)
            self.cluster_class_word_sets = read(self.cluster_class_words_path)
        else:
            self._compute_visual_clusters(clustering_alg, **clustering_args)

    def search_test_df(self, search: str) -> List[int]:
        """Get a list of test dataframe indices that have any of the listed terms in the passed string.

        :param search: The search string, can contain multiple terms delimited with '&' to search for
            entries that have all of the terms.
        :return: A list of the indices for the :code:`test_df`.
        """
        search_array = search.split("&")

        temp_df = pd.DataFrame.from_dict({"text": self.test_texts})

        index_df = temp_df[temp_df.text.str.contains(search_array[0], regex=True)].index

        if len(search_array) > 1:
            for i in range(1, len(search_array)):
                index_df = index_df.intersection(
                    temp_df[
                        temp_df.text.str.contains(search_array[0], regex=True)
                    ].index
                )

        return list(index_df)
