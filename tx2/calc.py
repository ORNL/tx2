"""Helper calculation functions for the wrapper and dashboard."""

from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans, AffinityPropagation, Birch, OPTICS, AgglomerativeClustering, SpectralClustering, SpectralBiclustering, SpectralCoclustering, MiniBatchKMeans, FeatureAgglomeration, MeanShift
from sklearn.feature_extraction.text import CountVectorizer
from typing import Dict, List, Tuple, Any

from tx2 import utils


def cluster_projections(projections, clustering_alg, **clustering_args) -> Dict[str, List[int]]:
    """Runs a clustering algorithm (currently only dbscan supported) on the
    provided embedded or projected points, and provides a dictionary of data point
    indices.

    :param projections: The data points to fit - should be numpy array of
        testing data points. Intended use is 2D UMAP projections, but should
        support any shape[1] size.
    :param clustering_alg: The name of the clustering algorithm to use, a class name from sklearn.cluster, see `sklearn's documentation <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster>`_. (:code:`"DBSCAN", "KMeans", "AffinityPropagation", "Birch", "OPTICS", "AgglomerativeClustering", "SpectralClustering", "SpectralBiclustering", "SpectralCoclustering", "MiniBatchKMeans", "FeatureAgglomeration", "MeanShift"`
    :param clustering_args: Any options to pass to the clustering algorithm.
        

    :return: A dictionary where each key is the cluster label and each value is
        an array of the indices from the projections array that are in that cluster.
    """
    # TODO: at some point add ability to use different unsupervised clustering algs

    alg = None
    if clustering_alg == "DBSCAN":
        alg = DBSCAN
    elif clustering_alg == "KMeans":
        alg = KMeans
    elif clustering_alg == "AffinityPropagation":
        alg = AffinityPropagation
    elif clustering_alg == "Birch":
        alg = Birch
    elif clustering_alg == "OPTICS":
        alg = OPTICS
    elif clustering_alg == "AgglomerativeClustering":
        alg = AgglomerativeClustering
    elif clustering_alg == "SpectralClustering":
        alg = SpectralClustering
    elif clustering_alg == "SpectralBiclustering":
        alg = SpectralBiclustering
    elif clustering_alg == "SpectralCoclustering":
        alg = SpectralCoclustering
    elif clustering_alg == "MiniBatchKMeans":
        alg = MiniBatchKMeans
    elif clustering_alg == "FeatureAgglomeration":
        alg = FeatureAgglomeration
    elif clustering_alg == "MeanShift":
        alg = MeanShift

    clustering = alg(**clustering_args).fit(projections)
    clusters = {}
    for index, entry in enumerate(clustering.labels_):
        if str(entry) not in clusters:
            clusters[str(entry)] = []
        clusters[str(entry)].append(index)

    return clusters


def aggregate_cluster_salience_maps(
    clusters: Dict[str, List[int]], salience_maps
) -> Dict[str, List[Tuple[str, float]]]:
    """Create an "aggregate" salience map for each cluster. This function combines
    the impact for each word from every instance where it appears in that cluster,
    and then sorts the results. This gives you how much overall impact the removal
    of a word has on all entries in that cluster - a proxy for importance of that word
    to that cluster.

    :param clusters: A dictionary of cluster names and arrays of associated point indices.
        This can directly take the output from :meth:`tx2.calc.cluster_projections`.
    :param salience_maps: A list of salience maps as computed in :meth:`tx2.calc.salience_map`.

    :return: A dictionary with an array for each cluster, where the array consists of
        tuples - the first element is the word, and the second is the aggregated impact:
        :code:`('WORD', AGGREGATE_DELTA)`
    """

    # iterate each cluster and get the aggregate salience map
    cluster_profiles = {}
    for cluster in clusters:
        cluster_profile = {}
        for index in clusters[cluster]:
            deltas = salience_maps[index]
            changes = sort_salience_map(deltas)

            # aggregate the deltas of each word across the entire current cluster
            already_seen_words = []
            for word, score in changes:
                # don't duplicate if we've already seen this word in this particular log
                if word in already_seen_words:
                    continue
                already_seen_words.append(word)

                if word not in cluster_profile:
                    cluster_profile[word] = 0
                cluster_profile[word] += score

        # sort the words by the aggregate delta
        tuples = sorted(
            zip(cluster_profile.keys(), cluster_profile.values()),
            key=lambda x: x[1],
            reverse=True,
        )
        cluster_profiles[cluster] = tuples
    return cluster_profiles


def frequent_words_in_cluster(
    df: pd.DataFrame,
    cluster_indices: List[int],
    input_col_name: str,
) -> List[Tuple[str, int]]:
    """Finds the most frequently occurring words for each cluster given.

    :param df: The dataframe containing the test data.
    :param cluster_indices: The list of indices of the points in the desired cluster.
    :param input_col_name: The name of the column in :code:`df` that contains the input text.

    :return: A list of tuples, each tuple containing the word and the number of times
        it appears in that cluster.
    """
    counter = CountVectorizer(stop_words=stopwords.words("english"))
    cv_fit = counter.fit_transform(df.iloc[cluster_indices][input_col_name].values)
    freq_words = sorted(
        zip(counter.get_feature_names(), cv_fit.toarray().sum(axis=0)),
        key=lambda x: x[1],
        reverse=True,
    )

    return freq_words


def frequent_words_by_class_in_cluster(
    df: pd.DataFrame,
    freq_words: List[Tuple[str, int]],
    encodings: Dict[str, int],
    cluster_indices: List[int],
    input_col_name: str,
    classification_col_name: str,
) -> Dict[str, Dict[Any, int]]:
    """Takes the frequent words of a cluster and splits the counts up based on the
    classification of the entry they fall under. (This gives a rough distribution for
    what category the words fall under within the cluster.)

    :param df: The dataframe containing the test data
    :param freq_words: An array of tuples of the words and their number of occurences,
        see :meth:`tx2.calc.frequent_words_in_cluster`
    :param encodings: The dictionary of class/category encodings.
    :param cluster_indices: The list of indices of the points in the desired cluster.
    :param classification_col_name: The name of the column in :code:`df` that contains
        the target class value.
    :param input_col_name: The name of the column in :code:`df` that contains the
        input text.

    :return: A dictionary with each word as the key. The value for each is a dictionary
        with a "total" key and a key for each encoded class, the value of which is the
        number of entries with that class containing the word.
    """
    vocab = [pair[0] for pair in freq_words]

    # grab the total count for each word and put it in the new dictionary
    word_dict = {}
    for pair in freq_words:
        if pair[0] in vocab:
            word_dict[pair[0]] = {"total": pair[1]}

    working_df = df.loc[cluster_indices]

    # iterate through each classification and get the number of entries with that word in it
    for classification in encodings.values():
        local_df = working_df[working_df[classification_col_name] == classification]
        counter = CountVectorizer(
            stop_words=stopwords.words("english"), vocabulary=vocab
        )
        cv_fit = counter.fit_transform(local_df[input_col_name].values)
        class_freq_words = list(
            zip(counter.get_feature_names(), cv_fit.toarray().sum(axis=0))
        )
        for pair in class_freq_words:
            word_dict[pair[0]][str(classification)] = pair[1]
    return word_dict


def salience_map(
    soft_classify_function, text: str, encodings: Dict[str, int], length: int = 256
) -> List[Tuple[str, np.ndarray, np.ndarray, str]]:
    """Calculates the total change in output classification probabilities when each
    individual word in the text is removed, a proxy for each word's "importance" in
    the model prediction.

    :param soft_classify_function: A function that takes as input an array of texts
        and returns an array of output values for each category.
    :param text: The text to compute the salience map for.
    :param encodings: The dictionary of class/category encodings.
    :param length: The maximum number of words to stop at. Since transformer tokens
        are unlikely to always be full words, this won't directly correspond to what
        the model actually uses, but it's to help at least marginally cut down on
        processing time. (Running this function on each text in an entire data frame
        can take a while.)

    :return: An array of tuples. Each tuple corresponds to that word being removed,
        and includes the word that was removed, the output prediction values, the diff
        between the unaltered text and altered text output prediction values, and the
        new predicted category with the text removed: :code:`("WORD", PRED_VALUES,
        PRED_VALUES - ORIGINAL_PRED_VALUES, PRED_CATEGORY)`. Note that the first entry
        in the map is the outputs for the original text.
    """
    # TODO: this function could probably be optimized - not utilizing batching capabilities of soft_classify_function
    # (pass in entire array of text variants, and recombine results afterwards)
    # also note that technically it's removing every instance of a word, not just the current indexed one
    words = text.split(" ")

    # get the output from an unaltered version of the text
    original_scores = np.array(soft_classify_function([text])[0])
    original_cat = utils.get_pred_cat(original_scores, encodings)

    deltas = [("", original_scores, original_scores - original_scores, original_cat)]

    # iteratively test remove each word individually and record the results in `deltas`
    for index, word in enumerate(words):
        if index > length:
            break
        # current_text = text.replace(word, "")
        current_text = " ".join(words[:index] + words[index + 1 :])
        new_scores = np.array(soft_classify_function([current_text])[0])
        new_cat = utils.get_pred_cat(new_scores, encodings)

        deltas.append((word, new_scores, new_scores - original_scores, new_cat))

    return deltas


def sort_salience_map(salience) -> List[Tuple[str, float]]:
    """Sort the passed salience map tuples by computing a total "delta" for each word,
    computed from the sum of absolute values for each predicted value.

    :param salience: A salience map as returned from :meth:`tx2.calc.salience_map`.

    :return: A new list of sorted tuples, where each tuple consists of :code:`("WORD", TOTAL_DELTA)`.
    """
    diffs = []

    for entry in salience:
        diff = abs(entry[2][0]).sum()
        diffs.append((entry[0], diff))

    diffs = sorted(diffs, key=lambda x: x[1], reverse=True)
    return diffs


def normalize_salience_map(salience) -> Dict[str, float]:
    """Get a salience map with the total scores as computed in :meth:`tx2.calc.sort_salience_map`
    and normalize those scores.

    :param salience: A salience map as returned from :meth:`tx2.calc.salience_map`.

    :return: A dictionary with each word removed as a key, and each value the normalized diff
        caused by removing that word.
    """
    changes = sort_salience_map(salience)

    # convert tuples into dictionary
    change_dict = {}
    for entry in changes:
        if entry[0] not in change_dict:
            change_dict[entry[0]] = entry[1]

    # normalize each total value
    factor = 1.0 / max(change_dict.values())
    normalised_d = {k: v * factor for k, v in change_dict.items()}

    return normalised_d


def prediction_scores(
    df: pd.DataFrame, target_col_name: str, predicted_col_name: str, encodings
):
    targets = df[target_col_name]
    preds = df[predicted_col_name]

    per_class_stats = {}

    for key, value in encodings.items():
        tp = df[(targets == value) & (preds == value)].shape[0]
        fp = df[(targets != value) & (preds == value)].shape[0]
        fn = df[(targets == value) & (preds != value)].shape[0]
        tn = df[(targets != value) & (preds != value)].shape[0]
        accuracy = (tp + tn) / (tp + tn + fn + fp)
        if tp == 0 and fp == 0:
            precision = 1.0  # ?
        else:
            precision = tp / (tp + fp)
        if tp == 0 and fn == 0:
            recall = 1.0  # ?
        else:
            recall = tp / (tp + fn)
        if (precision + recall) > 0:
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        per_class_stats[value] = {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    # macro avg scores
    macros = {}
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    for key, value in encodings.items():
        accuracy += per_class_stats[value]["accuracy"]
        precision += per_class_stats[value]["precision"]
        recall += per_class_stats[value]["recall"]
    accuracy /= len(encodings)
    precision /= len(encodings)
    recall /= len(encodings)
    f1 = (2 * precision * recall) / (precision + recall)

    macros["accuracy"] = accuracy
    macros["precision"] = precision
    macros["recall"] = recall
    macros["f1"] = f1

    tp += per_class_stats[value]["tp"]
    # micro avg score
    micros = {}
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for key, value in encodings.items():
        tp += per_class_stats[value]["tp"]
        fp += per_class_stats[value]["fp"]
        tn += per_class_stats[value]["tn"]
        fn += per_class_stats[value]["fn"]

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)
    micros["precision"] = precision
    micros["recall"] = recall
    micros["f1"] = f1

    return per_class_stats, macros, micros
