from tx2.calc import frequent_words_in_cluster, frequent_words_by_class_in_cluster


def test_frequent_words_in_cluster(dummy_df):
    freq_words = frequent_words_in_cluster(dummy_df.text)
    assert freq_words == [("row", 4), ("testing", 4), ("awesome", 2)]


def test_frequent_words_by_class_in_cluster(dummy_df, dummy_encodings, dummy_clusters):
    cluster_text = dummy_df.text[dummy_clusters["0"]]
    cluster_text_labels = dummy_df.target[dummy_clusters["0"]]

    freq_words = frequent_words_in_cluster(cluster_text)
    freq_words_by_class = frequent_words_by_class_in_cluster(
        freq_words, dummy_encodings, cluster_text, cluster_text_labels
    )

    expected_output = {
        "testing": {"total": 2, "0": 2, "1": 0},
        "row": {"total": 2, "0": 2, "1": 0},
    }

    assert freq_words_by_class == expected_output
