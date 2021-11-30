import numpy as np
from pytest_mock import mocker
from tx2.visualization import (_get_scatter_points_from_embeddings,
                               gen_wordcloud, prepare_wordclouds, plot_wordclouds, plot_big_wordcloud, plot_passed_wordcloud)
from tx2.dashboard import Dashboard


def test_gen_wordcloud_no_crash(dummy_df):
    gen_wordcloud(dummy_df.text)


def test_gen_wordcloud_np_no_crash(dummy_np_data):
    gen_wordcloud(dummy_np_data[0])


def test_prepare_wordclouds_no_crash(dummy_df, dummy_clusters):
    prepare_wordclouds(dummy_clusters, dummy_df.text)


def test_prepare_wordclouds_np_no_crash(dummy_np_data, dummy_clusters):
    prepare_wordclouds(dummy_clusters, dummy_np_data[0])


def test_prepare_wordclouds_calls_gen_correctly(mocker, dummy_df, dummy_clusters):
    mock = mocker.patch("tx2.visualization.gen_wordcloud")
    prepare_wordclouds(dummy_clusters, dummy_df.text)

    for i, call in enumerate(mock.call_args_list):
        expected_cluster_size = len(dummy_clusters[list(dummy_clusters.keys())[i]])
        size_of_texts_passed = call.args[0].shape[0]

        assert size_of_texts_passed == expected_cluster_size


def test_get_scatterpoints_w_labels(dummy_df, dummy_embeddings):
    x, y, colors = _get_scatter_points_from_embeddings(
        [0, 1], dummy_embeddings[0], dummy_df.target
    )
    assert (x == dummy_embeddings[1]).all()
    assert (y == dummy_embeddings[2]).all()
    assert (colors == dummy_df.target).all()


def test_get_scatterpoints_w_labels_np(dummy_np_data, dummy_embeddings):
    x, y, colors = _get_scatter_points_from_embeddings(
        [0, 1], dummy_embeddings[0], dummy_np_data[1]
    )
    assert (x == dummy_embeddings[1]).all()
    assert (y == dummy_embeddings[2]).all()
    assert (colors == dummy_np_data[1]).all()


def test_get_scatterpoints_wo_labels(dummy_df, dummy_embeddings):
    x, y, colors = _get_scatter_points_from_embeddings([0, 1], dummy_embeddings[0])
    assert (x == dummy_embeddings[1]).all()
    assert (y == dummy_embeddings[2]).all()
    assert (colors == np.zeros([len(dummy_df)])).all()


def test_plot_big_wordcloud_no_crash(dummy_df, dummy_clusters):
    prepare_wordclouds(dummy_clusters, dummy_df.text)
    plot_big_wordcloud(0, dummy_clusters)


def test_plot_passed_wordcloud_no_crash(dummy_df):
    cloud = gen_wordcloud(dummy_df.text)
    plot_passed_wordcloud(cloud, "test")


def test_plot_wordclouds_no_crash(dummy_df, dummy_clusters):
    dashboard = type('Dashboard', (object,), {})()
    dashboard.transformer_wrapper = type('Wrapper', (object,), {})()
    dashboard.transformer_wrapper.clusters = dummy_clusters
    # forgive me for I have sinned...(I couldn't figure out how to do the equivalent with mocker)

    prepare_wordclouds(dummy_clusters, dummy_df.text)
    plot_wordclouds(dashboard)
