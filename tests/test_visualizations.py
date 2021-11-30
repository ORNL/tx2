import numpy as np
from pytest_mock import mocker

from tx2.visualization import gen_wordcloud, prepare_wordclouds, _get_scatter_points_from_embeddings

def test_gen_wordcloud_no_crash(dummy_df):
    gen_wordcloud(dummy_df.text)


def test_prepare_wordclouds_no_crash(dummy_df, dummy_clusters):
    prepare_wordclouds(dummy_clusters, dummy_df.text)
    
    
def test_prepare_wordclouds_calls_gen_correctly(mocker, dummy_df, dummy_clusters):
    mock = mocker.patch('tx2.visualization.gen_wordcloud')
    prepare_wordclouds(dummy_clusters, dummy_df.text)

    for i, call in enumerate(mock.call_args_list):
        expected_cluster_size = len(dummy_clusters[list(dummy_clusters.keys())[i]])
        size_of_texts_passed = call.args[0].shape[0]
        
        assert size_of_texts_passed == expected_cluster_size


def test_get_scatterpoints_w_labels(dummy_df, dummy_embeddings):
    x, y, colors = _get_scatter_points_from_embeddings([0, 1], dummy_embeddings[0], dummy_df.target)
    assert (x == dummy_embeddings[1]).all()
    assert (y == dummy_embeddings[2]).all()
    assert (colors == dummy_df.target).all()
       

def test_get_scatterpoints_wo_labels(mocker, dummy_df, dummy_embeddings):
    x, y, colors = _get_scatter_points_from_embeddings([0, 1], dummy_embeddings[0])
    assert (x == dummy_embeddings[1]).all()
    assert (y == dummy_embeddings[2]).all()
    assert (colors == np.zeros([len(dummy_df)])).all()
