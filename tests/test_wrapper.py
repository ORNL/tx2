import numpy as np

from tx2.wrapper import Wrapper


# TODO: write support for numpy arrays (should take train_texts, train_y, test_texts, test_y)
# NOTE: it may actually be better to _only_ support numpy arrays. It's trivial to get those arrays from the dataframe, and it's
#   the same number of arguments. This also simplifies some of the function calls in the visualization module.


def test_wrapper_init_no_crash(dummy_df, dummy_encodings):
    wrapper = Wrapper(
        train_texts=dummy_df.text,
        train_labels=dummy_df.target,
        test_texts=dummy_df.text,
        test_labels=dummy_df.target,
        encodings=dummy_encodings,
    )


def test_wrapper_init_np_no_crash(dummy_np_data, dummy_encodings):
    wrapper = Wrapper(
        train_texts=dummy_np_data[0],
        train_labels=dummy_np_data[1],
        test_texts=dummy_np_data[0],
        test_labels=dummy_np_data[1],
        encodings=dummy_encodings,
    )


# TODO: tests to ensure exceptions are thrown in prepare if incorrect combination of things have not been specified.


def test_wrapper_prepare_no_crash(dummy_df, dummy_encodings, dummy_model, clear_files_teardown):
    wrapper = Wrapper(
        train_texts=dummy_df.text,
        train_labels=dummy_df.target,
        test_texts=dummy_df.text,
        test_labels=dummy_df.target,
        encodings=dummy_encodings,
        cache_path="testdata"
    )
    
    wrapper.encode_function = dummy_model.custom_encode
    wrapper.classification_function = dummy_model.custom_classify
    wrapper.embedding_function = dummy_model.custom_embedding
    wrapper.soft_classification_function = dummy_model.custom_softclassify
    
    wrapper.prepare(umap_args=dict(n_neighbors=2))

