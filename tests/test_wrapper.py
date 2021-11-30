import numpy as np

from tx2.wrapper import Wrapper




# TODO: write support for numpy arrays (should take train_texts, train_y, test_texts, test_y)
# NOTE: it may actually be better to _only_ support numpy arrays. It's trivial to get those arrays from the dataframe, and it's
#   the same number of arguments. This also simplifies some of the function calls in the visualization module.


def test_wrapper_init_no_crash(dummy_df):
    wrapper = Wrapper(
        train_df=dummy_df,
        test_df=dummy_df,
        encodings={},
        input_col_name="text",
        target_col_name="target",
    )


def test_wrapper_init_numpy(dummy_df):
    train_x = np.array(dummy_df.text)
    train_y = np.array(dummy_df.target)
    test_x = np.array(dummy_df.text)
    test_y = np.array(dummy_df.target)

    wrapper = Wrapper(
        train_texts=train_x,
        train_targets=train_y,
        test_texts=test_x,
        test_targets=test_y,
    )
