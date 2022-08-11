import numpy as np
import os
import pandas as pd
import pytest
import torch

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from tx2.wrapper import Wrapper


@pytest.fixture
def replacement_debounce():
    """This is so that we can replace utils.debounce and actually get
    error messages."""

    def undebounce(wait):
        def decorator(fn):
            def undebounced(*args, **kwargs):
                print("WE'RE DOING IT LIVE")
                fn(*args, **kwargs)

            return undebounced

        return decorator

    return undebounce


@pytest.fixture
def dummy_df():
    rows = [
        {"text": "testing row 0", "target": 0},
        {"text": "testing row 1", "target": 0},
        {"text": "testing row 2, awesome", "target": 1},
        {"text": "testing row 3, awesome", "target": 1},
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def dummy_empty_df():
    rows = [
        {"text": " ", "target": 0},
        {"text": " ", "target": 0},
        {"text": " ", "target": 1},
        {"text": " ", "target": 1},
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def dummy_np_data(dummy_df):
    texts = np.array(dummy_df.text)
    targets = np.array(dummy_df.target)

    return texts, targets


@pytest.fixture
def dummy_clusters():
    return {"0": [0, 1], "1": [2, 3]}


@pytest.fixture
def dummy_embeddings():
    return (
        np.array([[0.1, 0.2], [-0.1, -0.1], [0.5, -0.8], [0.4, -0.9]]),
        np.array([0.1, -0.1, 0.5, 0.4]),
        np.array([0.2, -0.1, -0.8, -0.9]),
    )


@pytest.fixture
def dummy_encodings():
    return {"awesome": 0, "not-awesome": 1}


@pytest.fixture
def dummy_model(dummy_df):
    class model:
        def __init__(self, df):
            self.vectorizer = CountVectorizer(stop_words="english")

            x = self.vectorizer.fit_transform(df.text)

            self.clf = LogisticRegression()
            self.clf.fit(x, df.target)

        def custom_encode(self, text):
            transformed = torch.tensor(self.vectorizer.transform([text]).toarray())
            return torch.squeeze(transformed)

        def custom_classify(self, inputs):
            return torch.tensor(self.clf.predict(inputs))

        def custom_embedding(self, inputs):
            return inputs

        def custom_softclassify(self, inputs):
            return torch.tensor(self.clf.predict_proba(inputs))

    return model(dummy_df)


@pytest.fixture(scope="function")
def dummy_wrapper(dummy_df, dummy_encodings, dummy_model, clear_files_teardown):
    wrapper = Wrapper(
        train_texts=dummy_df.text,
        train_labels=dummy_df.target,
        test_texts=dummy_df.text,
        test_labels=dummy_df.target,
        encodings=dummy_encodings,
        cache_path="testdata",
    )

    wrapper.encode_function = dummy_model.custom_encode
    wrapper.classification_function = dummy_model.custom_classify
    wrapper.embedding_function = dummy_model.custom_embedding
    wrapper.soft_classification_function = dummy_model.custom_softclassify

    wrapper.prepare(umap_args=dict(n_neighbors=2))

    return wrapper


@pytest.fixture(scope="session")
def clear_files_teardown():
    yield None
    os.system("rm -rf testdata/")
    os.system("rm -rf testdata2/")
