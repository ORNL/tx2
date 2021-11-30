import numpy as np
import os
import pandas as pd
import pytest
import torch

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


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
            self.vectorizer = CountVectorizer(stop_words='english')
            
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
    

@pytest.fixture(scope='session')
def clear_files_teardown():
    yield None
    os.system("rm -rf data/")
