from setuptools import setup

setup(
    name="tx2",
    version="2021.5",
    description="Transformer eXplainability and eXploration",
    long_description="This library is intended to aid in the explorability and explainability of transformer classification networks, or transformer language models with sequence classification heads. The basic function of this library is to take a trained transformer and test/train dataset and produce an ipywidget dashboard as seen in the screenshot below, which can be displayed in a jupyter notebook or jupyter lab.",
    author="Nathan Martindale, Scott L. Stewart",
    author_email="tx2-help@ornl.gov",
    packages=["tx2"],
    install_requires=[
        "sklearn",
        "nltk",
        "pandas",
        "numpy",
        "torch",
        "tqdm",
        "umap-learn",
        "matplotlib",
        "wordcloud",
        "ipywidgets",
    ],
)
