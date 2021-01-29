from setuptools import setup

setup(
    name="tx2",
    version="2021.1",
    description="Transformer eXplainability and eXploration",
    long_description="...",
    author="Nathan Martindale",
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
