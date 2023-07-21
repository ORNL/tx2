from setuptools import setup
import re


def get_property(prop):
    result = re.search(
        r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop), open("tx2/__init__.py").read()
    )
    return result.group(1)


with open("README.md", "r", encoding="utf-8") as infile:
    long_description = infile.read()

setup(
    name="tx2",
    version=get_property("__version__"),
    description="Transformer eXplainability and eXploration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Nathan Martindale, Scott L. Stewart",
    author_email="tx2-help@ornl.gov",
    url="https://github.com/ORNL/tx2",
    project_urls={"Documentation": "https://ornl.github.io/tx2/"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires="~=3.9",
    packages=["tx2"],
    install_requires=[
        "scikit-learn",
        "pandas",
        "numpy",
        "torch",
        "tqdm",
        "umap-learn",
        "matplotlib",
        "ipywidgets",
        "wordcloud",
    ],
)
