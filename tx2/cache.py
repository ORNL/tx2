""" Functions to cut down on boilerplate accessing cached objects """

import os
import pickle
import json
import logging
import numpy as np


def check(filepath: str, overwrite: bool) -> bool:
    """ Returns true if we have a cached version that we should load """
    if os.path.exists(filepath) and not overwrite:
        logging.info("cached version '%s' found", filepath)
        return True
    return False


def read(filepath: str):
    """ Wrapper for other readers for the super lazy """
    if is_json(filepath):
        return read_json(filepath)
    elif is_pickle(filepath):
        return read_pickle(filepath)


def write(obj, filepath: str):
    logging.info("Writing to %s", filepath)
    if is_json(filepath):
        write_json(obj, filepath)
    elif is_pickle(filepath):
        write_pickle(obj, filepath)


def is_json(filepath: str) -> bool:
    """ Is this a json file? """
    return filepath[-5:] == ".json"


def is_pickle(filepath: str) -> bool:
    """ Is this a pickle file? """
    return filepath[-7:] == ".pkl.gz"


def read_pickle(filepath: str):
    """ Reads a pickle file and returns it """
    with open(filepath, "rb") as infile:
        obj = pickle.load(infile)
    return obj


def write_pickle(obj, filepath: str):
    with open(filepath, "wb") as outfile:
        pickle.dump(obj, outfile)


def read_json(filepath: str):
    with open(filepath, "r") as infile:
        obj = json.load(infile)
    return obj


def write_json(obj, filepath: str):
    with open(filepath, "w") as outfile:
        json.dump(obj, outfile, indent=4, default=lambda x: str(x))
