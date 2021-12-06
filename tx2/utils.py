import asyncio
import numpy as np
from threading import Timer
from torch import cuda


DISABLE_DEBOUNCE = False
"""The main embedding plot is debounced to prevent excessive calling while editing text. Note that exceptions don't propagate due to threading when this occurs. Set this global to True to receive exceptions."""


def get_device() -> str:
    """Determine the device to put pytorch tensors on

    :return: "cuda" or "cpu"
    """
    device = "cuda" if cuda.is_available() else "cpu"
    return device


# used in visualization
def which_cluster(index, clusters):
    """TODO Get cluster label from index of point in one of the clusters"""
    for cluster in clusters:
        if index in clusters[cluster]:
            return cluster
    return None


# used in wrapper
def set_defaults(params, **defaults):
    """TODO Set given parameters in given set of arguments if they do not
    already have values.
    """

    for key in defaults:
        if key not in params:
            params[key] = defaults[key]

    return params


def array_elipsis(array, length, depth=1):
    string = "["
    for index, element in enumerate(array[:length]):
        if depth == 1:
            string += str(element)
        else:
            string += array_elipsis(array[index], length, depth - 1)

        if index < len(array) - 1:
            string += ", "
        if index == length - 1 and index < len(array) - 1:
            string += "..."
        if index >= length - 1:
            break
    string += "]"
    return string


# https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20Events.html#Debouncing
class Timer:
    # Used in debounce
    def __init__(self, timeout, callback):
        self._timeout = timeout
        self._callback = callback

    async def _job(self):
        await asyncio.sleep(self._timeout)
        self._callback()

    def start(self):
        if DISABLE_DEBOUNCE:
            self._callback()
        else:
            self._task = asyncio.ensure_future(self._job())

    def cancel(self):
        if not DISABLE_DEBOUNCE:
            self._task.cancel()


def debounce(wait):
    """TODO Decorator that will postpone a function's
    execution until after `wait` seconds
    have elapsed since the last time it was invoked."""

    def decorator(fn):
        timer = None

        def debounced(*args, **kwargs):
            nonlocal timer

            def call_it():
                fn(*args, **kwargs)

            if timer is not None and not DISABLE_DEBOUNCE:
                timer.cancel()
            timer = Timer(wait, call_it)
            timer.start()

        return debounced

    return decorator


def get_cat_by_index(idx, encodings):
    """Return the name of the category for the given encoded index"""
    for key in encodings.keys():
        if encodings[key] == idx:
            return key


def get_pred_cat(pred, encodings):
    """Determine which category is predicted based on passed model output"""
    idx = np.argmax(pred)
    return get_cat_by_index(idx, encodings)
