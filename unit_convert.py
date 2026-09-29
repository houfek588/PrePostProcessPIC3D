"""
unit_convert.py - Pomocné funkce pro škálování numerických dat a seznamů.
"""

import numpy as np


def rescale_list(input_list, scale):
    """
    Přenásobí prvky pole nebo seznamu daným měřítkem.
    Podporuje jak standardní Python seznamy, tak NumPy pole.
    """
    if isinstance(input_list, np.ndarray):
        return input_list * scale
    if hasattr(input_list, "__iter__") and not isinstance(input_list, (str, bytes)):
        return [n * scale for n in input_list]
    return input_list * scale


def rescale_list_of_lists(lst, multiplier):
    """
    Přenásobí každý prvek vnořeného pole nebo seznamu seznamů daným násobitelem.
    """
    if isinstance(lst, np.ndarray):
        return lst * multiplier
    return [[element * multiplier for element in sublist] for sublist in lst]