# https://www.kaggle.com/code/metric/open-polymer-2025

from typing import List

import numpy as np
import pandas as pd


class ParticipantVisibleError(Exception):
    pass


# These values are from the train data.
MINMAX_DICT =  {
        'Tg': [-139.000000, 495.000000],
        'FFV': [ 0.226992, 0.777097],
        'Tc': [0.082201, 0.618760],
        'Density': [0.742359, 1.914482],
        'Rg': [9.728355, 85.684019],
    }
NULL_FOR_SUBMISSION = -9999


def normalize_property_weight(property_weight: List[int]) -> np.ndarray:
    property_weight = np.array(property_weight)
    property_weight = np.sqrt(1 / property_weight)
    property_weight = (property_weight / np.sum(property_weight)) * len(property_weight)
    return property_weight

def scaling_error(labels, preds, property):
    error = np.abs(labels - preds)
    min_val, max_val = MINMAX_DICT[property]
    label_range = max_val - min_val
    return np.mean(error / label_range)


def get_property_weights(labels):
    property_weight = []
    for property in MINMAX_DICT.keys():
        valid_num = np.sum(labels[property] != NULL_FOR_SUBMISSION)
        property_weight.append(valid_num)
    return normalize_property_weight(property_weight)


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Compute weighted Mean Absolute Error (wMAE) for the Open Polymer challenge.

    Expected input:
      - solution and submission as pandas.DataFrame
      - Column 'id': unique identifier for each sequence
      - Columns 'Tg', 'FFV', 'Tc', 'Density', 'Rg' as the predicted targets

    Examples
    --------
    >>> import pandas as pd
    >>> row_id_column_name = "id"
    >>> solution = pd.DataFrame({'id': range(4), 'Tg': [0.2]*4, 'FFV': [0.2]*4, 'Tc': [0.2]*4, 'Density': [0.2]*4, 'Rg': [0.2]*4})
    >>> submission = pd.DataFrame({'id': range(4), 'Tg': [0.5]*4, 'FFV': [0.5]*4, 'Tc': [0.5]*4, 'Density': [0.5]*4, 'Rg': [0.5]*4})
    >>> round(score(solution, submission, row_id_column_name=row_id_column_name), 4)
    0.2922
    >>> submission = pd.DataFrame({'id': range(4), 'Tg': [0.2]*4, 'FFV': [0.2]*4, 'Tc': [0.2]*4, 'Density': [0.2]*4, 'Rg': [0.2]*4} )
    >>> score(solution, submission, row_id_column_name=row_id_column_name)
    0.0
    """
    chemical_properties = list(MINMAX_DICT.keys())
    property_maes = []
    property_weights = get_property_weights(solution[chemical_properties])
    for property in chemical_properties:
        is_labeled = solution[property] != NULL_FOR_SUBMISSION
        property_maes.append(scaling_error(solution.loc[is_labeled, property], submission.loc[is_labeled, property], property))

    if len(property_maes) == 0:
        raise RuntimeError('No labels')
    return float(np.average(property_maes, weights=property_weights))