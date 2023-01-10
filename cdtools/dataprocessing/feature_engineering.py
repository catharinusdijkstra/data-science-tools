import numpy as np
import pandas as pd
from sklearn import preprocessing
from typing import Tuple, Union


def encode_labels(
    column: Union[list, np.ndarray, pd.Series],
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Convert labels/words in a given column into numeric form.

    Inputs:
    column: list, np.ndarray, pd.Series
        Column with data to be labeled.

    Outputs:
    label_encoded_column: np.ndarray
        Numpy array with all the labels/words in the original column into numeric form.
    label_encoded_classes: np.ndarray
        Numpy array with the distinct labels/words in the original column in
        alphabetical order.

    Example:
    Given three columns, column_1, column_2, and column_3, convert the labels/words of
    these columns into numeric form:

    column_1 = np.array(["Male", "Female"])
    column_2 = [True, False]
    column_3 = ["Noot", "Mies", "Aap"]

    label_encoded_column_1, label_encoded_classes_1 = encode_labels(column_1)
    label_encoded_column_2, label_encoded_classes_2 = encode_labels(column_2)
    label_encoded_column_3, label_encoded_classes_3 = encode_labels(column_3)

    The above example generates the following output:

    label_encoded_column_1 = [1, 0]
    label_encoded_classes_1 = ["Female", "Male"]
    label_encoded_column_2 = [1, 0]
    label_encoded_classes_2 = [False, True]
    label_encoded_column_3 = [2, 1, 0]
    label_encoded_classes_3 = ["Aap", "Mies", "Noot"]

    Note that words are labeled (numbered) in alphabetical order of these words. The
    boolean values True and False are labeled as 1 and 0 respectively.
    """

    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(column)
    label_encoded_column = label_encoder.transform(column)
    label_encoded_classes = label_encoder.classes_

    return label_encoded_column, label_encoded_classes


def get_feature_lists(
    column_data_types: dict, keys: list, labels: list
) -> Tuple[list, list, list, list]:

    """
    Given a dictionary with column data types, a list of key columns, and a list of
    label columns, get lists of features from the dictionary with column data types.

    Inputs:
    column_data_types: dict
        A dictionary with column data types.
    keys: list
        A list of key columns, where the combination of keys uniquely identifies a row
        of data to which the dictionary with column data types is applicable.
    labels: list
        A list of label columns, where the labels are the target columns to be predicted
        by a machine learning model.

    Outputs:
    features: list
        A list with all the features from the dictionary with column data types.
    features_categorical: list
        A list with all the categorical features from the dictionary with column data
        types.
    features_numerical: list
        A list with all the numerical features from the dictionary with column data
        types.
    features_boolean: list
        A list with all the boolean features from the dictionary with column data types.

    Example:
    Given a dictionary with column data types, column_data_types, a list of key columns,
    keys, and a list of label columns, labels, get lists of features from the dictionary
    with column data types:

        column_data_types = {
            "key_1": str,
            "key_2": str,
            "feature_1": str,
            "feature_2": int,
            "feature_3": float,
            "feature_4": complex,
            "feature_5": bool,
            "label_1": bool,
            "label_2": float,
        }

        keys = ["key_1", "key_2"]
        labels = ["label_1", "label_2"]

        (
            features,
            features_categorical,
            features_numerical,
            features_boolean,
        ) = get_feature_lists(column_data_types, keys, labels)

    The above example generates the following output:

    features = [
        "feature_1",
        "feature_2",
        "feature_3",
        "feature_4",
        "feature_5"
    ]
    features_categorical = ["feature_1"]
    features_numerical = ["feature_2", "feature_3", "feature_4"]
    features_boolean = ["feature_5"]
    """

    features = [
        column for column in column_data_types.keys() if (column not in keys + labels)
    ]

    features_categorical = [
        column
        for (column, column_data_type) in column_data_types.items()
        if ((column not in keys + labels) and (column_data_type == str))
    ]

    features_numerical = [
        column
        for (column, column_data_type) in column_data_types.items()
        if (
            (column not in keys + labels)
            and (column_data_type in [int, float, complex])
        )
    ]

    features_boolean = [
        column
        for (column, column_data_type) in column_data_types.items()
        if ((column not in keys + labels) and (column_data_type in [bool]))
    ]

    return (features, features_categorical, features_numerical, features_boolean)
