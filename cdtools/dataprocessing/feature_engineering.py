from typing import Tuple


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
