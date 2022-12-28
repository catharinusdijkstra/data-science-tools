def get_feature_lists(column_data_types, keys, labels):

    features = [
        column for column in column_data_types.keys() if (column not in keys + labels)
    ]

    features_categorical = [
        column
        for (column, column_data_type) in column_data_types.items()
        if ((column not in keys + labels) and (column_data_type == str))
    ]

    features_numeric = [
        column
        for (column, column_data_type) in column_data_types.items()
        if ((column not in keys + labels) and (column_data_type in [int, float]))
    ]

    features_boolean = [
        column
        for (column, column_data_type) in column_data_types.items()
        if ((column not in keys + labels) and (column_data_type in [bool]))
    ]

    return features, features_categorical, features_numeric, features_boolean
