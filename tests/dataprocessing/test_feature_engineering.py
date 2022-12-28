from cdtools.dataprocessing.feature_engineering import get_feature_lists
import unittest


def test_get_feature_lists():

    column_data_types = {
        "key_1": str,
        "key_2": str,
        "feature_1": str,
        "feature_2": int,
        "feature_3": float,
        "feature_4": bool,
        "label_1": bool,
        "label_2": float,
    }

    keys = ["key_1", "key_2"]
    labels = ["label_1", "label_2"]

    features_expected = ["feature_1", "feature_2", "feature_3", "feature_4"]
    features_categorical_expected = ["feature_1"]
    features_numerical_expected = ["feature_2", "feature_3"]
    features_boolean_expected = ["feature_4"]

    (
        features_actual,
        features_categorical_actual,
        features_numerical_actual,
        features_boolean_actual,
    ) = get_feature_lists(column_data_types, keys, labels)

    unittest.TestCase().assertListEqual(features_expected, features_actual)
    unittest.TestCase().assertListEqual(
        features_categorical_expected, features_categorical_actual
    )
    unittest.TestCase().assertListEqual(
        features_numerical_expected, features_numerical_actual
    )
    unittest.TestCase().assertListEqual(
        features_boolean_expected, features_boolean_actual
    )
