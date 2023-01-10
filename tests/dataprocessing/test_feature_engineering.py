from cdtools.dataprocessing.feature_engineering import encode_labels, get_feature_lists
import numpy as np
import unittest


def test_encode_labels():

    column_1 = np.array(["Male", "Female"])
    column_2 = [True, False]
    column_3 = ["Noot", "Mies", "Aap"]

    label_encoded_column_1_expected = [1, 0]
    label_encoded_classes_1_expected = ["Female", "Male"]
    label_encoded_column_2_expected = [1, 0]
    label_encoded_classes_2_expected = [False, True]
    label_encoded_column_3_expected = [2, 1, 0]
    label_encoded_classes_3_expected = ["Aap", "Mies", "Noot"]

    label_encoded_column_1_actual, label_encoded_classes_1_actual = encode_labels(
        column_1
    )
    label_encoded_column_2_actual, label_encoded_classes_2_actual = encode_labels(
        column_2
    )
    label_encoded_column_3_actual, label_encoded_classes_3_actual = encode_labels(
        column_3
    )

    np.testing.assert_array_equal(
        label_encoded_column_1_expected, label_encoded_column_1_actual
    )
    np.testing.assert_array_equal(
        label_encoded_classes_1_expected, label_encoded_classes_1_actual
    )
    np.testing.assert_array_equal(
        label_encoded_column_2_expected, label_encoded_column_2_actual
    )
    np.testing.assert_array_equal(
        label_encoded_classes_2_expected, label_encoded_classes_2_actual
    )
    np.testing.assert_array_equal(
        label_encoded_column_3_expected, label_encoded_column_3_actual
    )
    np.testing.assert_array_equal(
        label_encoded_classes_3_expected, label_encoded_classes_3_actual
    )


def test_get_feature_lists():

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

    features_expected = [
        "feature_1",
        "feature_2",
        "feature_3",
        "feature_4",
        "feature_5",
    ]
    features_categorical_expected = ["feature_1"]
    features_numerical_expected = ["feature_2", "feature_3", "feature_4"]
    features_boolean_expected = ["feature_5"]

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
