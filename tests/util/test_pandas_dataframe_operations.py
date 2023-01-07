from cdtools.util import pandas_dataframe_operations
import numpy as np
import pandas as pd


def test_impute_dataframe():

    data = [
        ["alice", 20],
        ["alice", 25],
        ["alice", None],
        ["bob", 30],
        ["bob", 21],
        [None, 13],
    ]

    df = pd.DataFrame(data, columns=["Name", "Age"])

    imputed_data_expected = [
        ["alice", 20],
        ["alice", 25],
        ["alice", 30],
        ["bob", 30],
        ["bob", 21],
        ["alice", 13],
    ]

    df_imputed_expected = pd.DataFrame(imputed_data_expected, columns=["Name", "Age"])
    df_imputed_expected["Age"] = df_imputed_expected["Age"].astype(np.float64)

    df_imputed_actual = pandas_dataframe_operations.impute_dataframe(df, random_state=0)

    pd.testing.assert_frame_equal(df_imputed_expected, df_imputed_actual)
