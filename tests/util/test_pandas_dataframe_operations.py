from cdtools.util import pandas_dataframe_operations
from datetime import datetime
import numpy as np
import pandas as pd


def test_compare_dataframes():

    data1 = [
        ["alice", 20, "09/19/22 13:55:26", True],
        ["alice", 25, "09/19/22 13:55:26", False],
        ["alice", None, "09/19/22 13:55:26", True],
        ["bob", 30, "09/19/22 13:55:26", True],
        ["bob", 21, "09/19/22 13:55:26", None],
        [None, 13, "09/19/22 13:55:26", True],
        ["daisy", 27, "09/19/22 13:55:26", True],
    ]

    df1 = pd.DataFrame(data1, columns=["Name", "Age", "Datetime", "Active"])
    df1["Datetime"] = df1["Datetime"].apply(
        lambda x: datetime.strptime(x, "%m/%d/%y %H:%M:%S")
    )

    data2 = [
        ["alice", 21, "09/19/22 13:55:26", True],
        ["alice", 25, "09/19/24 14:38:12", False],
        ["alice", 33, "09/19/22 13:55:26", True],
        ["bob", 30, "09/19/22 13:55:26", False],
        ["bobby", 21, "09/19/22 13:55:26", None],
        ["charly", 13, "09/19/22 13:55:26", True],
        [None, 28, "10/19/22 13:55:26", False],
    ]

    df2 = pd.DataFrame(data2, columns=["Name", "Age", "Datetime", "Active"])
    df2["Datetime"] = df2["Datetime"].apply(
        lambda x: datetime.strptime(x, "%m/%d/%y %H:%M:%S")
    )

    compared_data_expected = [
        ["df1", 0, "alice", 20.0, "09/19/22 13:55:26", True],
        ["df2", 0, "alice", 21.0, "09/19/22 13:55:26", True],
        ["df1", 1, "alice", 25.0, "09/19/22 13:55:26", False],
        ["df2", 1, "alice", 25.0, "09/19/24 14:38:12", False],
        ["df1", 2, "alice", None, "09/19/22 13:55:26", True],
        ["df2", 2, "alice", 33.0, "09/19/22 13:55:26", True],
        ["df1", 3, "bob", 30.0, "09/19/22 13:55:26", True],
        ["df2", 3, "bob", 30.0, "09/19/22 13:55:26", False],
        ["df1", 4, "bob", 21.0, "09/19/22 13:55:26", None],
        ["df2", 4, "bobby", 21.0, "09/19/22 13:55:26", None],
        ["df1", 5, None, 13.0, "09/19/22 13:55:26", True],
        ["df2", 5, "charly", 13.0, "09/19/22 13:55:26", True],
        ["df1", 6, "daisy", 27.0, "09/19/22 13:55:26", True],
        ["df2", 6, None, 28.0, "10/19/22 13:55:26", False],
    ]

    df_compared_expected = pd.DataFrame(
        compared_data_expected,
        columns=["_dataframe_", "_index_", "Name", "Age", "Datetime", "Active"],
    ).set_index(["_dataframe_", "_index_"])
    df_compared_expected.index.names = [None, None]
    df_compared_expected["Age"] = df_compared_expected["Age"].astype(np.float64)
    df_compared_expected["Datetime"] = df_compared_expected["Datetime"].apply(
        lambda x: datetime.strptime(x, "%m/%d/%y %H:%M:%S")
    )

    df_compared_actual = pandas_dataframe_operations.compare_dataframes(df1, df2)

    pd.testing.assert_frame_equal(df_compared_expected, df_compared_actual)


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
