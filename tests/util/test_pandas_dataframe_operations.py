from cdtools.util import pandas_dataframe_operations
import numpy as np
import pandas as pd


def test_impute_dataframe():

    input_data = [
        ["alice", 20],
        ["alice", 25],
        ["alice", None],
        ["bob", 30],
        ["bob", 21],
        [None, 13],
    ]

    df_input = pd.DataFrame(input_data, columns=["Name", "Age"])

    output_data_expected = [
        ["alice", 20],
        ["alice", 25],
        ["alice", 30],
        ["bob", 30],
        ["bob", 21],
        ["alice", 13],
    ]

    df_output_expected = pd.DataFrame(output_data_expected, columns=["Name", "Age"])
    df_output_expected["Age"] = df_output_expected["Age"].astype(np.float64)

    df_output_actual = pandas_dataframe_operations.impute_dataframe(
        df_input, random_state=0
    )

    print(df_output_actual.dtypes)
    print(df_output_expected.dtypes)

    pd.testing.assert_frame_equal(df_output_expected, df_output_actual)
