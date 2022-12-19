import numpy as np
import pandas as pd


def impute_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values in a Pandas dataframe.

    For each missing value in a given column in the dataframe, a random value is chosen
    from the non-missing values in that column, where the randomly drawn value then
    replaces the missing value.

    Inputs:
    df: pd.DataFrame
        Pandas dataframe containing missing values.

    Outputs:
    df_imputed: pd.DataFrame
        Pandas dataframe with missing values imputed.

    Example:
    Take a dataframe containing missing values df1 and return a dataframe df2 with all
    missing values imputed:

        df2 = df_impute(df)
    """
    # Create a copy of the original dataframe.
    df_imputed = df.copy(deep=True)

    # Loop over all columns in the dataframe.
    for i in range(0, len(df.columns)):

        # Get the name of the current column.
        current_column_name = df.columns[i]

        # Get the current column as a dataframe.
        df_current_column = pd.DataFrame(df[current_column_name].copy(deep=True))

        # Initialize a dataframe that will contain the column with imputed values.
        df_current_column_imputed = df_current_column.copy()

        # Check if the current column contains missing values. If so, start processing
        # this column.
        if df_current_column.isnull().sum()[0] > 0:

            # Get a dataframe with missing values only, based on column type. When
            # dtype == 'object' the column contains text, otherwise is contains numbers.
            if df_current_column[current_column_name].dtype == "object":
                df_missing_values = df_current_column[
                    pd.isnull(df_current_column[current_column_name])
                ]
            else:
                df_missing_values = df_current_column[
                    np.isnan(df_current_column[current_column_name])
                ]

            # Loop over the records in the dataframe with missing values.
            for j in range(0, df_missing_values.shape[0]):

                # Draw a random sample from the non-missing elements in the column.
                random_sample_non_missing_data = (
                    df_current_column.dropna().sample(n=1).values[0][0]
                )

                # Get the row index of the current missing value that needs to be
                # imputed.
                row_index = df_missing_values[current_column_name].index[j]

                # Assign the random sample value to the current missing value in the
                # column.
                df_current_column_imputed.loc[
                    row_index, :
                ] = random_sample_non_missing_data

        # Update the output dataframe with the imputed values.
        df_imputed[current_column_name] = np.where(
            df[current_column_name].isnull(),
            df_current_column_imputed[df_current_column_imputed.columns[0]],
            df[current_column_name],
        )

    return df_imputed
