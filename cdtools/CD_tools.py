####################
# Import libraries #
####################

# General purpose
# import joblib  # Import this library before sklearn in order to avoid the following warning "DeprecationWarning: sklearn.externals.joblib is deprecated in 0.21 and will be removed in 0.23. Please import this functionality directly from joblib, which can be installed with: pip install joblib."
import matplotlib.pyplot as plt
import numpy as np

# import openpyxl
import pandas as pd
import seaborn as sns

# from pprint import pprint

# cdtools
from cdtools.dataprocessing.feature_engineering import encode_labels, get_feature_lists

# Imbalanced data
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

# Metrics
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    cross_val_score,
    cross_validate,
    RandomizedSearchCV,
    ShuffleSplit,
    StratifiedKFold,
    train_test_split,
)
from yellowbrick.model_selection import LearningCurve
from treeinterpreter import treeinterpreter as ti

# Machine learning algorithms
from lightgbm import LGBMClassifier
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeCV,
    RidgeClassifierCV,
)  # For machine learning using linear models (for regression an classification problems)
from sklearn.naive_bayes import (
    GaussianNB,
)  # For machine learning using Gaussian Naive Bayes (GNB)
from sklearn.neighbors import (
    KNeighborsClassifier,
)  # For machine learning using K-Nearest Neighbors (K-NN)
from sklearn.neural_network import (
    MLPClassifier,
)  # For machine learning using a multi-layer perceptron (MLP) neural network
from sklearn.svm import (
    SVC,
)  # For machine learning using C-Support Vector Classification.
from sklearn.tree import DecisionTreeClassifier

# from xgboost import XGBClassifier

# Feature selection
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

# Interpretation
import lime
import lime.lime_tabular

##################
# Define classes #
##################

# Class: CDML_Object
#
# Author: Rien Dijkstra
#
# Source:
#
# Syntax: result = CDML_Object()
#
# Purpose: Creates/initializes an empty object for use in the class CDML
#
# Inputs: None
#
# Outputs: An empty object for use in the class CDML
#
# Example: Create an empty object to store data into, for use in the class CDML
#
#          self.data = CDML_Object()
#
# Dependencies: None
#
class CDML_Object(object):
    pass


# Class: CDML
#
# Author: Rien Dijkstra
#
# Source:
#
# Syntax:
#
# Purpose: Various function for Machine Learning purposes in Python
#
# Inputs:
#
# Outputs:
#
# Example:
#
# Dependencies: None
#
class CDML:
    def __init__(
        self,
        dataframe,
        column_data_types,
        keys,
        labels,
        features_categorical_classes_2_drop=[],
    ):

        self.data = CDML_Object()
        self.store_basic_information(
            dataframe,
            column_data_types,
            keys,
            labels,
            features_categorical_classes_2_drop,
        )
        self.preprocess_categorical_features()
        self.label_encode_labels()

    def define_column_transformer(self):

        #############################
        # Define column transformer #
        #############################

        # Create a column transformer that can be used for one-hot-encoding later. See:
        # https://jorisvandenbossche.github.io/blog/2018/05/28/scikit-learn-columntransformer/
        # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
        # https://datascience.stackexchange.com/questions/41113/deprecationwarning-the-categorical-features-keyword-is-deprecated-in-version
        if "" not in self.data.features_categorical_classes_2_drop:
            self.data.ColumnTransformer = ColumnTransformer(
                [
                    (
                        "one_hot_encoder",
                        preprocessing.OneHotEncoder(
                            drop=self.data.features_categorical_classes_2_drop
                        ),
                        self.data.features_categorical_idx,
                    )
                ],  # Transformers- List of (name, transformer, column(s)) tuples specifying the transformer objects to be applied to subsets of the data and the numbers of the columns to be transformed.
                remainder="passthrough",  # Leave the rest of the columns untouched
            )
        else:
            self.data.ColumnTransformer = ColumnTransformer(
                [
                    (
                        "one_hot_encoder",
                        preprocessing.OneHotEncoder(drop=None),
                        self.data.features_categorical_idx,
                    )
                ],  # Transformers- List of (name, transformer, column(s)) tuples specifying the transformer objects to be applied to subsets of the data and the numbers of the columns to be transformed.
                remainder="passthrough",  # Leave the rest of the columns untouched
            )

    def determine_dummy_columns_to_drop(self):

        # Determine which dummy columns can be dropped
        if (self.data.features_categorical != []) and (
            self.data.features_categorical_classes_2_drop != []
        ):

            for i in range(0, len(self.data.features_categorical)):
                if self.data.features_categorical_classes_2_drop[i] != "":
                    self.data.features_categorical_dummy_2_drop = (
                        self.data.features_categorical_dummy_2_drop
                        + [
                            self.data.features_categorical[i]
                            + " "
                            + str(self.data.features_categorical_classes_2_drop[i])
                        ]
                    )
                else:
                    self.data.features_categorical_dummy_2_drop = (
                        self.data.features_categorical_dummy_2_drop + [""]
                    )

        else:

            self.data.features_categorical_dummy_2_drop = []

    def determine_index_categorical_features_in_feature_list(self):

        # Determine the index of the categorical features in the feature list
        if (self.data.features_categorical != []) and (
            self.data.features_categorical_classes_2_drop != []
        ):

            for i in range(0, len(self.data.features_categorical)):

                self.data.features_categorical_idx = (
                    self.data.features_categorical_idx
                    + [
                        self.data.df[self.data.features].columns.get_loc(
                            self.data.features_categorical[i]
                        )
                    ]
                )

        else:

            self.data.features_categorical_idx = []

    def get_categorical_features_as_numpy_array(self):

        # Get the categorical features as numpy.ndarray
        self.data.df_features_categorical = self.data.df.copy(deep=True)
        self.data.df_features_categorical = self.data.df_features_categorical[
            self.data.features_categorical
        ]
        self.data.np_features_categorical = self.data.df_features_categorical.values
        self.data.np_features_categorical_original = (
            self.data.df_features_categorical.values
        )

    def include_dummy_variables(self):

        # ###################################################################
        # Add dummy variables for the categorical features to the dataframe #
        #####################################################################

        # Loop over the categorical features and create dummy variables for each of them
        for i in self.data.features_categorical:
            self.data.df = pd.concat(
                [
                    self.data.df,
                    pd.get_dummies(
                        self.data.df[i],
                        columns=[i],
                        prefix=i,
                        prefix_sep=" ",
                        drop_first=False,
                    ),
                ],
                axis=1,
            )

        # Create a list with the names of the dummy variables
        self.data.features_categorical_dummy = [
            x
            for x in self.data.df.columns
            if x
            not in self.data.keys
            + self.data.features
            + self.data.features_categorical_le
            + self.data.labels
        ]

        # Create a list with the names of the dummy categorical features and the numerical features in the order of the original list of features.
        flag = 0
        self.data.features_dummy = []
        for i in range(0, len(self.data.features)):

            # Check which features are contained in the list of categorical features for
            # which dummy variables were created. For details, see
            # https://stackoverflow.com/questions/16380326/check-if-substring-is-in-a-list-of-strings
            result = [
                self.data.features[i] in word
                for word in self.data.features_categorical_dummy
            ]

            for j in range(0, len(result)):
                if result[j] == True:
                    self.data.features_dummy = self.data.features_dummy + [
                        self.data.features_categorical_dummy[j]
                    ]
                    flag = 1

            if flag == 0:
                self.data.features_dummy = self.data.features_dummy + [
                    self.data.features[i]
                ]

            flag = 0

        self.data.features_dummy = [
            x
            for x in self.data.features_dummy
            if x not in self.data.features_categorical_dummy_2_drop
        ]  # Remove surplus dummy columns from the list

    def label_encode_categorical_features(self):

        # ##########################################################################
        # 1b1) Label encode the categorical features and add them to the dataframe #
        ############################################################################
        self.get_categorical_features_as_numpy_array()

        # Label encode the categorical features and store the original unique values for each of these features
        self.data.features_categorical_distinct_values = {}
        self.data.features_categorical_classes_2_drop_le = []

        # When there are categorical features, initialize and fill all relevant variables.
        if (self.data.features_categorical != []) and (
            self.data.features_categorical_classes_2_drop != []
        ):

            for feature_index in range(0, len(self.data.features_categorical_idx)):
                le = preprocessing.LabelEncoder()
                le.fit(self.data.np_features_categorical[:, feature_index])
                self.data.np_features_categorical[:, feature_index] = le.transform(
                    self.data.np_features_categorical[:, feature_index]
                )
                self.data.features_categorical_distinct_values[
                    feature_index
                ] = le.classes_.astype(str)
                if "" not in self.data.features_categorical_classes_2_drop:
                    self.data.features_categorical_classes_2_drop_le = (
                        self.data.features_categorical_classes_2_drop_le
                        + le.transform(
                            [
                                self.data.features_categorical_classes_2_drop[
                                    feature_index
                                ]
                            ]
                        ).tolist()
                    )
                else:
                    self.data.features_categorical_classes_2_drop_le = (
                        self.data.features_categorical_classes_2_drop_le + [""]
                    )

        # Convert the numpy array to dataframe (https://stackoverflow.com/questions/20763012/creating-a-pandas-dataframe-from-a-numpy-array-how-do-i-specify-the-index-colum)
        self.data.df_features_categorical_le = pd.DataFrame(
            data=self.data.np_features_categorical,  # values
            index=self.data.df_features_categorical.index,  # index
            columns=self.data.df_features_categorical.columns + "_le",
        )  # column names

        self.data.features_categorical_le = (
            self.data.df_features_categorical_le.columns.tolist()
        )

        # Add label encoded categorical features to the dataframe
        self.data.df = self.data.df.merge(
            self.data.df_features_categorical_le, left_index=True, right_index=True
        )

        # Create a list with the names of the label encoded categorical features and the numerical features in the order of the original list of features.
        self.data.features_le = []
        for i in range(0, len(self.data.features)):
            if self.data.features[i] + "_le" in self.data.features_categorical_le:
                self.data.features_le = self.data.features_le + [
                    self.data.features[i] + "_le"
                ]
            else:
                self.data.features_le = self.data.features_le + [self.data.features[i]]

    def label_encode_labels(self):

        self.data.label_class_names = [None] * len(self.data.labels)
        for i in range(0, len(self.data.labels)):
            self.data.labels = self.data.labels + [
                self.data.labels[i] + "_label_encoded"
            ]
            (
                self.data.df[self.data.labels[i] + "_label_encoded"],
                self.data.label_class_names[i],
            ) = encode_labels(self.data.df[self.data.labels[i]])

    def preprocess_categorical_features(self):

        self.data.features_categorical_dummy_2_drop = []
        self.data.features_categorical_idx = []

        self.determine_dummy_columns_to_drop()
        self.determine_index_categorical_features_in_feature_list()
        self.label_encode_categorical_features()
        self.define_column_transformer()
        self.include_dummy_variables()

    def store_basic_information(
        self,
        dataframe,
        column_data_types,
        keys,
        labels,
        features_categorical_classes_2_drop,
    ):

        self.data.df = dataframe.copy(deep=True)
        self.data.keys = keys
        (
            self.data.features,
            self.data.features_categorical,
            self.data.features_numerical,
            self.data.features_boolean,
        ) = get_feature_lists(column_data_types, keys, labels)
        self.data.labels = labels
        self.data.features_categorical_classes_2_drop = (
            features_categorical_classes_2_drop
        )

    ###################################################
    # 2) Split the data in training and test datasets #
    ###################################################
    def split_data(self, test_size=0.5, random_state=0, sampling=None):

        self.data_splitted = CDML_Object()
        self.data_splitted.test_size = test_size
        self.data_splitted.random_state = random_state
        self.data_splitted.sampling = sampling

        # Idx_train : contains the dataframe index for the training dataset
        # Idx_test : contains the dataframe index for the test dataset
        # Id_train : contains the keys for the training dataset
        # Id_test : contains the keys for the test dataset
        # X_train_original : contains the original versions of the features for the training dataset
        # X_test_original : contains the original versions of the features for the test dataset
        # X_train_le : contains the label encoded versions of the features for the training dataset
        # X_test_le : contains the label encoded versions of the features for the test dataset
        # X_train : contains the dummy encoded versions of the features for the training dataset (this is the default data to be used)
        # X_test : contains the dummy encoded versions of the features for the test dataset (this is the default data to be used)
        # X_train_ohe : contains the one hot encoded versions of the features for the training dataset
        # X_test_ohe : contains the one hot encoded versions of the features for the test dataset
        # y_train : contains the original version of the label for the training dataset
        # y_test : contains the original version of the label for the test dataset
        # y_train_le : contains the label encoded version of the label for the training dataset
        # y_test_le : contains the label encoded version of the label for the test dataset
        (
            self.data_splitted.Idx_train,
            self.data_splitted.Idx_test,
            self.data_splitted.Id_train,
            self.data_splitted.Id_test,
            self.data_splitted.X_train_original,
            self.data_splitted.X_test_original,
            self.data_splitted.X_train_le,
            self.data_splitted.X_test_le,
            self.data_splitted.X_train,
            self.data_splitted.X_test,
            self.data_splitted.y_train,
            self.data_splitted.y_test,
            self.data_splitted.y_train_le,
            self.data_splitted.y_test_le,
        ) = train_test_split(
            self.data.df.index.tolist(),  # Dataframe index
            self.data.df[self.data.keys].values.tolist(),  # Dataset keys
            self.data.df[
                self.data.features
            ].values.tolist(),  # Model features (original)
            self.data.df[
                self.data.features_le
            ].values.tolist(),  # Model features (label encoded)
            self.data.df[
                self.data.features_dummy
            ].values.tolist(),  # Model features (dummy) --> This is the default data to be used, e.g. create X_train etc...
            self.data.df[self.data.labels[0]].values.tolist(),  # Model labels
            self.data.df[
                self.data.labels[1]
            ].values.tolist(),  # Model labels (label encoded)
            test_size=self.data_splitted.test_size,  # % of test data of the original data
            random_state=self.data_splitted.random_state,
        )  # Make sure data is always split the same way

        # self.data_splitted.X_train_ohe = np.array(
        #    self.data.ColumnTransformer.fit_transform(
        #        self.data_splitted.X_train_original
        #    ),
        #    dtype=np.float,
        # )
        # self.data_splitted.X_test_ohe = np.array(
        #    self.data.ColumnTransformer.fit_transform(
        #        self.data_splitted.X_test_original
        #    ),
        #    dtype=np.float,
        # )

        # When asked for, perform sampling operations for balancing training data
        # FIX: also apply to the label and one hot encoded data
        if sampling != None:

            if sampling == "SMOTE":
                print("Apply oversampling using SMOTE")
                smt = SMOTE()
                self.data_splitted.X_train, self.data_splitted.y_train = smt.fit_sample(
                    self.data_splitted.X_train, self.data_splitted.y_train
                )
            elif sampling == "NearMiss":
                print("Apply undersampling using NearMiss")
                nr = NearMiss()
                self.data_splitted.X_train, self.data_splitted.y_train = nr.fit_sample(
                    self.data_splitted.X_train, self.data_splitted.y_train
                )
            else:
                print(
                    "Note: No valid sampling method specified. Sampling will be set to None!"
                )
                sampling = None

        # Create dataframes for the training and test datasets
        self.data_splitted.df_train = pd.DataFrame(
            columns=self.data.keys + self.data.features_dummy + [self.data.labels[0]]
        )
        self.data_splitted.df_test = pd.DataFrame(
            columns=self.data.keys + self.data.features_dummy + [self.data.labels[0]]
        )

        # Fill the dataframes with features
        for i in range(0, len(self.data.features_dummy)):
            self.data_splitted.df_train[self.data.features_dummy[i]] = [
                item[i] for item in self.data_splitted.X_train
            ]
            self.data_splitted.df_test[self.data.features_dummy[i]] = [
                item[i] for item in self.data_splitted.X_test
            ]

        # Fill the dataframes with labels
        self.data_splitted.df_train[self.data.labels[0]] = self.data_splitted.y_train
        self.data_splitted.df_test[self.data.labels[0]] = self.data_splitted.y_test

        if sampling == "SMOTE":

            # Set index name to 'index'. Original indices are no longer available after SMOTE.
            self.data_splitted.df_train.reset_index(level=0, inplace=True)
            self.data_splitted.df_train = self.data_splitted.df_train.set_index("index")
            self.data_splitted.df_test.reset_index(level=0, inplace=True)
            self.data_splitted.df_test = self.data_splitted.df_test.set_index("index")

        elif sampling == None:

            # Fill the dataframes with keys
            for i in range(0, len(self.data.keys)):
                self.data_splitted.df_train[self.data.keys[i]] = [
                    item[i] for item in self.data_splitted.Id_train
                ]
                self.data_splitted.df_test[self.data.keys[i]] = [
                    item[i] for item in self.data_splitted.Id_test
                ]

            # Add original indices to the dataframes
            self.data_splitted.df_train["index"] = self.data_splitted.Idx_train
            self.data_splitted.df_train = self.data_splitted.df_train.set_index("index")
            self.data_splitted.df_test["index"] = self.data_splitted.Idx_test
            self.data_splitted.df_test = self.data_splitted.df_test.set_index("index")

        return

    #####################
    # 3) Model the data #
    #####################

    ##################
    # 3a) Regression #
    ##################

    def regression_model_data(self, algorithm, nrfolds=5):

        ##########################
        # 3a1) Model calculation #
        ##########################

        self.model = CDML_Object()

        # 0. Initialize model
        exec("self.model.algorithm = " + algorithm)

        # 1. Fit model
        self.model.algorithm = self.model.algorithm.fit(
            self.data_splitted.X_train, self.data_splitted.y_train
        )

        # 2. Make predictions
        self.model.y_train_model = self.model.algorithm.predict(
            self.data_splitted.X_train
        )
        self.model.y_test_model = self.model.algorithm.predict(
            self.data_splitted.X_test
        )

        # 3. Calculate (percentage) residuals
        self.model.y_train_residual = (
            self.model.y_train_model - self.data_splitted.y_train
        )
        self.model.y_train_residual_percentage = 100 * (
            self.model.y_train_residual / self.data_splitted.y_train
        )
        self.model.y_test_residual = self.model.y_test_model - self.data_splitted.y_test
        self.model.y_test_residual_percentage = 100 * (
            self.model.y_test_residual / self.data_splitted.y_test
        )

        # 4. Calculate metrics
        self.model.explained_variance_score_train = explained_variance_score(
            self.data_splitted.y_train, self.model.y_train_model
        )
        self.model.mean_absolute_error_train = mean_absolute_error(
            self.data_splitted.y_train, self.model.y_train_model
        )
        self.model.mean_squared_error_train = mean_squared_error(
            self.data_splitted.y_train, self.model.y_train_model
        )
        # self.model.mean_squared_log_error_train = mean_squared_log_error(self.data_splitted.y_train,self.model.y_train_model)
        self.model.median_absolute_error_train = median_absolute_error(
            self.data_splitted.y_train, self.model.y_train_model
        )
        self.model.r2_score_train = r2_score(
            self.data_splitted.y_train, self.model.y_train_model
        )

        self.model.explained_variance_score_test = explained_variance_score(
            self.data_splitted.y_test, self.model.y_test_model
        )
        self.model.mean_absolute_error_test = mean_absolute_error(
            self.data_splitted.y_test, self.model.y_test_model
        )
        self.model.mean_squared_error_test = mean_squared_error(
            self.data_splitted.y_test, self.model.y_test_model
        )
        # self.model.mean_squared_log_error_test = mean_squared_log_error(self.data_splitted.y_test,self.model.y_test_model)
        self.model.median_absolute_error_test = median_absolute_error(
            self.data_splitted.y_test, self.model.y_test_model
        )
        self.model.r2_score_test = r2_score(
            self.data_splitted.y_test, self.model.y_test_model
        )

        # 5. Create dataframes for the training and test datasets, now including the predictions
        self.model.df_train_incl_predictions = self.data_splitted.df_train.copy(
            deep=True
        )
        self.model.df_train_incl_predictions["y"] = self.data_splitted.y_train
        self.model.df_train_incl_predictions["y_model"] = self.model.y_train_model
        self.model.df_train_incl_predictions["residual"] = self.model.y_train_residual
        self.model.df_train_incl_predictions[
            "residual_percentage"
        ] = self.model.y_train_residual_percentage

        self.model.df_test_incl_predictions = self.data_splitted.df_test.copy(deep=True)
        self.model.df_test_incl_predictions["y"] = self.data_splitted.y_test
        self.model.df_test_incl_predictions["y_model"] = self.model.y_test_model
        self.model.df_test_incl_predictions["residual"] = self.model.y_test_residual
        self.model.df_test_incl_predictions[
            "residual_percentage"
        ] = self.model.y_test_residual_percentage

        #########################
        # 3a2) Cross validation #
        #########################

        cross_validation = CDML_Object()

        # Perform cross validation on this model, the results of which can be used for
        # selection of the best model later on. For scoring metrics, see the list on
        # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        cross_validation.nrfolds = nrfolds
        cross_validation_tmp = cross_validate(
            self.model.algorithm,
            self.data_splitted.X_train,
            self.data_splitted.y_train,
            cv=nrfolds,
            scoring=(
                "explained_variance",  # Explained variance
                "neg_mean_absolute_error",  # Mean absolute error (MnAE)
                "neg_mean_squared_error",  # Mean squared error (MSE)
                #'neg_mean_squared_log_error', # Mean squared log error (MSLE)
                "neg_median_absolute_error",  # Median absolute error (MdAE)
                "r2",  # R-squared
            ),
            return_train_score=True,
        )

        # Store the results in cross_validation_tmp in separate
        # variables under the cross_validation object
        for key in cross_validation_tmp:
            exec("cross_validation." + key + " = cross_validation_tmp['" + key + "']")

        # In scikit-learn, all scorer objects follow the convention that higher return values
        # are better than lower return values. Thus metrics which measure the distance between
        # the model and the data, like metrics.mean_squared_error, are available as
        # neg_mean_squared_error which return the negated value of the metric (see
        # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter).
        # However, we are also interested in the original positive values of the negated
        # metrics. These are calculated below by taking their absolute values (using np.abs).
        cross_validation.train_mean_absolute_error = np.abs(
            cross_validation.train_neg_mean_absolute_error
        )
        cross_validation.train_mean_squared_error = np.abs(
            cross_validation.train_neg_mean_squared_error
        )
        # cross_validation.train_mean_squared_log_error = np.abs(cross_validation.train_neg_mean_squared_log_error)
        cross_validation.train_median_absolute_error = np.abs(
            cross_validation.train_neg_median_absolute_error
        )
        cross_validation.test_mean_absolute_error = np.abs(
            cross_validation.test_neg_mean_absolute_error
        )
        cross_validation.test_mean_squared_error = np.abs(
            cross_validation.test_neg_mean_squared_error
        )
        # cross_validation.test_mean_squared_log_error = np.abs(cross_validation.test_neg_mean_squared_log_error)
        cross_validation.test_median_absolute_error = np.abs(
            cross_validation.test_neg_median_absolute_error
        )

        self.model.cross_validation = cross_validation

        return

    ######################
    # 3b) Classification #
    ######################

    def classification_model_data(self, algorithm, nrfolds=5, threshold=0.5):

        ##########################
        # 3b1) Model calculation #
        ##########################

        self.model = CDML_Object()

        # 0. Initialize model
        exec("self.model.algorithm = " + algorithm)
        self.model.threshold = threshold

        # 1. Fit model
        self.model.algorithm = self.model.algorithm.fit(
            self.data_splitted.X_train, self.data_splitted.y_train
        )

        # 2. Make predictions
        self.model.y_train_model = self.model.algorithm.predict(
            self.data_splitted.X_train
        )
        self.model.y_test_model = self.model.algorithm.predict(
            self.data_splitted.X_test
        )

        # 3. Calculate the confusion matrix
        self.model.confusion_matrix_train = metrics.confusion_matrix(
            self.data_splitted.y_train, self.model.y_train_model
        )
        self.model.confusion_matrix_test = metrics.confusion_matrix(
            self.data_splitted.y_test, self.model.y_test_model
        )

        # 4. Calculate the classification accuracy and error
        self.model.accuracy_score_train = metrics.accuracy_score(
            self.data_splitted.y_train, self.model.y_train_model
        )
        self.model.accuracy_score_test = metrics.accuracy_score(
            self.data_splitted.y_test, self.model.y_test_model
        )
        self.model.error_score_train = 1 - self.model.accuracy_score_train
        self.model.error_score_test = 1 - self.model.accuracy_score_test

        # 5. Calculate probabilities
        self.model.y_probability_train = self.model.algorithm.predict_proba(
            self.data_splitted.X_train
        )[
            :, 1
        ]  # Discrete classifier, convert values into probabilities
        self.model.y_probability_test = self.model.algorithm.predict_proba(
            self.data_splitted.X_test
        )[
            :, 1
        ]  # Discrete classifier, convert values into probabilities

        # 6. Calculate the False Positive Rate (FPR) and the True Positive Rate
        (
            self.model.fpr_train,
            self.model.tpr_train,
            self.model.threshold_roc_curve_train,
        ) = roc_curve(self.data_splitted.y_train, self.model.y_probability_train)
        (
            self.model.fpr_test,
            self.model.tpr_test,
            self.model.threshold_roc_curve_test,
        ) = roc_curve(self.data_splitted.y_test, self.model.y_probability_test)

        # 7. Calculate the Area Under the Curve (AUC) for the ROC curve

        # 7a. First method using roc_auc_score function with observed labels and predicted probabilities as input (should yield the same result as in 7b.)
        self.model.roc_auc_score_train = roc_auc_score(
            self.data_splitted.y_train, self.model.y_probability_train
        )
        self.model.roc_auc_score_test = roc_auc_score(
            self.data_splitted.y_test, self.model.y_probability_test
        )

        # 7b. Second method using auc function with FPR and TPR as input arguments (should yield the same result as in 7a.)
        self.model.roc_auc_train = auc(self.model.fpr_train, self.model.tpr_train)
        self.model.roc_auc_test = auc(self.model.fpr_test, self.model.tpr_test)

        # 8. Calculate the precision, recall (=True Positive Rate (TPR)) and the f1-score
        # at each threshold (see point 12 below for an explanation of the f1 score)
        (
            self.model.precision_train,
            self.model.recall_train,
            self.model.threshold_precision_recall_curve_train,
        ) = precision_recall_curve(
            self.data_splitted.y_train, self.model.y_probability_train
        )
        (
            self.model.precision_test,
            self.model.recall_test,
            self.model.threshold_precision_recall_curve_test,
        ) = precision_recall_curve(
            self.data_splitted.y_test, self.model.y_probability_test
        )

        self.model.f1_train = (
            2
            * (self.model.precision_train * self.model.recall_train)
            / (self.model.precision_train + self.model.recall_train)
        )
        self.model.f1_test = (
            2
            * (self.model.precision_test * self.model.recall_test)
            / (self.model.precision_test + self.model.recall_test)
        )

        # 9. Calculate the Area Under the Curve (AUC) for the precision-recall curve
        self.model.precision_recall_auc_train = auc(
            self.model.recall_train, self.model.precision_train
        )
        self.model.precision_recall_auc_test = auc(
            self.model.recall_test, self.model.precision_test
        )

        # 10. Calculate the precision and average precision (AP) scores (single value statistics). AP summarizes a precision-recall
        # curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold
        # used as the weight (see http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
        self.model.precision_score_train = precision_score(
            self.data_splitted.y_train, self.model.y_train_model
        )
        self.model.precision_score_test = precision_score(
            self.data_splitted.y_test, self.model.y_test_model
        )
        self.model.average_precision_score_train = average_precision_score(
            self.data_splitted.y_train, self.model.y_probability_train
        )
        self.model.average_precision_score_test = average_precision_score(
            self.data_splitted.y_test, self.model.y_probability_test
        )

        # 11. Calculate the recall score (single value statistic)
        # (see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)
        self.model.recall_score_train = recall_score(
            self.data_splitted.y_train, self.model.y_train_model
        )
        self.model.recall_score_test = recall_score(
            self.data_splitted.y_test, self.model.y_test_model
        )

        # 12. Calculate the F1-score (single value statistic)
        # (see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score)
        #
        # The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its
        # best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal.
        # The formula for the F1 score is:
        #
        # F1 = 2 * (precision*recall) / (precision+recall)
        #
        self.model.f1_score_train = f1_score(
            self.data_splitted.y_train, self.model.y_train_model
        )
        self.model.f1_score_test = f1_score(
            self.data_splitted.y_test, self.model.y_test_model
        )

        # 13. Create dataframes for the training and test datasets, now including the predictions
        self.model.df_train_incl_predictions = self.data_splitted.df_train.copy(
            deep=True
        )
        self.model.df_train_incl_predictions["y"] = np.array(self.data_splitted.y_train)
        self.model.df_train_incl_predictions["y_prob"] = self.model.y_probability_train
        self.model.df_train_incl_predictions["threshold"] = self.model.threshold
        self.model.df_train_incl_predictions["y_pred"] = [
            1 if (x >= threshold) else 0 for x in self.model.y_probability_train
        ]
        self.model.df_train_incl_predictions.loc[
            (self.model.df_train_incl_predictions["y"] == 1)
            & (self.model.df_train_incl_predictions["y_pred"] == 1),
            "pred_category",
        ] = "TP"
        self.model.df_train_incl_predictions.loc[
            (self.model.df_train_incl_predictions["y"] == 1)
            & (self.model.df_train_incl_predictions["y_pred"] == 0),
            "pred_category",
        ] = "FN"
        self.model.df_train_incl_predictions.loc[
            (self.model.df_train_incl_predictions["y"] == 0)
            & (self.model.df_train_incl_predictions["y_pred"] == 0),
            "pred_category",
        ] = "TN"
        self.model.df_train_incl_predictions.loc[
            (self.model.df_train_incl_predictions["y"] == 0)
            & (self.model.df_train_incl_predictions["y_pred"] == 1),
            "pred_category",
        ] = "FP"

        self.model.df_test_incl_predictions = self.data_splitted.df_test.copy(deep=True)
        self.model.df_test_incl_predictions["y"] = np.array(self.data_splitted.y_test)
        self.model.df_test_incl_predictions["y_prob"] = self.model.y_probability_test
        self.model.df_test_incl_predictions["threshold"] = self.model.threshold
        self.model.df_test_incl_predictions["y_pred"] = [
            1 if (x >= threshold) else 0 for x in self.model.y_probability_test
        ]
        self.model.df_test_incl_predictions.loc[
            (self.model.df_test_incl_predictions["y"] == 1)
            & (self.model.df_test_incl_predictions["y_pred"] == 1),
            "pred_category",
        ] = "TP"
        self.model.df_test_incl_predictions.loc[
            (self.model.df_test_incl_predictions["y"] == 1)
            & (self.model.df_test_incl_predictions["y_pred"] == 0),
            "pred_category",
        ] = "FN"
        self.model.df_test_incl_predictions.loc[
            (self.model.df_test_incl_predictions["y"] == 0)
            & (self.model.df_test_incl_predictions["y_pred"] == 0),
            "pred_category",
        ] = "TN"
        self.model.df_test_incl_predictions.loc[
            (self.model.df_test_incl_predictions["y"] == 0)
            & (self.model.df_test_incl_predictions["y_pred"] == 1),
            "pred_category",
        ] = "FP"

        #########################
        # 3b2) Cross validation #
        #########################

        cross_validation = CDML_Object()

        # Perform cross validation on this model, the results of which can be used for
        # selection of the best model later on. For scoring metrics, see the list on
        # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        cross_validation.nrfolds = nrfolds
        cross_validation_tmp = cross_validate(
            self.model.algorithm,
            self.data_splitted.X_train,
            self.data_splitted.y_train,
            cv=nrfolds,
            scoring=("accuracy", "average_precision", "precision", "recall", "roc_auc"),
            return_train_score=True,
        )

        # Store the results in cross_validation_tmp in separate
        # variables under the cross_validation object
        for key in cross_validation_tmp:
            exec("cross_validation." + key + " = cross_validation_tmp['" + key + "']")

        self.model.cross_validation = cross_validation

        return

    #########################
    # 4) Show model results #
    #########################

    ##################
    # 4a) Regression #
    ##################

    def regression_save_weights(self):

        try:
            # Create a dataframe with weights of features (needed for weight plot)
            self.model.df_weights = pd.DataFrame()
            self.model.df_weights["feature"] = self.data.features_dummy
            self.model.df_weights["weight"] = self.model.algorithm.coef_
            self.model.df_weights = self.model.df_weights.sort_values(
                by=["weight"], ascending=False
            )

            df_intercept = pd.DataFrame(
                {"feature": "intercept", "weight": self.model.algorithm.intercept_},
                index=[0],
            )

            self.model.df_weights = pd.concat(
                [df_intercept, self.model.df_weights]
            ).reset_index(drop=True)

        except:
            print("Note: Weights are not available for this model")

        return

    def regression_show_residuals(
        self, xmin=-180, xmax=180, ymin=-180, ymax=180, nrbins=201
    ):

        plt.figure(figsize=(16, 32))
        plt.subplots_adjust(
            left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3
        )

        x = [xmin, xmax]
        y = x

        ########################################
        # Plot model vs data for training data #
        ########################################
        plt.subplot(421)
        plt.plot(self.data_splitted.y_train, self.model.y_train_model, "bo")
        plt.plot(x, y, color="red", lw=2, linestyle="--")
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        plt.xlabel("y")
        plt.ylabel("y_model")
        plt.title("y_model vs y (training data)")

        ###################################
        # Plot model vs data for test data#
        ###################################
        plt.subplot(422)
        plt.plot(self.data_splitted.y_test, self.model.y_test_model, "bo")
        plt.plot(x, y, color="red", lw=2, linestyle="--")
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        plt.xlabel("y")
        plt.ylabel("y_model")
        plt.title("y_model vs y (test data)")

        ################################
        # Plot residuals training data #
        ################################
        plt.subplot(423)
        plt.plot(self.data_splitted.y_train, self.model.y_train_residual, "bo")
        plt.plot([xmin, xmax], [0.0, 0.0], color="red", lw=2, linestyle="--")
        plt.xlim([xmin, xmax])
        plt.xlabel("y")
        plt.ylabel("y - y_model")
        plt.title("residuals (training data)")

        ############################
        # Plot residuals test data #
        ############################
        plt.subplot(424)
        plt.plot(self.data_splitted.y_test, self.model.y_test_residual, "bo")
        plt.plot([xmin, xmax], [0.0, 0.0], color="red", lw=2, linestyle="--")
        plt.xlim([xmin, xmax])
        plt.xlabel("y")
        plt.ylabel("y - y_model")
        plt.title("residuals (test data)")

        ###########################################
        # Plot percentage residuals training data #
        ###########################################
        plt.subplot(425)
        plt.plot(
            self.data_splitted.y_train, self.model.y_train_residual_percentage, "bo"
        )
        plt.plot([xmin, xmax], [0.0, 0.0], color="red", lw=2, linestyle="--")
        plt.xlim([xmin, xmax])
        plt.xlabel("y")
        plt.ylabel("y - y_model")
        plt.title("percentage residuals (training data)")

        #######################################
        # Plot percentage residuals test data #
        #######################################
        plt.subplot(426)
        plt.plot(self.data_splitted.y_test, self.model.y_test_residual_percentage, "bo")
        plt.plot([xmin, xmax], [0.0, 0.0], color="red", lw=2, linestyle="--")
        plt.xlim([xmin, xmax])
        plt.xlabel("y")
        plt.ylabel("y - y_model")
        plt.title("percentage residuals (test data)")

        ############################################
        # Plot residual distribution training data #
        ############################################
        plt.subplot(427)
        self.model.df_train_incl_predictions["residual"].hist(bins=nrbins)
        plt.xlabel("y - y_model")
        plt.ylabel("N")
        plt.title("residual distribution (training data)")

        ########################################
        # Plot residual distribution test data #
        ########################################
        plt.subplot(428)
        self.model.df_test_incl_predictions["residual"].hist(bins=nrbins)
        plt.xlabel("y - y_model")
        plt.ylabel("N")
        plt.title("residual distribution (test data)")

        plt.show()

        return

    ######################
    # 4b) Classification #
    ######################

    def classification_show_ROC_precision_recall_curves(
        self,
        show_labels="N",
        label_interval=2,
        label_offsets_ROC=[5, -5],
        label_offsets_PR=[-15, -10],
    ):

        plt.figure(figsize=(18, 9))

        ##################
        # Plot ROC curve #
        ##################
        plt.subplot(121)
        plt.plot(
            self.model.fpr_test,
            self.model.tpr_test,
            color="red",
            lw=2,
            label="Model (AUC= {1:0.2f})" "".format(0, self.model.roc_auc_score_test),
        )
        if show_labels == "Y":
            for i, label in enumerate(
                self.model.threshold_roc_curve_test.round(2).astype("str")
            ):  # Plot labels next to data points
                if i % label_interval == 0:
                    plt.plot(self.model.fpr_test[i], self.model.tpr_test[i], "bo")
                    plt.annotate(
                        text=label,
                        xy=[self.model.fpr_test[i], self.model.tpr_test[i]],
                        xytext=(label_offsets_ROC[0], label_offsets_ROC[1]),
                        textcoords="offset points",
                        # s=label,
                        rotation=45,
                    )
                # else:
                #    plt.plot(self.model.fpr_test[i], self.model.tpr_test[i], 'bo', markerfacecolor="None", markeredgewidth=1)
        plt.plot(
            [0, 1],
            [0, 1],
            color="navy",
            lw=2,
            linestyle="--",
            label="Random classifier",
        )
        plt.xlim([0.0, 1.02])
        plt.ylim([0.0, 1.02])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) curve")
        plt.legend(bbox_to_anchor=(0.37, -0.1), loc=2, borderaxespad=0.0)

        ###############################
        # Plot precision-recall curve #
        ###############################
        plt.subplot(122)
        plt.step(
            self.model.recall_test,
            self.model.precision_test,
            color="red",
            lw=2,
            label="Model (AUC = {1:0.2f})"
            "".format(0, self.model.precision_recall_auc_test),
        )
        if show_labels == "Y":
            for i, label in enumerate(
                self.model.threshold_precision_recall_curve_test.round(2).astype("str")
            ):  # Plot labels next to data points
                if i % label_interval == 0:
                    plt.plot(
                        self.model.recall_test[i], self.model.precision_test[i], "bo"
                    )
                    plt.annotate(
                        text=label,
                        xy=[self.model.recall_test[i], self.model.precision_test[i]],
                        xytext=(label_offsets_PR[0], label_offsets_PR[1]),
                        textcoords="offset points",
                        # s=label,
                        rotation=-45,
                    )
                # else:
                #    plt.plot(self.model.recall_test[i], self.model.precision_test[i], 'bo', markerfacecolor="None", markeredgewidth=1)
        N = (
            self.data_splitted.df_test.groupby([self.data.labels[0]])
            .agg({self.data.labels[0]: "count"})
            .values[0]
        )  # Total number of negatives (N) in test dataset
        P = (
            self.data_splitted.df_test.groupby([self.data.labels[0]])
            .agg({self.data.labels[0]: "count"})
            .values[1]
        )  # Total number of positives (P) in test dataset
        plt.plot(
            [0, 1],
            [P / (P + N), P / (P + N)],
            color="navy",
            lw=2,
            linestyle="--",
            label="Random classifier",
        )  # see https://classeval.wordpress.com/introduction/introduction-to-the-precision-recall-plot/ for details
        plt.plot(
            [0, 1],
            [
                self.model.average_precision_score_test,
                self.model.average_precision_score_test,
            ],
            color="navy",
            lw=2,
            linestyle=":",
            label="Average Precision (AP) line (AP = {1:0.2f})"
            "".format(0, self.model.average_precision_score_test),
        )
        plt.plot(
            [self.model.recall_score_test, self.model.recall_score_test],
            [self.model.precision_score_test, self.model.precision_score_test],
            "yo",
            label="(recall,precision)-score point (F1-score = {1:0.2f})"
            "".format(0, self.model.f1_score_test),
        )
        plt.ylim([0.0, 1.02])
        plt.xlim([0.0, 1.02])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall curve")
        plt.legend(bbox_to_anchor=(0.375, -0.1), loc=2, borderaxespad=0.0)

        plt.show()

        return

    def classification_show_prediction_distributions(self, nrbins=201):

        plt.figure(figsize=(18, 9))

        ##################################################
        # Prediction distribution plot ('True', 'False') #
        ##################################################
        plt.subplot(121)
        plt.hist(
            [
                self.model.df_test_incl_predictions[
                    (self.model.df_test_incl_predictions["pred_category"] == "TP")
                    | (self.model.df_test_incl_predictions["pred_category"] == "FN")
                ]["y_prob"],
                self.model.df_test_incl_predictions[
                    (self.model.df_test_incl_predictions["pred_category"] == "TN")
                    | (self.model.df_test_incl_predictions["pred_category"] == "FP")
                ]["y_prob"],
            ],
            bins=np.linspace(0, 1, nrbins),
            color=["springgreen", "indianred"],
            label=["True", "False"],
        )
        plt.xlim(0, 1)
        plt.title("Distribution of predicted probabilities with measured labels")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.legend(["1 (P)", "0 (N)"])
        # plt.show()

        #######################################################################################
        # Prediction distribution plot ('True (TP)', 'True (FN)', 'False (TN)', 'False (FP)') #
        #######################################################################################
        plt.subplot(122)
        plt.hist(
            [
                self.model.df_test_incl_predictions[
                    self.model.df_test_incl_predictions["pred_category"] == "TP"
                ]["y_prob"],
                self.model.df_test_incl_predictions[
                    self.model.df_test_incl_predictions["pred_category"] == "FN"
                ]["y_prob"],
                self.model.df_test_incl_predictions[
                    self.model.df_test_incl_predictions["pred_category"] == "TN"
                ]["y_prob"],
                self.model.df_test_incl_predictions[
                    self.model.df_test_incl_predictions["pred_category"] == "FP"
                ]["y_prob"],
            ],
            bins=np.linspace(0, 1, nrbins),
            color=["springgreen", "green", "indianred", "red"],
            label=["True (TP)", "True (FN)", "False (TN)", "False (FP)"],
        )
        plt.xlim(0, 1)
        plt.title(
            "Distribution of predicted probabilities with measured labels and classification errors"
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.legend(["1 (TP)", "1 (FN)", "0 (TN)", "0 (FP)"])

        plt.show()

        return

    # https://github.com/andosa/treeinterpreter
    # https://blog.datadive.net/random-forest-interpretation-with-scikit-learn/
    # https://pypi.org/project/treeinterpreter/
    def classification_show_interpretation_table(self, sort_columns="N", top=10):

        try:
            self.model.df_feature_importance.head()
        except:
            print(
                "Message: No results generated! Run calculate_feature_importance() function before using this function!"
            )
            return

        try:
            # Calculate prediction, bias, and contributions
            prediction, bias, contributions = ti.predict(
                self.model.algorithm, pd.DataFrame(self.data_splitted.X_test)
            )
            outcome = pd.DataFrame(
                {
                    "ID": self.data_splitted.Id_test,
                    "prediction": prediction[:, 1],
                    "mean": bias[:, 1],
                }
            )

            # Define colum names for the result table
            columns_value = self.data.features_dummy
            columns_contribution = (
                self.data.df[self.data.features_dummy]
                .add_suffix(" (contribution)")
                .columns.tolist()
            )
            columns_feature_importance = self.model.df_feature_importance[
                "feature"
            ].values.tolist()
            columns_feature_importance_contribution = self.model.df_feature_importance[
                "feature"
            ].values.tolist()
            for i in range(0, len(columns_feature_importance_contribution)):
                columns_feature_importance_contribution[i] = (
                    columns_feature_importance_contribution[i] + " (contribution)"
                )

            # Initialize result table
            df_result = pd.DataFrame(
                columns=self.data.keys
                + [self.data.labels[0]]
                + ["prediction", "bias"]
                + sorted(columns_value + columns_contribution)
            )

            # Add index to result table
            df_result["index"] = self.data_splitted.Idx_test
            df_result = df_result.set_index("index")

            # Fill key columns
            for i in range(0, len(self.data.keys)):
                df_result[self.data.keys[i]] = [
                    item[i] for item in self.data_splitted.Id_test
                ]

            # Fill label column
            df_result[self.data.labels[0]] = self.data_splitted.y_test

            # Fill prediction and mean columns
            df_result["prediction"] = prediction[
                :, 1
            ]  # This is, as expected, identical to self.model.df_test_incl_predictions['y_prob']!!
            df_result["bias"] = bias[:, 1]

            # Fill value and contribution of each column
            for i in range(0, len(self.data.features_dummy)):
                df_result[columns_value[i]] = self.data_splitted.df_test[
                    columns_value[i]
                ]
                df_result[columns_contribution[i]] = contributions[:, i, 1]

            # Sort the columns
            if sort_columns == "N":
                print("Note: Model features are sorted alphabetically!")
                columns = df_result.columns
            else:
                print("Note: Model features are sorted by importance for the model!")
                columns = (
                    self.data.keys + [self.data.labels[0]] + ["prediction", "bias"]
                )

                for i in range(0, len(columns_feature_importance)):
                    columns = (
                        columns
                        + [columns_feature_importance[i]]
                        + [columns_feature_importance_contribution[i]]
                    )

            # Sort the records (in order of descending probability)
            df_result = df_result.sort_values(by=["prediction"], ascending=False)

            # Display the interpretation table
            zero_importance_features = self.model.df_feature_importance[
                self.model.df_feature_importance["importance"] == 0
            ]["feature"].tolist()
            print(
                "Features with zero importance for the model: ",
                zero_importance_features,
            )
            display(
                df_result[columns]
                .drop(zero_importance_features, axis=1)
                .head(top)
                .style.bar(align="zero", color=["#d65f5f", "#5fba7d"])
            )

            # Store the interpretation table
            self.model.df_interpretation = df_result

        except:
            print("Note: Interpretation table is not available for this model")

        return

    def classification_transpose_interpretation_table(self):

        # Make a copy of the interpretation table
        df = self.model.df_interpretation.copy(deep=True)

        # Transpose the interpretation table
        df_T = pd.melt(
            df, id_vars=self.data.keys, var_name="Entity", value_name="Value"
        ).sort_values(by=self.data.keys + ["Entity"])

        # Extract the features from the interpretation table, by removing records
        # containing '(contribution)' in the name or records equal to 'prediction',
        # 'bias', or the name of the target (label)
        df_T_features = df_T[
            (~df_T["Entity"].str.contains("\(contribution\)"))
            & (~df_T["Entity"].isin(["prediction", "bias"] + [self.data.labels[0]]))
        ]
        df_T_features = df_T_features.rename(columns={"Value": "Feature"})

        # Extract the label from the interpretation table
        df_T_label = df_T[df_T["Entity"].isin([self.data.labels[0]])]
        df_T_label = df_T_label.rename(columns={"Value": "Label"})

        # Extract the variable contribution and bias from the interpretation table.
        df_T_contributions = df_T[
            (df_T["Entity"].str.contains("\(contribution\)"))
            | (df_T["Entity"].isin(["bias"]))
        ]
        df_T_contributions = df_T_contributions.rename(
            columns={"Value": "Contribution"}
        )
        df_T_contributions["Entity"] = df_T_contributions["Entity"].apply(
            lambda x: x.replace(" (contribution)", "")
        )

        # Extract the predictions from the interpretation table
        df_T_predictions = df_T[df_T["Entity"].isin(["prediction"])]
        df_T_predictions = df_T_predictions.rename(columns={"Value": "Prediction"})

        # Create final results by merging features and label
        df_result = (
            df_T_features.merge(
                df_T_label,
                left_on=self.data.keys + ["Entity"],
                right_on=self.data.keys + ["Entity"],
                how="outer",
            )
            .sort_values(by=self.data.keys + ["Entity"])
            .reset_index(drop=True)
        )

        # Merge contributions into the final results
        df_result = (
            df_result.merge(
                df_T_contributions,
                left_on=self.data.keys + ["Entity"],
                right_on=self.data.keys + ["Entity"],
                how="outer",
            )
            .sort_values(by=self.data.keys + ["Entity"])
            .reset_index(drop=True)
        )

        # Merge predictions into the final results
        df_result = (
            df_result.merge(
                df_T_predictions,
                left_on=self.data.keys + ["Entity"],
                right_on=self.data.keys + ["Entity"],
                how="outer",
            )
            .sort_values(by=self.data.keys + ["Entity"])
            .reset_index(drop=True)
        )

        # Store the transposed interpretation table
        self.model.df_interpretation_transposed = df_result

        return

    # Local Interpretable Model-agnostic Explanations (LIME)
    # https://github.com/marcotcr/lime/issues/210
    def classification_show_interpretation_table_LIME(
        self, kernel_width=None, num_samples=5000, seed=1, top=10
    ):

        # Create objects for storing LIME results
        self.model.LIME = CDML_Object()
        self.model.LIME.kernel_width = kernel_width
        self.model.LIME.num_samples = num_samples
        self.model.LIME.seed = seed

        #############################
        # Perform the LIME analysis #
        #############################

        # Explain the predictions of the classifier on the data
        np.random.seed(
            self.model.LIME.seed
        )  # If this is not used, the results vary slightly with each run. I suspect because of different random sampling around the original data point

        self.model.LIME.explainer = lime.lime_tabular.LimeTabularExplainer(
            np.asarray(self.data_splitted.X_train_le),
            feature_names=self.data.features,
            class_names=self.data.label_class_names,
            discretize_continuous=True,
            categorical_features=self.data.features_categorical_idx,
            categorical_names=self.data.features_categorical_distinct_values,
            kernel_width=self.model.LIME.kernel_width,
        )

        # Extract prediction results for each individual record

        # https://towardsdatascience.com/decrypting-your-machine-learning-model-using-lime-5adc035109b5
        # https://github.com/marcotcr/lime

        # Define colum names for the result table
        columns_value = self.data.features
        columns_contribution = (
            self.data.df[self.data.features]
            .add_suffix(" (contribution)")
            .columns.tolist()
        )
        columns_decision = (
            self.data.df[self.data.features].add_suffix(" (decision)").columns.tolist()
        )

        # Initialize result table
        df_result = pd.DataFrame(
            columns=self.data.keys
            + [self.data.labels[0]]
            + ["prediction", "intercept"]
            + sorted(columns_value + columns_contribution + columns_decision)
        )

        # Fill key columns
        for i in range(0, len(self.data.keys)):
            df_result[self.data.keys[i]] = [
                item[i] for item in self.data_splitted.Id_test
            ]

        # Fill label column
        df_result[self.data.labels[0]] = self.data_splitted.y_test

        # Fill the value, contribution and decision columns

        # Define predict function.
        def predict_fn(x):

            ##############################################################
            # Convert data from label encoded to dummy variable encoded  #
            ##############################################################

            # Create a DataFrame from the label encoded data stored as the numpy array x,
            # which contains the sample of data points artificially generated by LIME (the
            # number of records in this sample is set by the LIME variabele num_samples).
            # The columns in the dataframe are given the names of the label encoded features
            # (see https://stackoverflow.com/questions/20763012/creating-a-pandas-dataframe-
            # from-a-numpy-array-how-do-i-specify-the-index-colum on how to convert a numpy
            # array to a DataFrame using pd.DataFrame.from_records)
            df_x = pd.DataFrame.from_records(x, columns=self.data.features_le)

            # Convert the categorical label encoded feature values explicitly to integers
            # (just in case the happen to be floats, which would yield weird names such
            # as <varname>_2.0 when creating dummy variables later.
            for j in self.data.features_categorical:
                df_x = df_x.astype({j + "_le": int})

            # Inverse transform the label encoded values back to their original values.
            for feature_index in range(0, len(self.data.features_categorical_idx)):
                le = preprocessing.LabelEncoder()
                le.fit(self.data.np_features_categorical_original[:, feature_index])
                df_x[
                    self.data.features_categorical[feature_index]
                ] = le.inverse_transform(
                    df_x.iloc[
                        :,
                        self.data.features_categorical_idx[
                            feature_index
                        ] : self.data.features_categorical_idx[feature_index]
                        + 1,
                    ].values.ravel()
                )

            # Only select the columns with the original feature names
            df_x = df_x[self.data.features]

            # Create dummy variables for the categorical features. First, create an
            # empty framework dataframe that contains all dummy features that are to
            # be expected based on the original data. Second, apply pd.get_dummies
            # to the dataframe that contains the sample of data points artificially
            # generated by LIME, df_x. This dataframe may have some categories missing
            # for some variables, so not all expected dummy column will be generated.
            # Third, append the resulting dataframe to the framework dataframe. Fourth,
            # after appending, the framework dataframe will still contain missing values
            # for those dummy columns that were not generated while applying pd.get_dummies
            # to the data points artificially generated by LIME. Replace those missing
            # values with the number 0.
            df_x_framework = pd.DataFrame(columns=self.data.features_dummy)
            for j in self.data.features_categorical:
                df_x = pd.concat(
                    [
                        df_x,
                        pd.get_dummies(
                            df_x[j],
                            columns=[j],
                            prefix=j,
                            prefix_sep=" ",
                            drop_first=False,
                        ),
                    ],
                    axis=1,
                )
            df_x_framework = df_x_framework.append(df_x)
            df_x = df_x_framework.fillna(0)

            # Remove surplus dummy variables from the datafame.
            df_x = df_x[self.data.features_dummy]

            # Convert DataFrame to a numpy array.
            np_x = df_x.values

            ################################################################################################
            # Calculate the probabilities for each data point in the sample artificially generated by LIME #
            ################################################################################################
            probabilities_artificial_LIME_sample = self.model.algorithm.predict_proba(
                np_x
            )

            return probabilities_artificial_LIME_sample

        # Loop over all records in the test dataset
        for i in range(0, len(self.data_splitted.X_test_le)):

            # Get the local surrogate model, i.e. the explanation model
            #
            # An explanation is a local linear approximation of the model's behaviour. While the model may
            # be very complex globally, it is easier to approximate it around the vicinity of a particular
            # instance. While treating the model as a black box, we perturb the instance, X, we want to explain
            # and learn a sparse linear model around it, as an explanation. We sample instances around X, and
            # weight them according to their proximity to X. We then learn a linear model that approximates the
            # model well in the vicinity of X, but not necessarily globally (see https://github.com/marcotcr/lime)
            explanation_model = self.model.LIME.explainer.explain_instance(
                np.asarray(self.data_splitted.X_test_le[i]),
                predict_fn,
                num_features=len(self.data.features),
                num_samples=self.model.LIME.num_samples,
            )

            # Get the intercept for the local surrogate model
            intercept = explanation_model.intercept[1]
            df_result.at[i, "intercept"] = intercept

            # Get the weights for local the surrogate model
            weights = explanation_model.as_map()[1]

            # Get the local prediction found by LIME
            local_prediction_LIME = explanation_model.local_pred[0]
            df_result.at[i, "prediction"] = local_prediction_LIME

            # Loop over all the weigths
            for j in range(0, len(weights)):
                idx = weights[j][
                    0
                ]  # Determine the index of the feature that belongs to the current weight
                weight_current_feature = weights[j][1]
                value_current_feature = self.data_splitted.X_test_original[i][idx]
                df_result.at[i, self.data.features[idx]] = value_current_feature
                df_result.at[
                    i, self.data.features[idx] + " (contribution)"
                ] = weight_current_feature
                df_result.at[
                    i, self.data.features[idx] + " (decision)"
                ] = explanation_model.domain_mapper.discretized_feature_names[idx]

        # Add index to result table
        df_result["index"] = self.data_splitted.Idx_test
        df_result = df_result.set_index("index")

        # Sort the records (in order of descending probability)
        # df_result = df_result.sort_values(by=['prediction'],ascending=False)

        # Display the interpretation table
        display(df_result.head(top))

        # Store the interpretation table
        self.model.LIME.df_interpretation = df_result

        return

    def classification_transpose_interpretation_table_LIME(self):

        # Make a copy of the interpretation table
        df = self.model.LIME.df_interpretation.copy(deep=True)

        # Transpose the interpretation table
        df_T = pd.melt(
            df, id_vars=self.data.keys, var_name="Entity", value_name="Value"
        ).sort_values(by=self.data.keys + ["Entity"])

        # Extract the features from the interpretation table, by removing records
        # containing '(contribution)' in the name or records equal to 'prediction',
        # 'bias', or the name of the target (label)
        df_T_features = df_T[
            (~df_T["Entity"].str.contains("\(contribution\)"))
            & (~df_T["Entity"].str.contains("\(decision\)"))
            & (
                ~df_T["Entity"].isin(
                    ["prediction", "intercept"] + [self.data.labels[0]]
                )
            )
        ]
        df_T_features = df_T_features.rename(columns={"Value": "Feature"})

        # Extract the label from the interpretation table
        df_T_label = df_T[df_T["Entity"].isin([self.data.labels[0]])]
        df_T_label = df_T_label.rename(columns={"Value": "Label"})

        # Extract the variable contribution and intercept from the interpretation table.
        df_T_contributions = df_T[
            (df_T["Entity"].str.contains("\(contribution\)"))
            | (df_T["Entity"].isin(["intercept"]))
        ]
        df_T_contributions = df_T_contributions.rename(
            columns={"Value": "Contribution"}
        )
        df_T_contributions["Entity"] = df_T_contributions["Entity"].apply(
            lambda x: x.replace(" (contribution)", "")
        )

        # Extract the decisions from the interpretation table
        df_T_decisions = df_T[(df_T["Entity"].str.contains("\(decision\)"))]
        df_T_decisions = df_T_decisions.rename(columns={"Value": "Decision"})
        df_T_decisions["Entity"] = df_T_decisions["Entity"].apply(
            lambda x: x.replace(" (decision)", "")
        )

        # Extract the predictions from the interpretation table
        df_T_predictions = df_T[df_T["Entity"].isin(["prediction"])]
        df_T_predictions = df_T_predictions.rename(columns={"Value": "Prediction"})

        # Create final results by merging features and label
        df_result = (
            df_T_features.merge(
                df_T_label,
                left_on=self.data.keys + ["Entity"],
                right_on=self.data.keys + ["Entity"],
                how="outer",
            )
            .sort_values(by=self.data.keys + ["Entity"])
            .reset_index(drop=True)
        )

        # Merge contributions into the final results
        df_result = (
            df_result.merge(
                df_T_contributions,
                left_on=self.data.keys + ["Entity"],
                right_on=self.data.keys + ["Entity"],
                how="outer",
            )
            .sort_values(by=self.data.keys + ["Entity"])
            .reset_index(drop=True)
        )

        # Merge decisions into the final results
        df_result = (
            df_result.merge(
                df_T_decisions,
                left_on=self.data.keys + ["Entity"],
                right_on=self.data.keys + ["Entity"],
                how="outer",
            )
            .sort_values(by=self.data.keys + ["Entity"])
            .reset_index(drop=True)
        )

        # Merge predictions into the final results
        df_result = (
            df_result.merge(
                df_T_predictions,
                left_on=self.data.keys + ["Entity"],
                right_on=self.data.keys + ["Entity"],
                how="outer",
            )
            .sort_values(by=self.data.keys + ["Entity"])
            .reset_index(drop=True)
        )

        # Store the transposed interpretation table
        self.model.LIME.df_interpretation_transposed = df_result

        return

    #####################################
    # 4c) Regression and classification #
    #####################################

    def save_feature_importance(self):

        try:
            # Create a dataframe with importance of features (needed for feature importance plots)
            self.model.df_feature_importance = pd.DataFrame()
            self.model.df_feature_importance["feature"] = self.data.features_dummy
            self.model.df_feature_importance[
                "importance"
            ] = self.model.algorithm.feature_importances_
            self.model.df_feature_importance = (
                self.model.df_feature_importance.sort_values(
                    by=["importance"], ascending=False
                )
            )

        except:
            print("Note: Feature importance is not available for this model")

        return

    # Source: https://www.scikit-yb.org/en/latest/api/model_selection/learning_curve.html
    def show_learning_curve(
        self,
        ylim=None,
        cv=None,
        n_jobs=None,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring="r2",
        X_asarray="N",
    ):

        estimator = self.model.algorithm

        # Some algorithms (e.g. XGBoost) require X to be a numpy array i.s.o. list when
        # using the viz.fit() later in this code. Set X_asarray='Y' in order to convert
        # X to a numpy array (default: X_asarray='N', i.e. X is a list).
        if X_asarray == "Y":
            X = np.asarray(self.data.df[self.data.features_dummy].values.tolist())
        else:
            X = self.data.df[self.data.features_dummy].values.tolist()

        y = self.data.df[self.data.labels[0]].values.tolist()

        if ylim is not None:
            plt.ylim(*ylim)

        # Learning curve for true error measure
        if scoring == "error":

            viz = LearningCurve(
                estimator,
                cv=cv,
                n_jobs=n_jobs,
                train_sizes=train_sizes,
                scoring="neg_mean_squared_error",
            )
            viz.fit(X, y)
            viz.poof()

            # Calculate error and its uncertainty ranges for the training data
            error_train_mean = np.sqrt(-viz.train_scores_mean_)
            error_train_std = np.sqrt(viz.train_scores_std_)
            error_train_upper_bound = error_train_mean + error_train_std
            error_train_lower_bound = error_train_mean - error_train_std

            # Calculate error and its uncertainty ranges for the cross validation data
            error_test_mean = np.sqrt(-viz.test_scores_mean_)
            error_test_std = np.sqrt(viz.test_scores_std_)
            error_test_upper_bound = error_test_mean + error_test_std
            error_test_lower_bound = error_test_mean - error_test_std

            # Plot the learning curve
            # plt.figure(figsize=(16,8))
            plt.plot(
                viz.train_sizes_,
                error_train_mean,
                linestyle="-",
                marker="o",
                color="blue",
                label="Training Sore",
            )
            plt.fill_between(
                viz.train_sizes_,
                error_train_lower_bound,
                error_train_upper_bound,
                color="blue",
                alpha=0.1,
            )
            plt.plot(
                viz.train_sizes_,
                np.sqrt(-viz.test_scores_mean_),
                linestyle="-",
                marker="o",
                color="orange",
                label="Cross Validation Score",
            )
            plt.fill_between(
                viz.train_sizes_,
                error_test_lower_bound,
                error_test_upper_bound,
                color="orange",
                alpha=0.1,
            )
            plt.xlabel("Training Instances")
            plt.ylabel("Error")
            # plt.title(title)
            plt.legend()
            plt.show()

        # Learning curve for other error measures
        else:
            viz = LearningCurve(
                estimator,
                cv=cv,
                n_jobs=n_jobs,
                train_sizes=train_sizes,
                scoring=scoring,
            )
            viz.fit(X, y)
            viz.poof()

        return

    #########################
    # 5) Model optimization #
    #########################

    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
    # Note for scoring input argument: If None, the estimator’s score method is used (see above webpage)
    # See https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter for list of
    # possible scoring parameters to choose from.
    def RandomizedSearchCV(
        self,
        parameter_grid,
        n_iter=100,
        scoring=None,
        n_jobs=-1,
        cv=5,
        verbose=0,
        random_state=0,
    ):

        estimator = self.model.algorithm

        # Initialize randomized_search_CV
        randomized_search_CV = CDML_Object()
        randomized_search_CV.parameter_grid = parameter_grid
        randomized_search_CV.n_iter = n_iter
        randomized_search_CV.scoring = scoring
        randomized_search_CV.n_jobs = n_jobs
        randomized_search_CV.cv = cv
        randomized_search_CV.verbose = verbose
        randomized_search_CV.random_state = random_state

        # Look at parameters used by the base model
        # print("Parameters currently in use by the base model:\n")
        # pprint(estimator.get_params())
        # print()

        # Look at the parameter grid to be searched
        # print("Parameter grid to be searched:\n")
        # pprint(parameter_grid)
        # print()

        # Initialize random search
        random_search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=parameter_grid,
            n_iter=n_iter,
            scoring=scoring,
            n_jobs=n_jobs,
            cv=cv,
            verbose=verbose,
            random_state=0,
        )

        # Search parameters
        random_search.fit(self.data_splitted.X_train, self.data_splitted.y_train)

        # Store the search results
        randomized_search_CV.random_search = random_search

        self.model.randomized_search_CV = randomized_search_CV

        return

    # http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/
    def sequential_feature_selection(
        self,
        k_features=5,
        forward=True,
        floating=False,
        verbose=2,
        scoring="accuracy",
        cv=5,
    ):

        estimator = self.model.algorithm

        # Initialize sequential_feature_selection
        sequential_feature_selection = CDML_Object()
        sequential_feature_selection.k_features = k_features
        sequential_feature_selection.forward = forward
        sequential_feature_selection.floating = floating
        sequential_feature_selection.verbose = verbose
        sequential_feature_selection.scoring = scoring
        sequential_feature_selection.cv = cv

        # Build Sequential Feature Selection (SFS) object
        sequential_feature_selection.sfs = sfs(
            estimator,
            k_features=k_features,
            forward=forward,
            floating=floating,
            verbose=verbose,
            scoring=scoring,
            cv=cv,
        )

        # Perform Sequential Feature Selection
        sequential_feature_selection.sfs = sequential_feature_selection.sfs.fit(
            np.asarray(self.data_splitted.X_train),
            np.asarray(self.data_splitted.y_train),
        )

        # Store results
        self.model.sequential_feature_selection = sequential_feature_selection

        return


####################
# Define functions #
####################


def compare_binary_columns(df, column1, column2):

    CM = confusion_matrix(df[column1], df[column2])
    ACC = accuracy_score(df[column1], df[column2])
    TPR = CM[0, 0] / (CM[0, 0] + CM[0, 1])
    FPR = CM[1, 0] / (CM[1, 0] + CM[1, 1])

    plt.figure(figsize=(8, 8))
    plt.scatter(FPR, TPR, marker="o", s=80, c="red")
    plt.plot([0, 1], [0, 1])
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.title(column1 + " vs " + column2, fontsize=15)
    plt.tick_params(axis="both", which="major", labelsize=15)
    plt.show()

    print("Confusion matrix:")
    display(CM)

    print("Accuracy (ACC), False Positive Rate (FPR), True Positive Rate (TPR):")
    display(ACC, FPR, TPR)


# Write a Pandas DataFrame to an EXCEL file
def df_2_xlsx(df, filename, index=False):

    # Write DataFrame to EXCEL
    writer = pd.ExcelWriter(filename)
    df.to_excel(writer, index=index)
    writer.save()

    return


# Function: df_balance
#
# Source:
#
# Purpose: Balance a dataframe according to a given label
#
# Syntax: result = df_balance(df,label,random_state=0)
#
# Inputs: df = dataframe
#         label = label on which the dataframe needs to be
#                 balanced
#         random_state = random state for randomly drawing data
#                        in order to create equally sized subsets
#                        for each unique value of label. This
#                        option only works for mode='undersample'
#
# Outputs: result = dataframe balanced according to the given label
#
# Example: Balance a dataframe, df2, according to a given label,
#          label2, using a randomstate 10. Put the result in a new
#          dataframe called df_new
#
#          df_new = df_balance(df2,label2,random_state=10)
#
# Dependencies: None
#
def df_balance(df, label, random_state=0):

    # Create an empty copy of the original dataframe
    df2 = pd.DataFrame(columns=df.columns)

    # Find the unique label values and counts for each label value
    label_value = df[label].value_counts().index.tolist()
    label_count = df[label].value_counts().tolist()

    # Considering all unique labels, determine the smallest count
    # seen among these labels. This will be the count that will need
    # to be applied to all unique labels in order to balance the data
    n_min = np.min(label_count)

    # Loop over the unique labels
    for i in range(0, len(label_value)):

        # For the current unique label, create a random sample of size n_min
        df3 = df[df[label] == label_value[i]].sample(n=n_min, random_state=random_state)

        # Append the random sample for the current unique label to the results
        df2 = df2.append(df3)

    # Sort the output dataframe according to the original indices of the dataframe
    df2.sort_index(inplace=True)

    return df2


# Function: df_subset_remove
#
# Source: https://stackoverflow.com/questions/37313691/how-to-remove-a-pandas-dataframe-from-another-dataframe
#
# Syntax: result = df_subset_remove(df,df_subset)
#
# Purpose: Create a subset of a given dataframe
#
# Inputs: df = the given dataframe
#         df_subset = the subset dataframe to be removed from the dataframe
#
# Outputs: result = dataframe with the subset dataframe removed
#
# Example: From a given dataframe, df1, remove subset dataframe df2 and store
#          the result in a new dataframe, df3
#
#          df3 = df_subset_remove(df1,df2)
#
# Dependencies: None
#
def df_subset_remove(df, df_subset):

    # Explanation
    #
    # pd.concat adds the two DataFrames together by appending one right after the other.
    # If there is any overlap, it will be captured by the drop_duplicates method. However,
    # drop_duplicates by default leaves the first observation and removes every other
    # observation. In this case, we want every duplicate removed. Hence, the keep=False
    # parameter which does exactly that. A special note to the repeated df2. With only one
    # df2 any row in df2 not in df1 won't be considered a duplicate and will remain. This
    # solution with only one df2 only works when df2 is a subset of df1. However, if we
    # concat df2 twice, it is guaranteed to be a duplicate and will subsequently be removed.
    df1 = df.copy(deep=True)
    df2 = df_subset.copy(deep=True)
    df3 = pd.concat([df1, df2, df2]).drop_duplicates(keep=False)

    return df3


def high_correlation_filter(df, threshold=0.7):

    # Create a linear correlation matrix of the data
    df_corr = df.corr()

    # Set the diagonal values in the matrix to 0 such that they won't pass the next filter
    df_corr.values[[np.arange(df_corr.shape[0])] * 2] = 0

    # Only pass correlations with an absolute value equal to or above the threshold (default threshold = 0.7)
    df_corr = df_corr[(df_corr <= -threshold) | (df_corr >= threshold)]

    # Remove all rows and columns where, after applying the previous filter, there are no more values inside the matrix
    df_corr = df_corr.dropna(axis=0, how="all")
    df_corr = df_corr.dropna(axis=1, how="all")

    return df_corr


def PCA_analyse(df, features, label):

    # Center and scale the data
    scaled_data = preprocessing.scale(df[features])

    # Perform PCA on the data
    pca = PCA()  # create a PCA object
    pca.fit(scaled_data)  # do the math
    pca_data = pca.transform(scaled_data)  # get PCA coordinates for scaled_data

    # Create dataframes with PC information (needed for PC1 - PC2 plots and Scree plot)
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = ["PC" + str(x) for x in range(1, len(per_var) + 1)]
    pca_df = pd.DataFrame(pca_data, columns=labels)

    # Add original labels (targets) to the dataframe
    pca_df = pca_df.join(df[[label]])

    # Show how every datapunt is expressed in terms of PC coordinates
    display(pca_df.head())
    display(pca_df.tail())

    # Create a dataframe with PC1 and PC2 loading scores for the features (needed for loading scores plots)
    pca_loading_scores = pd.DataFrame()
    pca_loading_scores["feature"] = df[features].columns.tolist()
    pca_loading_scores["PC1 Loading Score"] = pca.components_[0]
    pca_loading_scores["PC2 Loading Score"] = pca.components_[1]
    pca_loading_scores = pca_loading_scores.sort_values(
        by=["PC1 Loading Score", "PC2 Loading Score"], ascending=True
    )

    # PC1 - PC2 plot
    plt.figure(figsize=(18, 9))
    sns.scatterplot(x=pca_df.PC1, y=pca_df.PC2)
    plt.title("PC1 - PC2 plot")
    plt.xlabel("PC1 - {0}%".format(per_var[0]))
    plt.ylabel("PC2 - {0}%".format(per_var[1]))
    plt.title("PC1 - PC2 plot")
    plt.xlim(np.floor(np.min(pca_df.PC1)), np.ceil(np.max(pca_df.PC1)))
    plt.ylim(np.floor(np.min(pca_df.PC2)), np.ceil(np.max(pca_df.PC2)))
    plt.grid(True)
    plt.show()

    # Scree plot
    plt.figure(figsize=(18, 9))
    plt.bar(range(1, len(per_var) + 1), per_var, tick_label=labels)
    plt.xlabel("Principal Component")
    plt.ylabel("Percentage of Explained Variance")
    plt.title("Scree Plot")
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.show()

    # PC1 - Loading Scores
    height_factor = len(pca_loading_scores["PC1 Loading Score"]) / 100.0
    plt.figure(figsize=(18, 40 * height_factor))
    plt.subplot(221)
    plt.barh(
        range(1, len(features) + 1),
        pca_loading_scores["PC1 Loading Score"],
        tick_label=pca_loading_scores["feature"].tolist(),
    )
    plt.xlabel("Loading Score")
    plt.ylabel("Feature")
    plt.title("PC1 - Loading Scores")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0, fontsize=10)
    plt.grid(True)

    # PC2 - Loading Scores
    plt.subplot(222)
    plt.barh(
        range(1, len(features) + 1),
        pca_loading_scores["PC2 Loading Score"],
        tick_label=pca_loading_scores["feature"].tolist(),
    )
    plt.xlabel("Loading Score")
    plt.title("PC2 - Loading Scores")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0, fontsize=0)
    plt.grid(True)
    plt.show()

    return


# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble
# https://scikit-learn.org/stable/modules/ensemble.html
def voting_classifier(models, voting="hard", scoring="accuracy"):

    # Initialize the lists with estimators and their names
    estimator_list = []
    estimator_list_VC = []
    estimator_names = []

    # Fill the lists with estimators and their names
    for i in range(0, len(models)):
        exec("clf" + str(i) + " = models[" + str(i) + "].model.algorithm")
        exec("estimator_list = estimator_list.append(clf" + str(i) + ")")
        exec(
            "estimator_list_VC = estimator_list_VC.append(('Estimator "
            + str(i + 1)
            + "', clf"
            + str(i)
            + "))"
        )
        exec("estimator_names = estimator_names.append('Estimator " + str(i + 1) + "')")

    # Define the ensamble classifier and add it to the lists with estimators and their names
    eclf = VotingClassifier(estimators=estimator_list_VC, voting=voting)
    exec("estimator_list = estimator_list.append(eclf)")
    estimator_names = estimator_names + ["Ensemble estimator"]

    # Extract the data to be modeled
    X = models[0].data.df[models[0].data.features].values.tolist()
    y = models[0].data.df[models[0].data.label].values.ravel().tolist()

    # Print the scores of the estimators, including errors
    for clf, label in zip(estimator_list, estimator_names):
        scores = cross_val_score(clf, X, y, cv=5, scoring=scoring)
        print(
            scoring + ": %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label)
        )

    return


# Function: zero_variance_columns
#
# Author: Rien Dijkstra
#
# Source:
#
# Syntax: result = zero_variance_columns(df)
#
# Purpose: For a given dataframe, create a list of names of the
#          columns that have zero variance (as measured by the
#          standard deviation) in case of numeric columns and
#          columns that only contain one single value in case
#          of text columns.
#
# Inputs: df = dataframe
#
# Outputs: result = list of names of columns in the dataframe that have zero variance
#                   in case of numeric columns or only one single value in case of
#                   text columns
#
# Example: Create a list of the names of the zero variance numeric (single valued text)
#          columns in dataframe df
#
#          result = zero_variance_columns(df)
#
# Dependencies: None
#
def zero_variance_columns(df):

    # Store the numeric and text columns in two separate dataframes
    df_numeric = df.select_dtypes(exclude=["object"])
    df_text = df.select_dtypes(include=["object"])

    # Get summary statistics of the data
    summary_numeric = df_numeric.describe()
    summary_text = df_text.describe()

    # Select record with standard deviations (std) for numeric
    # columns or number of unique values (unq) for text columns
    std = summary_numeric[summary_numeric.index == "std"]
    uniq = summary_text[summary_text.index == "unique"]

    # Transpose the record with standard deviations or unique values
    # into a new column
    std_T = std.T
    uniq_T = uniq.T

    # Select the records from the new column where the value of std = 0 or uniq = 1
    # These records represent numeric dataframe columns with zero variance, i.e.
    # constants, or text dataframe columns with only one single value
    std_T_zero = std_T[std_T["std"] == 0]
    uniq_T_one = uniq_T[uniq_T["unique"] == 1]

    # Retrieve the names of the columns that have zero variance
    # or only single values
    std_T_zero_names = std_T_zero.index.tolist()
    uniq_T_one_names = uniq_T_one.index.tolist()
    names = std_T_zero_names + uniq_T_one_names

    return names
