import pandas as pd
import numpy as np
import Helper
import time

from sklearn.model_selection import KFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
)

np.random.seed(231)


def preprocess(X_train, X_test):
    """
    This method will preprocess the dataset based on X_train.
    The prepocess must follow the order:
    - Converting from categorical attributes to numeric attributes
    - Handle missing values
    - Standardising and/or Normalising the data
    :param X_train: Training set, must be Panda dataframe
    :param X_test: Test set, must be Panda dataframe
    :return: preprocessed X_train and X_test, must be Panda dataframe
    """
    # Define ordinal and nominal features
    ordinal_cols = ["Att1", "Att3", "Att6", "Att7"]
    nominal_cols = [
        col
        for col in X_train.columns
        if X_train[col].dtype == "object" and col not in ordinal_cols
    ]

    # Define the order for each ordinal attribute
    order_A1 = ["A14", "A11", "A12", "A13"]  # From no checking account to high balance
    order_A3 = [
        "A30",
        "A33",
        "A31",
        "A32",
        "A34",
    ]  # From no credits to critical account
    order_A6 = ["A65", "A61", "A62", "A63", "A64"]  # From no savings to high savings
    order_A7 = [
        "A71",
        "A72",
        "A73",
        "A74",
        "A75",
    ]  # From unemployed to long-term employment

    # Initialize OrdinalEncoder with these categories
    ordinal_encoder = OrdinalEncoder(
        categories=[order_A1, order_A3, order_A6, order_A7]
    )

    # Handle ordinal columns, first impute and then encode
    ordinal_imputer = SimpleImputer(strategy="most_frequent")
    X_train[ordinal_cols] = ordinal_imputer.fit_transform(X_train[ordinal_cols])
    X_test[ordinal_cols] = ordinal_imputer.transform(X_test[ordinal_cols])

    # Apply the encoder
    X_train_ordinal = ordinal_encoder.fit_transform(X_train[ordinal_cols])
    X_test_ordinal = ordinal_encoder.transform(X_test[ordinal_cols])

    # Handle nominal columns with one-hot encoding
    onehot_encoder = OneHotEncoder(drop="first", sparse_output=False)
    X_train_nominal = onehot_encoder.fit_transform(X_train[nominal_cols])
    X_test_nominal = onehot_encoder.transform(X_test[nominal_cols])

    # Identify numeric columns and impute missing values
    numeric_cols = [
        col
        for col in X_train.columns
        if X_train[col].dtype in ["int64", "float64"] and col not in ordinal_cols
    ]
    numeric_imputer = SimpleImputer(strategy="median")
    X_train_numeric = numeric_imputer.fit_transform(X_train[numeric_cols])
    X_test_numeric = numeric_imputer.transform(X_test[numeric_cols])

    # Normalize numeric columns
    scaler = StandardScaler()
    X_train_numeric = scaler.fit_transform(X_train_numeric)
    X_test_numeric = scaler.transform(X_test_numeric)

    # Concatenate all processed columns
    X_train_processed = np.concatenate(
        [X_train_numeric, X_train_nominal, X_train_ordinal], axis=1
    )
    X_test_processed = np.concatenate(
        [X_test_numeric, X_test_nominal, X_test_ordinal], axis=1
    )

    # Convert processed data back to DataFrame and handle columns names for clarity
    new_column_names = (
        numeric_cols
        + list(onehot_encoder.get_feature_names_out(nominal_cols))
        + ordinal_cols
    )

    return pd.DataFrame(X_train_processed, columns=new_column_names), pd.DataFrame(
        X_test_processed, columns=new_column_names
    )


def feature_ranking(X_train, y_train, no_features=5):
    """
    Rank features based on the mutual information between each feature and the class label.
    Step 1: Calculate the mutual information between each feature and the class label.
    Step 2: Sort features based on their mutual information.
    Step 3: Get top "no_features" features.

    :param X_train: numpy array
    :param y_train: numpy array
    :param no_features: int
    :return: selected features indices
    """
    # Step 1: Compute mutual information scores
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)

    # Step 2: Sort features based on their mutual information scores in descending order
    sorted_indices = np.argsort(mi_scores)[::-1]

    # Step 3: Select the indices of the top "no_features"
    top_features_indices = sorted_indices[:no_features]

    return top_features_indices


def sequential_score(clf, X, y, S, n_folds=10):
    """
    Given the training set (X, y), we would like to evaluate the performance of the feature subset S
    1. Extract part of the training set containing only S
    1. Divide the dataset into n_folds
    2. for each fold
        - use that fold as the test set and all other folds as the training set
        - train clf on the training set and test it on the test set
    3. calculate and return the average performance across n_folds
    :param clf: a classification algorithm
    :param X: feature matrix, must be numpy array
    :param y: label vector, must be numpy array
    :param S: subset of selected features, must be numpy array
    :param n_folds: number of folds, default is 10
    :return:
    """
    split = KFold(n_splits=n_folds)
    X_sel = X[:, S]
    scores = []

    for train_idx, test_idx in split.split(X_sel):
        X_train, X_test = X_sel[train_idx], X_sel[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(X_train, y_train)  # Train the classifier
        score = clf.score(X_test, y_test)  # Evaluate the classifier
        scores.append(score)

    return np.mean(scores)  # Return the average score across all folds


def sequential_feature_selection(X, y, no_features=5):
    """
    1. Initialise subset of selected features to an empty set
    2. Initialise subset of remaining features to the original set of features
    3. Repeat the following steps until a predefined number of features (no_features) is selected:
        - For each remaining feature
            + form a new feature subset consisting of the remaining feature and the subset of selected features
            + extract a feature matrix containing only the new feature subset
            + use the newly extracted feature matrix in using sequential_score() to get the learning performance (score)
            + record the best remaining feature which results in the highest score
        - Add the best remaining feature to the subset of selected features
        - Remove the best remaining feature from the subset of remaining features
    4. Return the subset of selected features
    :param X: training feature matrix X, must be numpy array
    :param y: training label vector y, must be numpy array
    :param no_features: number of selected feature
    :return: subset of selected features
    """
    # classifier to be used in sequential_score()
    clf = KNN(n_neighbors=3)

    # initialise the subset of selected features and the subset of remaining features
    selected_features = np.array([], dtype=int)
    remaining_features = np.arange(0, X.shape[1])

    # Repeat until the desired number of features is selected
    for _ in range(no_features):
        best_score = -np.inf
        best_feature = None

        # Evaluate each remaining feature
        for feature in remaining_features:
            current_features = np.append(selected_features, feature)

            score = sequential_score(clf, X, y, current_features)

            if score > best_score:
                best_score = score
                best_feature = feature

        if best_feature is not None:
            selected_features = np.append(selected_features, best_feature)
            remaining_features = np.delete(
                remaining_features, np.where(remaining_features == best_feature)
            )

    return selected_features


if __name__ == "__main__":
    # load data to data frame
    df = pd.read_csv("Data.csv")
    X, y = df.drop("Class", axis=1), df["Class"]

    # splitting dataset into training and test set, only training set has missing values
    X_train, X_test, y_train, y_test = Helper.special_split(X, y, ratio=0.7)

    # Preprocess the data
    X_train, X_test = preprocess(X_train, X_test)

    # Converting from data frame to numpy for the selection tasks
    X_train, X_test = X_train.values, X_test.values
    y_train, y_test = y_train.values, y_test.values
    clf = KNN(n_neighbors=3)

    # performance of using all features
    print("****************All features********************\n")
    all_acc = Helper.evaluation(clf, X_train, y_train, X_test, y_test)
    print("Accuracy of using all features: %.2f\n" % (all_acc * 100))

    # Feature ranking
    start = time.time()
    print("****************Feature ranking********************")
    top_features = feature_ranking(X_train, y_train, no_features=5)
    rank_acc = Helper.evaluation(
        clf, X_train[:, top_features], y_train, X_test[:, top_features], y_test
    )
    print(
        "Top 5 ranking features: %s "
        % ("[" + ", ".join([str(val) for val in top_features]) + "]")
    )
    print("Accuracy of using top five features: %.2f" % (rank_acc * 100))
    end = time.time()
    print("Computational time for feature ranking: %.10f seconds\n" % ((end - start)))

    # Sequential Feature selection
    start = time.time()
    print("****************Sequential Forward Feature Selection********************")
    sel_features = sequential_feature_selection(X_train, y_train, no_features=5)
    sffs_acc = Helper.evaluation(
        clf, X_train[:, sel_features], y_train, X_test[:, sel_features], y_test
    )
    print(
        "5 features selected by SFFS: %s"
        % ("[" + ", ".join([str(val) for val in sel_features]) + "]")
    )
    print("Accuracy of SFFS: %.2f" % (sffs_acc * 100))
    end = time.time()
    print("Computational time for SFFS: %.10f seconds" % ((end - start)))
