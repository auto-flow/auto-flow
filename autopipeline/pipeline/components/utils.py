import numpy as np

def stack_Xs(X_train, X_valid=None, X_test=None):
    Xs=[X_train]
    if X_valid is not None:
        Xs.append(X_valid)
    if X_test is not None:
        Xs.append(X_test)
    return np.vstack(Xs)

def get_categorical_features_indices(origin_grp):
    categorical_features_indices = []
    for i, elem in origin_grp:
        if "cat" in elem:
            categorical_features_indices.append(i)
    return categorical_features_indices