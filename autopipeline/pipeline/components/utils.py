import numpy as np

def stack_Xs(X_train, X_valid=None, X_test=None):
    Xs=[X_train]
    if X_valid is not None:
        Xs.append(X_valid)
    if X_test is not None:
        Xs.append(X_test)
    return np.vstack(Xs)