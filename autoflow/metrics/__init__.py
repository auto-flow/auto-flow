import copy
from abc import ABCMeta, abstractmethod
from functools import partial
from typing import List

import numpy as np
import pandas as pd
import scipy.stats
import sklearn.metrics
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import type_of_target

from autoflow.metrics import classification_metrics
from autoflow.utils.array_ import sanitize_array
from autoflow.utils.ml_task import MLTask


class Scorer(object, metaclass=ABCMeta):
    def __init__(self, name, score_func, optimum, sign, kwargs):
        self.name = name
        self._kwargs = kwargs
        self._score_func = score_func
        self._optimum = optimum
        self._sign = sign
        self.score = None

    @abstractmethod
    def __call__(self, y_true, y_pred, sample_weight=None):
        pass

    def __repr__(self):
        return self.name


class _PredictScorer(Scorer):
    def __call__(self, y_true, y_pred, sample_weight=None):
        """Evaluate predicted target values for X relative to y_true.

        Parameters
        ----------
        y_true : array-like
            Gold standard target values for X.

        y_pred : array-like, [n_samples x n_classes]
            Model predictions

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of component on X.
        """
        type_true = type_of_target(y_true)
        if len(y_pred.shape) == 1 or y_pred.shape[1] == 1 or \
                type_true == 'continuous':
            # must be regression, all other ml_task types would return at least
            # two probabilities
            pass
        elif type_true in ['binary', 'multiclass']:
            y_pred = np.argmax(y_pred, axis=1)
        elif type_true == 'multilabel-indicator':
            y_pred[y_pred > 0.5] = 1.0
            y_pred[y_pred <= 0.5] = 0.0
        else:
            raise ValueError(type_true)

        if sample_weight is not None:
            self.score = self._score_func(y_true, y_pred,
                                          sample_weight=sample_weight,
                                          **self._kwargs)
        else:
            self.score = self._score_func(y_true, y_pred,
                                          **self._kwargs)
        return self._sign * self.score


class _ProbaScorer(Scorer):
    def __call__(self, y_true, y_pred, sample_weight=None):
        """Evaluate predicted probabilities for X relative to y_true.
        Parameters
        ----------
        y_true : array-like
            Gold standard target values for X. These must be class labels,
            not probabilities.

        y_pred : array-like, [n_samples x n_classes]
            Model predictions

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of component on X.
        """
        if sample_weight is not None:
            self.score = self._score_func(y_true, y_pred,
                                          sample_weight=sample_weight,
                                          **self._kwargs)
        else:
            self.score = self._score_func(y_true, y_pred,
                                          **self._kwargs)
        return self._sign * self.score


class _ThresholdScorer(Scorer):
    def __call__(self, y_true, y_pred, sample_weight=None):
        """Evaluate decision function output for X relative to y_true.
        Parameters
        ----------
        y_true : array-like
            Gold standard target values for X. These must be class labels,
            not probabilities.

        y_pred : array-like, [n_samples x n_classes]
            Model predictions

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of component on X.
        """
        y_type = type_of_target(y_true)
        if y_type not in ("binary", "multilabel-indicator"):
            raise ValueError("{0} format is not supported".format(y_type))

        if y_type == "binary":
            y_pred = y_pred[:, 1]
        elif isinstance(y_pred, list):
            y_pred = np.vstack([p[:, -1] for p in y_pred]).T

        if sample_weight is not None:
            self.score = self._score_func(y_true, y_pred,
                                          sample_weight=sample_weight,
                                          **self._kwargs)
        else:
            self.score = self._score_func(y_true, y_pred,
                                          **self._kwargs)
        return self._sign * self.score


def make_scorer(name, score_func, optimum=1, greater_is_better=True,
                needs_proba=False, needs_threshold=False, **kwargs):
    """Make a scorer from a performance metric or loss function.

    Factory inspired by scikit-learn which wraps scikit-learn scoring functions
    to be used in auto-sklearn.

    Parameters
    ----------
    score_func : callable
        Score function (or loss function) with signature
        ``score_func(y, y_pred, **kwargs)``.

    optimum : int or float, default=1
        The best score achievable by the score function, i.e. maximum in case of
        scorer function and minimum in case of loss function.

    greater_is_better : boolean, default=True
        Whether score_func is a score function (default), meaning high is good,
        or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the score_func.

    needs_proba : boolean, default=False
        Whether score_func requires predict_proba to get probability estimates
        out of a classifier.

    needs_threshold : boolean, default=False
        Whether score_func takes a continuous decision certainty.
        This only works for binary classification.

    **kwargs : additional arguments
        Additional parameters to be passed to score_func.

    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score; greater is better.
    """
    sign = 1 if greater_is_better else -1
    if needs_proba:
        cls = _ProbaScorer
    elif needs_threshold:
        cls = _ThresholdScorer
    else:
        cls = _PredictScorer
    return cls(name, score_func, optimum, sign, kwargs)


# Standard regression scores
r2 = make_scorer('r2',
                 sklearn.metrics.r2_score)

pearsonr = make_scorer(
    'pearsonr',
    lambda x, y: scipy.stats.pearsonr(x, y)[0]
)
spearmanr = make_scorer(
    'spearmanr',
    lambda x, y: scipy.stats.spearmanr(x, y)[0]
)
kendalltau = make_scorer(
    'kendalltau',
    lambda x, y: scipy.stats.kendalltau(x, y)[0]
)
mean_squared_error = make_scorer('mean_squared_error',
                                 sklearn.metrics.mean_squared_error,
                                 optimum=0,
                                 greater_is_better=False)
mean_absolute_error = make_scorer('mean_absolute_error',
                                  sklearn.metrics.mean_absolute_error,
                                  optimum=0,
                                  greater_is_better=False)
median_absolute_error = make_scorer('median_absolute_error',
                                    sklearn.metrics.median_absolute_error,
                                    optimum=0,
                                    greater_is_better=False)

# Standard Classification Scores
accuracy = make_scorer('accuracy',
                       sklearn.metrics.accuracy_score)
mcc = make_scorer('mcc', sklearn.metrics.matthews_corrcoef)
sensitivity = make_scorer("sensitivity", classification_metrics.sensitivity)
specificity = make_scorer("specificity", classification_metrics.specificity)
balanced_accuracy = make_scorer('balanced_accuracy',
                                classification_metrics.balanced_accuracy)
f1 = make_scorer('f1',
                 sklearn.metrics.f1_score)

# Score functions that need decision values

average_precision = make_scorer('average_precision',
                                sklearn.metrics.average_precision_score,
                                needs_threshold=True)
precision = make_scorer('precision',
                        sklearn.metrics.precision_score)
recall = make_scorer('recall',
                     sklearn.metrics.recall_score)

# Score function for probabilistic classification
log_loss = make_scorer('log_loss',
                       sklearn.metrics.log_loss,
                       optimum=0,
                       greater_is_better=False,
                       needs_proba=True)
pac_score = make_scorer('pac_score',
                        classification_metrics.pac_score,
                        greater_is_better=True,
                        needs_proba=True)
# TODO cohen_kappa


REGRESSION_METRICS = dict()
for scorer in [
    r2, mean_squared_error, mean_absolute_error,
    median_absolute_error, pearsonr, spearmanr, kendalltau
]:
    REGRESSION_METRICS[scorer.name] = scorer

CLASSIFICATION_METRICS = dict()

for scorer in [
    accuracy, average_precision, log_loss,
    balanced_accuracy, pac_score, mcc, sensitivity, specificity
]:
    CLASSIFICATION_METRICS[scorer.name] = scorer

for multi_class in ["ovo", "ovr"]:
    for average in ['macro', 'micro', 'samples', 'weighted']:
        qualified_name = '{0}_{1}_{2}'.format("roc_auc", multi_class, average)
        globals()[qualified_name] = make_scorer(qualified_name,
                                                partial(sklearn.metrics.roc_auc_score,
                                                        multi_class=multi_class,
                                                        average=average), needs_proba=True)
        CLASSIFICATION_METRICS[qualified_name] = globals()[qualified_name]

for name, metric in [
    ('precision', sklearn.metrics.precision_score),
    ('recall', sklearn.metrics.recall_score),
    ('f1', sklearn.metrics.f1_score),
    ('roc_auc', sklearn.metrics.roc_auc_score)
]:
    globals()[name] = make_scorer(name, metric)
    CLASSIFICATION_METRICS[name] = globals()[name]
    for average in ['macro', 'micro', 'samples', 'weighted']:
        qualified_name = '{0}_{1}'.format(name, average)
        if name == "roc_auc":
            globals()[qualified_name] = make_scorer(qualified_name,
                                                    partial(metric, average=average),
                                                    needs_threshold=True)
        else:
            globals()[qualified_name] = make_scorer(qualified_name,
                                                    partial(metric,
                                                            average=average))
        CLASSIFICATION_METRICS[qualified_name] = globals()[qualified_name]


def calculate_score(solution, prediction, ml_task: MLTask, metric,
                    should_calc_all_metric=False):
    if isinstance(solution, (pd.Series, pd.DataFrame)):
        solution = solution.values
    if should_calc_all_metric:
        score = dict()
        true_score = {}
        if ml_task.mainTask == "regression":
            # TODO put this into the regression metric itself
            cprediction = sanitize_array(prediction)
            metric_dict = copy.copy(REGRESSION_METRICS)
            metric_dict[metric.name] = metric
            for metric_ in REGRESSION_METRICS:
                func: Scorer = REGRESSION_METRICS[metric_]
                score[func.name] = func(solution, cprediction)
                true_score[func.name] = func.score

        else:
            metric_dict = copy.copy(CLASSIFICATION_METRICS)
            metric_dict[metric.name] = metric
            for metric_ in metric_dict:
                func = CLASSIFICATION_METRICS[metric_]

                # TODO maybe annotate metrics to define which cases they can
                # handle?

                try:
                    score[func.name] = float(func(solution, prediction))
                    true_score[func.name] = func.score
                except ValueError as e:
                    if e.args[0] == 'multiclass format is not supported':
                        continue
                    elif e.args[0] == "Samplewise metrics are not available " \
                                      "outside of multilabel classification.":
                        continue
                    elif e.args[0] == "Target is multiclass but " \
                                      "average='binary'. Please choose another average " \
                                      "setting, one of [None, 'micro', 'macro', 'weighted'].":
                        continue
                    # else:
                    #     raise e

    else:
        if ml_task.mainTask == "regression":
            # TODO put this into the regression metric itself
            cprediction = sanitize_array(prediction)
            score = metric(solution, cprediction)
        else:
            score = metric(solution, prediction)

    return score, true_score


def calculate_confusion_matrix(y_true, y_pred) -> List[List[int]]:
    # return 2d list
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    return confusion_matrix(y_true, y_pred).tolist()
