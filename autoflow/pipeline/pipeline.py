from copy import deepcopy

from sklearn import clone
from sklearn.pipeline import Pipeline
from sklearn.utils import _print_elapsed_time
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import check_memory

from autoflow.utils.ml_task import MLTask


def _fit_transform_one(transformer,
                       X_train, y_train, X_valid=None, y_valid=None, X_test=None, y_test=None, resource_manager=None,
                       message_clsname='',
                       message=None):
    """
    Fits ``transformer`` to ``X`` and ``y``. The transformed result is returned
    with the fitted transformer. If ``weight`` is not ``None``, the result will
    be multiplied by ``weight``.
    """
    transformer.resource_manager = resource_manager
    with _print_elapsed_time(message_clsname, message):
        if hasattr(transformer, 'fit_transform'):
            result = transformer.fit_transform(X_train, y_train, X_valid, y_valid, X_test, y_test)
        else:
            result = transformer.fit(X_train, y_train, X_valid, y_valid, X_test, y_test). \
                transform(X_train, X_valid, X_test, y_train)
    transformer.resource_manager = None
    return result, transformer


class GenericPipeline(Pipeline):
    # 可以当做Transformer，又可以当做estimator！
    resource_manager = None

    # todo: 适配当做普通Pipeline的情况
    def _fit(self, X_train, y_train, X_valid=None, y_valid=None, X_test=None, y_test=None, intermediate_result=None):
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)
        for (step_idx,
             name,
             transformer) in self._iter(with_final=False,
                                        filter_passthrough=False):
            if (transformer is None or transformer == 'passthrough'):
                continue

            if hasattr(memory, 'location'):
                # joblib >= 0.12
                if memory.location is None:
                    # we do not clone when caching is disabled to
                    # preserve backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)
            elif hasattr(memory, 'cachedir'):
                # joblib < 0.11
                if memory.cachedir is None:
                    # we do not clone when caching is disabled to
                    # preserve backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)
            else:
                cloned_transformer = clone(transformer)
            # Fit or load from cache the current transformer

            result, fitted_transformer = fit_transform_one_cached(
                cloned_transformer, X_train, y_train, X_valid, y_valid, X_test, y_test, self.resource_manager,
                message_clsname='Pipeline',
                message=self._log_message(step_idx))
            X_train = result["X_train"]
            X_valid = result.get("X_valid")
            X_test = result.get("X_test")
            y_train = result.get("y_train")
            if intermediate_result is not None and isinstance(intermediate_result,list):
                intermediate_result.append({
                    "X_train":deepcopy(X_train),
                    "X_valid":deepcopy(X_valid),
                    "X_test":deepcopy(X_test),
                })
            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            self.steps[step_idx] = (name, fitted_transformer)

        return {"X_train": X_train, "X_valid": X_valid, "X_test": X_test, "y_train": y_train}

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_test=None, y_test=None, intermediate_result=None):
        result = self._fit(X_train, y_train, X_valid, y_valid, X_test, y_test, intermediate_result)
        X_train = result["X_train"]
        X_valid = result.get("X_valid")
        X_test = result.get("X_test")
        y_train = result.get("y_train")
        self.last_data = result
        with _print_elapsed_time('Pipeline',
                                 self._log_message(len(self.steps) - 1)):
            self._final_estimator.resource_manager = self.resource_manager
            self._final_estimator.fit(X_train, y_train, X_valid, y_valid, X_test, y_test)
            self._final_estimator.resource_manager = None
        return self

    def fit_transform(self, X_train, y_train=None, X_valid=None, y_valid=None, X_test=None, y_test=None,intermediate_result=None):
        return self.fit(X_train, y_train, X_valid, y_valid, X_test, y_test,intermediate_result).transform(X_train, X_valid, X_test, y_train)

    def procedure(self, ml_task: MLTask, X_train, y_train, X_valid=None, y_valid=None, X_test=None, y_test=None,
                  resource_manager=None,intermediate_result=None):
        self.resource_manager = resource_manager
        self.fit(X_train, y_train, X_valid, y_valid, X_test, y_test,intermediate_result)
        X_train = self.last_data["X_train"]
        y_train = self.last_data["y_train"]
        X_valid = self.last_data.get("X_valid")
        X_test = self.last_data.get("X_test")
        self.last_data = None  # GC
        if ml_task.mainTask == "classification":
            pred_valid = self._final_estimator.predict_proba(X_valid)
            pred_test = self._final_estimator.predict_proba(X_test) if X_test is not None else None
        else:
            pred_valid = self._final_estimator.predict(X_valid)
            pred_test = self._final_estimator.predict(X_test) if X_test is not None else None
        self.resource_manager = None
        return {
            "pred_valid": pred_valid,
            "pred_test": pred_test,
            "y_train": y_train  # todo: evaluator 中做相应的改变
        }

    def transform(self, X_train, X_valid=None, X_test=None, y_train=None,
                  with_final=True):
        for _, _, transformer in self._iter(with_final=with_final):
            result = transformer.transform(X_train, X_valid, X_test, y_train)  # predict procedure
            X_train = result["X_train"]
            X_valid = result.get("X_valid")
            X_test = result.get("X_test")
            y_train = result.get("y_train")
        return {"X_train": X_train, "X_valid": X_valid, "X_test": X_test, "y_train": y_train}

    @if_delegate_has_method(delegate='_final_estimator')
    def predict(self, X):
        result = self.transform(X, with_final=False)
        X = result["X_train"]
        return self.steps[-1][-1].predict(X)

    @if_delegate_has_method(delegate='_final_estimator')
    def predict_proba(self, X):
        result = self.transform(X, with_final=False)
        X = result["X_train"]
        return self.steps[-1][-1].predict_proba(X)
