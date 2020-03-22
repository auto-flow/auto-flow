from sklearn import clone
from sklearn.pipeline import Pipeline
from sklearn.utils import _print_elapsed_time
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import check_memory

from autopipeline.constants import Task


def _fit_transform_one(transformer,
                       X_train, y_train, X_valid=None, y_valid=None, X_test=None, y_test=None, is_train=False,
                       message_clsname='',
                       message=None,
                       **fit_params):
    """
    Fits ``transformer`` to ``X`` and ``y``. The transformed result is returned
    with the fitted transformer. If ``weight`` is not ``None``, the result will
    be multiplied by ``weight``.
    """
    with _print_elapsed_time(message_clsname, message):
        if hasattr(transformer, 'fit_transform'):
            res = transformer.fit_transform(X_train, y_train, X_valid, y_valid, X_test, y_test, is_train)
        else:
            res = transformer.fit(X_train, y_train, X_valid, y_valid, X_test, y_test). \
                transform(X_train, X_valid, X_test, is_train)

    return res, transformer


class GenericPipeline(Pipeline):
    # 可以当做Transformer，又可以当做estimator！
    # todo: 适配当做普通Pipeline的情况
    def _fit(self, X_train, y_train, X_valid=None, y_valid=None, X_test=None, y_test=None):
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

            ret, fitted_transformer = fit_transform_one_cached(
                cloned_transformer, X_train, y_train, X_valid, y_valid, X_test, y_test, True,
                message_clsname='Pipeline',
                message=self._log_message(step_idx))
            X_train = ret["X_train"]
            X_valid = ret.get("X_valid")
            X_test = ret.get("X_test")
            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            self.steps[step_idx] = (name, fitted_transformer)
        if self._final_estimator == 'passthrough':
            return X_train
        return {"X_train": X_train, "X_valid": X_valid, "X_test": X_test}

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_test=None, y_test=None):
        ret = self._fit(X_train, y_train, X_valid, y_valid, X_test, y_test)
        X_train = ret["X_train"]
        X_valid = ret.get("X_valid")
        X_test = ret.get("X_test")
        self.last_data = ret
        with _print_elapsed_time('Pipeline',
                                 self._log_message(len(self.steps) - 1)):
            if self._final_estimator != 'passthrough':
                self._final_estimator.fit(X_train, y_train, X_valid, y_valid, X_test, y_test)
        return self

    def procedure(self, task: Task, X_train, y_train, X_valid=None, y_valid=None, X_test=None, y_test=None):
        self.fit(X_train, y_train, X_valid, y_valid, X_test, y_test)
        X_train = self.last_data["X_train"]
        X_valid = self.last_data.get("X_valid")
        X_test = self.last_data.get("X_test")
        self.last_data = None  # GC
        if task.mainTask == "classification":
            pred_valid = self._final_estimator.predict_proba(X_valid)
            pred_test = self._final_estimator.predict_proba(X_test) if X_test is not None else None
        else:
            pred_valid = self._final_estimator.predict(X_valid)
            pred_test = self._final_estimator.predict(X_test) if X_test is not None else None
        return {
            "pred_valid":pred_valid,
            "pred_test":pred_test,
        }

    def transform(self, X_train, X_valid=None, X_test=None, is_train=False,
                  with_final=True):
        for _, _, transform in self._iter(with_final=with_final):
            ret = transform.transform(X_train, X_valid, X_test, is_train)  # predict procedure
            X_train = ret["X_train"]
            X_valid = ret.get("X_valid")
            X_test = ret.get("X_test")
        return {"X_train": X_train, "X_valid": X_valid, "X_test": X_test}

    @if_delegate_has_method(delegate='_final_estimator')
    def predict(self, X):
        ret = self.transform(X, is_train=False, with_final=False)
        X = ret["X_train"]
        return self.steps[-1][-1].predict(X)

    @if_delegate_has_method(delegate='_final_estimator')
    def predict_proba(self, X):
        ret = self.transform(X, is_train=False, with_final=False)
        X = ret["X_train"]
        return self.steps[-1][-1].predict_proba(X)

    def fit_transform(self, X_train, y_train=None, X_valid=None, y_valid=None, X_test=None, y_test=None, is_train=False):
        last_step = self._final_estimator
        ret = self._fit(X_train, y_train, X_valid, y_valid, X_test, y_test)
        X_train = ret["X_train"]
        X_valid = ret.get("X_valid")
        X_test = ret.get("X_test")
        with _print_elapsed_time('Pipeline',
                                 self._log_message(len(self.steps) - 1)):
            if last_step == 'passthrough':
                return ret["X_train"]
            if hasattr(last_step, 'fit_transform'):
                return last_step.fit_transform(X_train, y_train, X_valid, y_valid, X_test, y_test)
            else:
                return last_step.fit(X_train, y_train, X_valid, y_valid, X_test, y_test).transform(X_train, y_train,
                                                                                                   X_valid, y_valid,
                                                                                                   X_test, y_test, True)
