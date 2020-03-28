from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm


class BaseEncoder(AutoPLPreprocessingAlgorithm):

    def _transform_proc(self, X):
        if X is None:
            return None
        else:
            trans = self.estimator.transform(X)
            trans[X == -999] = -999
            return trans
