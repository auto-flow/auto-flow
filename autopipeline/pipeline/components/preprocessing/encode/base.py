from autopipeline.pipeline.components.feature_engineer_base import AutoPLFeatureEngineerAlgorithm


class BaseEncoder(AutoPLFeatureEngineerAlgorithm):

    def _transform_proc(self, X):
        if X is None:
            return None
        else:
            trans = self.estimator.transform(X)
            trans[X == -999] = -999
            return trans
