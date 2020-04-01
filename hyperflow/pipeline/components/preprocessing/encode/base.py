from hyperflow.pipeline.components.feature_engineer_base import HyperFlowFeatureEngineerAlgorithm


class BaseEncoder(HyperFlowFeatureEngineerAlgorithm):

    def _transform_proc(self, X):
        if X is None:
            return None
        else:
            trans = self.estimator.transform(X)
            trans[X == -999] = -999
            return trans
