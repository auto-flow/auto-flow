from hyperflow.pipeline.components.feature_engineer_base import HyperFlowFeatureEngineerAlgorithm
import pandas as pd

class BaseEncoder(HyperFlowFeatureEngineerAlgorithm):

    def _transform_proc(self, X):
        if X is None:
            return None
        else:
            trans = self.estimator.transform(X)
            trans[pd.DataFrame(X) == -999] = -999  # todo: 有没有更优化的解决办法
            return trans
