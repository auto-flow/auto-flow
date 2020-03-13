from scipy.sparse import issparse

from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm

__all__=["OneHotEncoder"]
class OneHotEncoder(AutoPLPreprocessingAlgorithm):
    class__ = "OneHotEncoder"
    module__ = "sklearn.preprocessing"

    def after_pred_X(self, X):
        # todo: 考虑在后续模型中支持系数矩阵
        # todo: 增加encoding方法
        if issparse(X):
            return X.todense().getA()
        else:
            return X