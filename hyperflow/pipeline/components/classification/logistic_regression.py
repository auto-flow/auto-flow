from hyperflow.pipeline.components.classification_base import HyperFlowClassificationAlgorithm

__all__=["LogisticRegression"]

class LogisticRegression(HyperFlowClassificationAlgorithm):
    class__ = "LogisticRegression"
    module__ = "sklearn.linear_model"
