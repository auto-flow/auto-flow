from autoflow.pipeline.components.classification_base import AutoFlowClassificationAlgorithm

__all__=["LogisticRegression"]

class LogisticRegression(AutoFlowClassificationAlgorithm):
    class__ = "LogisticRegression"
    module__ = "sklearn.linear_model"
