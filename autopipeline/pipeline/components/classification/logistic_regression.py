from autopipeline.pipeline.components.classification_base import AutoPLClassificationAlgorithm

__all__=["LogisticRegression"]

class LogisticRegression(AutoPLClassificationAlgorithm):
    class__ = "LogisticRegression"
    module__ = "sklearn.linear_model"
