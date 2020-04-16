from autoflow.pipeline.components.classification_base import AutoFlowClassificationAlgorithm

__all__=["RandomForest"]

class RandomForest(AutoFlowClassificationAlgorithm):
    class__ = "RandomForestClassifier"
    module__ = "sklearn.ensemble"