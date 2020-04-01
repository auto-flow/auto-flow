from hyperflow.pipeline.components.classification_base import HyperFlowClassificationAlgorithm

__all__=["RandomForest"]

class RandomForest(HyperFlowClassificationAlgorithm):
    class__ = "RandomForestClassifier"
    module__ = "sklearn.ensemble"