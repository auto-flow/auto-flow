from autopipeline.pipeline.components.classification_base import AutoPLClassificationAlgorithm

__all__=["RandomForest"]

class RandomForest(AutoPLClassificationAlgorithm):
    class__ = "RandomForestClassifier"
    module__ = "sklearn.ensemble"