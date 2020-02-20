from autopipeline.pipeline.components.base import AutoPLClassificationAlgorithm


class RandomForest(AutoPLClassificationAlgorithm):
    class__ = "RandomForestClassifier"
    module__ = "sklearn.ensemble"