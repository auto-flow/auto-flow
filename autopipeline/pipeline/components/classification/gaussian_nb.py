from autopipeline.pipeline.components.classification_base import AutoPLClassificationAlgorithm

__all__=["GaussianNB"]


class GaussianNB(AutoPLClassificationAlgorithm):
    class__ = "GaussianNB"
    module__ = "sklearn.naive_bayes"
    OVR__ = True

