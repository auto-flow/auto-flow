from autoflow.pipeline.components.classification_base import AutoFlowClassificationAlgorithm

__all__=["GaussianNB"]


class GaussianNB(AutoFlowClassificationAlgorithm):
    class__ = "GaussianNB"
    module__ = "sklearn.naive_bayes"
    OVR__ = True

