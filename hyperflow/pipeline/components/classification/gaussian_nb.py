from hyperflow.pipeline.components.classification_base import HyperFlowClassificationAlgorithm

__all__=["GaussianNB"]


class GaussianNB(HyperFlowClassificationAlgorithm):
    class__ = "GaussianNB"
    module__ = "sklearn.naive_bayes"
    OVR__ = True

