import numpy as np

from hyperflow.pipeline.components.classification_base import HyperFlowClassificationAlgorithm

__all__=["BernoulliNB"]


class BernoulliNB(HyperFlowClassificationAlgorithm):
    class__ = "BernoulliNB"
    module__ = "sklearn.naive_bayes"
    OVR__ = True
