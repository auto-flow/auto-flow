import numpy as np

from autopipeline.pipeline.components.base import AutoPLClassificationAlgorithm


class BernoulliNB(AutoPLClassificationAlgorithm):
    class__ = "BernoulliNB"
    module__ = "sklearn.naive_bayes"
    OVR__ = True
