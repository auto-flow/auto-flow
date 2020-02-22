
from autopipeline.pipeline.components.base import AutoPLClassificationAlgorithm


class GaussianNB(AutoPLClassificationAlgorithm):
    class__ = "GaussianNB"
    module__ = "sklearn.naive_bayes"
    OVR__ = True

