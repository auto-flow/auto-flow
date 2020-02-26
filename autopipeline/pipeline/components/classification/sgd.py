import numpy as np

from autopipeline.pipeline.components.classification_base import AutoPLClassificationAlgorithm

__all__=["SGD"]


class SGD(
    AutoPLClassificationAlgorithm
):
    module__ = "sklearn.linear_model.stochastic_gradient"
    class__ = "SGDClassifier"

