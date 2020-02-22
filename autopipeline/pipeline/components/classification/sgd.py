import numpy as np

from autopipeline.pipeline.components.base import AutoPLClassificationAlgorithm


class SGD(
    AutoPLClassificationAlgorithm
):
    module__ = "sklearn.linear_model.stochastic_gradient"
    class__ = "SGDClassifier"

