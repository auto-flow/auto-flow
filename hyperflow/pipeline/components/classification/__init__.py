import os

from hyperflow.pipeline.components.classification_base import HyperFlowClassificationAlgorithm
from hyperflow.utils.packages import find_components

classifier_directory = os.path.split(__file__)[0]
_classifiers = find_components(__package__,
                               classifier_directory,
                               HyperFlowClassificationAlgorithm)
