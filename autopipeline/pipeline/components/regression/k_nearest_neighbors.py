
from autopipeline.pipeline.components.base import AutoPLRegressionAlgorithm


class KNearestNeighborsRegressor(AutoPLRegressionAlgorithm):
    class__ = "KNeighborsRegressor"
    module__ = "sklearn.neighbors"