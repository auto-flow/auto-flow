from autopipeline.pipeline.components.regression_base import AutoPLRegressionAlgorithm


class KNearestNeighborsRegressor(AutoPLRegressionAlgorithm):
    class__ = "KNeighborsRegressor"
    module__ = "sklearn.neighbors"