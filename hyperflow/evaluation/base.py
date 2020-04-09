from ConfigSpace import Configuration


class BaseEvaluator():
    def init_data(self,**kwargs):
        pass

    def __call__(self, shp:Configuration):
        pass