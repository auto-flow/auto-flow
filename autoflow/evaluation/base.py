from ConfigSpace import Configuration

from autoflow.utils.klass import StrSignatureMixin


class BaseEvaluator(StrSignatureMixin):
    def init_data(self,**kwargs):
        pass

    def __call__(self, shp:Configuration):
        pass