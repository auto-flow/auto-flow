from ConfigSpace import Configuration

from hyperflow.utils.klass import StrSignatureMixin


class BaseEvaluator(StrSignatureMixin):
    def init_data(self,**kwargs):
        pass

    def __call__(self, shp:Configuration):
        pass