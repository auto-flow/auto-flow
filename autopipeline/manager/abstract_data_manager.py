# -*- encoding: utf-8 -*-
import abc


class AbstractDataManager():
    __metaclass__ = abc.ABCMeta

    def __init__(self, name):

        self._data = dict()
        self._info = dict()
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def data(self):
        return self._data

    @property
    def info(self):
        return self._info

    @property
    def feat_type(self):
        return self._feat_type

    @feat_type.setter
    def feat_type(self, value):
        self._feat_type = value

    @property
    def encoder(self):
        return self._encoder

    @encoder.setter
    def encoder(self, value):
        self._encoder = value


    def __repr__(self):
        return 'DataManager : ' + self.name

    def __str__(self):
        val = 'DataManager : ' + self.name + '\ninfo:\n'
        for item in self.info:
            val = val + '\t' + item + ' = ' + str(self.info[item]) + '\n'
        val = val + 'data:\n'

        for subset in self.data:
            val = val + '\t%s = %s %s %s\n' % (subset, type(self.data[subset]),
                                               str(self.data[subset].shape),
                                               str(self.data[subset].dtype))

        val = val + 'feat_type:\t' + str(self.feat_type) + '\n'
        return val
