'''The base class for different recommendation systems'''

from abc import ABCMeta, abstractmethod

class BaseModel(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, datasource):
        '''training models'''

    @abstractmethod
    def predict(self, data):
        '''prediction using the model'''