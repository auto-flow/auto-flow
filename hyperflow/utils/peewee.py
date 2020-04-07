import peewee as pw
import pickle

class PickleFiled(pw.BitField):
    def db_value(self, value):
        return pickle.dumps(value)

    def python_value(self, value):
        return pickle.loads(value)