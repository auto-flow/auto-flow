import pickle
from uuid import uuid4

import peewee as pw


def get_id():
    return str(uuid4().hex)


def dumps_pickle(data):
    return pickle.dumps(data)

def loads_pickle(bits):
    return pickle.loads(bits)


def get_db_class_by_db_type(db_type):
    if db_type == "sqlite":
        cls = pw.SqliteDatabase
    elif db_type == "postgresql":
        cls = pw.PostgresqlDatabase
    elif db_type == "mysql":
        cls = pw.MySQLDatabase
    else:
        raise NotImplementedError
    return cls