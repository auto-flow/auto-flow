#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import logging
import pickle

import peewee as pw

logger = logging.getLogger(__name__)


def create_database(database,db_type, db_params: dict):
    return
    if db_type == "sqlite":
        pass
    elif db_type == "postgresql":
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
        conn = psycopg2.connect(
            dbname="postgres",
            user=db_params.get("user", "postgres"),
            host=db_params.get("host", "0.0.0.0"),
            port=db_params.get("port", 5432),
            password=db_params.get("password", None)
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        try:
            cur.execute(f"create database {database};")
        except Exception as e:
            logger.info(f"postgresql database {database} exists.")
        cur.close()
        conn.close()

    elif db_type == "mysql":
        raise NotImplementedError
    else:
        raise NotImplementedError


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


def get_JSONField(db_type):
    if db_type == "sqlite":
        from playhouse.sqlite_ext import JSONField
        JSONField = JSONField
    elif db_type == "postgresql":
        from playhouse.postgres_ext import JSONField
        JSONField = JSONField
    elif db_type == "mysql":
        from playhouse.mysql_ext import JSONField
        JSONField = JSONField
    else:
        raise NotImplementedError
    return JSONField


class PickleField(pw.BlobField):
    def db_value(self, value):
        if value is None or \
                (isinstance(value, str) and value == "") or \
                (isinstance(value, bytes) and value == b"") or \
                (isinstance(value, int) and value == 0):
            return b""
        return pickle.dumps(value)

    def python_value(self, value):
        if value == b"":
            return None
        try:
            return pickle.loads(value)
        except Exception as e:
            logger.warning(f"Failed in PickleFiled: \n{e}")
