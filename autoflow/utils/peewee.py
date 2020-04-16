import pickle

import peewee as pw

from autoflow.utils.logging import get_logger

logger = get_logger(__name__)


class PickleFiled(pw.BitField):
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
