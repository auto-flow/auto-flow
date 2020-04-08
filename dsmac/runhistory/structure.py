import collections
import json
from enum import Enum

from dsmac.tae.execute_ta_run import StatusType

RunKey = collections.namedtuple(
    'RunKey', ['config_id', 'instance_id', 'seed'])
InstSeedKey = collections.namedtuple(
    'InstSeedKey', ['instance', 'seed'])
RunValue = collections.namedtuple(
    'RunValue', ['cost', 'time', 'status', 'additional_info'])


class EnumEncoder(json.JSONEncoder):
    """Custom encoder for enum-serialization
    (implemented for StatusType from tae/execute_ta_run).
    Using encoder implied using object_hook as defined in StatusType
    to deserialize from json.
    """

    def default(self, obj):
        if isinstance(obj, StatusType):
            return {"__enum__": str(obj)}
        return json.JSONEncoder.default(self, obj)


class DataOrigin(Enum):
    """
    Definition of how data in the runhistory is used.

    * ``INTERNAL``: internal data which was gathered during the current
      optimization run. It will be saved to disk, used for building EPMs and
      during intensify.
    * ``EXTERNAL_SAME_INSTANCES``: external data, which was gathered by running
       another program on the same instances as the current optimization run
       runs on (for example pSMAC). It will not be saved to disk, but used both
       for EPM building and during intensify.
    * ``EXTERNAL_DIFFERENT_INSTANCES``: external data, which was gathered on a
       different instance set as the one currently used, but due to having the
       same instance features can still provide useful information. Will not be
       saved to disk and only used for EPM building.
    """
    INTERNAL = 1
    EXTERNAL_SAME_INSTANCES = 2
    EXTERNAL_DIFFERENT_INSTANCES = 3