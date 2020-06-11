import datetime
import os
from typing import Tuple, Optional, List

import numpy as np
import peewee as pw
from ConfigSpace import ConfigurationSpace, Configuration
from frozendict import frozendict

from dsmac.runhistory.structure import DataOrigin
from dsmac.runhistory.utils import get_id_of_config
from dsmac.tae.execute_ta_run import StatusType
from dsmac.utils.logging import PickableLoggerAdapter
from generic_fs.utils.db import get_db_class_by_db_type, get_JSONField


class RunHistoryDB():

    def __init__(self, config_space: ConfigurationSpace, runhistory, db_type="sqlite",
                 db_params=frozendict(), db_table_name="runhistory", instance_id=""):

        self.instance_id = instance_id
        self.db_table_name = db_table_name
        self.runhistory = runhistory
        self.db_type = db_type
        self.db_params = db_params
        self.Datebase = self.get_db_cls()
        self.db: pw.Database = self.get_db()
        self.logger = PickableLoggerAdapter(__name__)
        # --JSONField-----------------------------------------
        self.JSONField = get_JSONField(self.db_type)
        # -----------------------------------------------------
        self.Model: pw.Model = self.get_model()
        self.config_space: ConfigurationSpace = config_space
        self.timestamp = datetime.datetime.now()

    def get_db_cls(self):
        return get_db_class_by_db_type(self.db_type)

    def get_db(self):
        return self.Datebase(**self.db_params)

    def get_model(self) -> pw.Model:
        class Run_History(pw.Model):
            run_id = pw.FixedCharField(max_length=256, primary_key=True)
            config_id = pw.FixedCharField(max_length=128, default="")
            config = self.JSONField(default={})
            # config_bin = pw.BlobField(default=b"")  # PickleField(default=b"")
            config_origin = pw.CharField(max_length=64, default="")
            cost = pw.FloatField(default=65535)
            time = pw.FloatField(default=0.0)
            instance_id = pw.FixedCharField(max_length=128, default="", index=True)  # 设置索引
            seed = pw.IntegerField(default=0)
            status = pw.IntegerField(default=0)
            additional_info = self.JSONField(default={})
            origin = pw.IntegerField(default=0)
            # weight = pw.FloatField(default=0.0)
            pid = pw.IntegerField(default=os.getpid)  # todo: 改为 worker id
            create_time = pw.DateTimeField(default=datetime.datetime.now)
            modify_time = pw.DateTimeField(default=datetime.datetime.now)

            class Meta:
                database = self.db
                table_name = self.db_table_name

        self.db.create_tables([Run_History])
        return Run_History

    @staticmethod
    def get_run_id(instance_id, config_id):
        return instance_id + "-" + config_id

    def appointment_config(self, config, instance_id) -> Tuple[bool, Optional[pw.Model]]:
        config_id = get_id_of_config(config)
        run_id = self.get_run_id(instance_id, config_id)
        return self._appointment_config(run_id, instance_id)

    def _appointment_config(self, run_id, instance_id) -> Tuple[bool, Optional[List[dict]]]:
        query = list(self.Model.select().where(self.Model.run_id == run_id).dicts())
        if len(query) > 0:
            query_ = query[0]
            if query_["origin"] >= 0:
                record = query_
            else:
                record = None
            return False, record
        try:
            self.Model.create(
                run_id=run_id,
                instance_id=instance_id,
                origin=-1
            )
        except Exception as e:
            return False, None
        return True, None

    def insert_runhistory_record(self, config: Configuration, cost: float, time: float,
                                 status: StatusType, instance_id: str = "",
                                 seed: int = 0,
                                 additional_info: dict = frozendict(),
                                 origin: DataOrigin = DataOrigin.INTERNAL):
        config_id = get_id_of_config(config)
        run_id = self.get_run_id(instance_id, config_id)
        if instance_id is None:
            instance_id = ""
        # pickle.dumps(config)
        self._insert_runhistory_record(run_id, config_id, config.get_dictionary(), config.origin, cost,
                                       time, status.value, instance_id, seed, additional_info, origin.value,
                                       os.getpid())

    def _insert_runhistory_record(
            self, run_id, config_id, config, config_origin, cost: float, time: float,
            status: int, instance_id: str,
            seed: int,
            additional_info: dict,
            origin: int,
            pid: int
    ):
        try:
            self.Model(
                run_id=run_id,
                config_id=config_id,
                config=config,
                config_origin=config_origin,
                # config_bin=config_bin,
                cost=cost,
                time=time,
                instance_id=instance_id,
                seed=seed,
                status=status,
                additional_info=dict(additional_info),
                origin=origin,
                modify_time=datetime.datetime.now(),
                pid=pid
            ).save()
        except Exception as e:
            pass

    def fetch_new_runhistory(self, instance_id, is_init=False) -> Tuple[float, Configuration]:
        query = self._fetch_new_runhistory(instance_id, os.getpid(), self.timestamp, is_init)
        self.timestamp = datetime.datetime.now()
        final_cost = np.inf
        final_config = None
        for model in query:
            run_id = model["run_id"]
            config_id = model["config_id"]
            config = model["config"]
            # config_bin = model["config_bin"]
            config_origin = model["config_origin"]
            cost = model["cost"]
            time = model["time"]
            instance_id = model["instance_id"]
            seed = model["seed"]
            status = model["status"]
            additional_info = model["additional_info"]
            origin = model["origin"]
            # try:
            #     config = pickle.loads(config_bin)
            # except Exception as e:
            # self.logger.error(f"{e}\nUsing config json instead to build Configuration.")
            config = Configuration(self.config_space, values=config, origin=config_origin)
            self.runhistory.add(config, cost, time, StatusType(status), instance_id, seed, additional_info,
                                DataOrigin(origin))
            if cost < final_cost:
                final_config = config
        return final_cost, final_config

    def _fetch_new_runhistory(self, instance_id, pid, timestamp, is_init):
        if is_init:
            # n_del = self.Model.delete().where(self.Model.origin < 0).execute()
            # if n_del > 0:
            #     self.logger.info(f"Delete {n_del} invalid records in run_history database.")
            query = self.Model.select().where(
                (self.Model.instance_id == instance_id) & (self.Model.origin >= 0)
            ).dicts()
        else:
            query = self.Model.select().where(
                (self.Model.instance_id == instance_id) & (self.Model.origin >= 0) &
                (self.Model.create_time > timestamp) & (self.Model.pid != pid)
            ).dicts()
        return list(query)
