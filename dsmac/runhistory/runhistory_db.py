import datetime
import json
import os
import pickle

import peewee as pw
from ConfigSpace import ConfigurationSpace, Configuration
from frozendict import frozendict

from dsmac.runhistory.structure import DataOrigin
from dsmac.runhistory.utils import get_id_of_config
from dsmac.tae.execute_ta_run import StatusType
from dsmac.utils.logging import PickableLoggerAdapter
from generic_fs.utils import get_db_class_by_db_type


class RunHistoryDB():

    def __init__(self, config_space: ConfigurationSpace, runhistory, db_type="sqlite",
                 db_params=frozendict(), db_table_name="runhistory"):

        self.db_table_name = db_table_name
        self.runhistory = runhistory
        self.db_type = db_type
        self.db_params = db_params
        self.Datebase = get_db_class_by_db_type(self.db_type)
        self.db: pw.Database = self.Datebase(**self.db_params)
        self.logger = PickableLoggerAdapter(__name__)
        # --JSONField-----------------------------------------
        if self.db_type == "sqlite":
            from playhouse.sqlite_ext import JSONField
            self.JSONField = JSONField
        elif self.db_type == "postgresql":
            from playhouse.postgres_ext import JSONField
            self.JSONField = JSONField
        elif self.db_type == "mysql":
            from playhouse.mysql_ext import JSONField
            self.JSONField = JSONField
        # -----------------------------------------------------
        self.Model: pw.Model = self.get_model()
        self.config_space: ConfigurationSpace = config_space

    def get_model(self) -> pw.Model:
        class Run_History(pw.Model):
            run_id = pw.CharField(primary_key=True)
            config_id = pw.CharField(default="")
            config = self.JSONField(default={})
            config_bin = pw.BitField(default=0)
            config_origin = pw.TextField(default="")
            cost = pw.FloatField(default=65535)
            time = pw.FloatField(default=0.0)
            instance_id = pw.CharField(default="")
            seed = pw.IntegerField(default=0)
            status = pw.IntegerField(default=0)
            additional_info = pw.CharField(default="")
            origin = pw.IntegerField(default=0)
            weight = pw.FloatField(default=0.0)
            pid = pw.IntegerField(default=os.getpid)
            timestamp = pw.DateTimeField(default=datetime.datetime.now)

            class Meta:
                database = self.db
                table_name = self.db_table_name

        self.db.create_tables([Run_History])
        return Run_History

    def get_run_id(self, instance_id, config_id):
        return instance_id + "-" + config_id

    def appointment_config(self, config, instance_id) -> bool:
        config_id = get_id_of_config(config)
        run_id = self.get_run_id(instance_id, config_id)
        query = self.Model.select().where(self.Model.run_id == run_id)
        if query.exists():
            return False
        try:
            self.Model.create(
                run_id=run_id,
                origin=-1
            )
        except Exception as e:
            return False
        return True

    def insert_runhistory(self, config: Configuration, cost: float, time: float,
                          status: StatusType, instance_id: str = "",
                          seed: int = 0,
                          additional_info: dict = frozendict(),
                          origin: DataOrigin = DataOrigin.INTERNAL):
        config_id = get_id_of_config(config)
        run_id = self.get_run_id(instance_id, config_id)
        if instance_id is None:
            instance_id = ""
        try:
            self.Model.create(
                run_id=run_id,
                config_id=config_id,
                config=config.get_dictionary(),
                config_origin=config.origin,
                config_bin=pickle.dumps(config),
                cost=cost,
                time=time,
                instance_id=instance_id,
                seed=seed,
                status=status.value,
                additional_info=dict(additional_info),
                origin=origin.value,
            )
        except pw.IntegrityError:
            self.Model(
                run_id=run_id,
                config_id=config_id,
                config=config.get_dictionary(),
                config_origin=config.origin,
                config_bin=pickle.dumps(config),
                cost=cost,
                time=time,
                instance_id=instance_id,
                seed=seed,
                status=status.value,
                additional_info=dict(additional_info),
                origin=origin.value,
            ).save()
        self.timestamp = datetime.datetime.now()

    def fetch_new_runhistory(self, is_init=False):
        if is_init:
            n_del = self.Model.delete().where(self.Model.origin < 0).execute()
            if n_del > 0:
                self.logger.info(f"Delete {n_del} invalid records in run_history database.")
            query = self.Model.select().where(self.Model.origin >= 0)
        else:
            query = self.Model.select().where(self.Model.pid != os.getpid()).where(self.Model.origin >= 0)
        for model in query:
            run_id = model.run_id
            config_id = model.config_id
            config = model.config
            config_bin = model.config_bin
            config_origin = model.config_origin
            cost = model.cost
            time = model.time
            instance_id = model.instance_id
            seed = model.seed
            status = model.status
            additional_info = model.additional_info
            origin = model.origin
            timestamp = model.timestamp
            try:
                config = pickle.loads(config_bin)
            except Exception as e:
                self.logger.error(f"{e}\nUsing config json instead to build Configuration.")
                config = Configuration(self.config_space, values=config, origin=config_origin)
            try:
                additional_info = json.loads(additional_info)
            except Exception as e:
                self.logger.error(f"{e}\nSet default to additional_info.")
                additional_info = {}
            self.runhistory.add(config, cost, time, StatusType(status), instance_id, seed, additional_info,
                                DataOrigin(origin))
        self.timestamp = datetime.datetime.now()
