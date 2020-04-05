import datetime
import hashlib
import os
from getpass import getuser
from typing import Dict, Tuple, List, Union

import json5 as json
import peewee as pw
from frozendict import frozendict
from joblib import load
from redis import Redis

import generic_fs
from generic_fs import FileSystem
from generic_fs.utils import dumps_pickle, loads_pickle, get_db_class_by_db_type
from hyperflow.constants import MLTask
from hyperflow.ensemble.mean.regressor import MeanRegressor
from hyperflow.ensemble.vote.classifier import VoteClassifier
from hyperflow.manager.xy_data_manager import XYDataManager
from hyperflow.metrics import Scorer
from hyperflow.utils.hash import get_hash_of_Xy, get_hash_of_str, get_hash_of_dict
from hyperflow.utils.packages import find_components


class ResourceManager():
    '''
    ResourceManager: file_system and data_base
    '''

    def __init__(
            self,
            store_path="~/hyperflow",
            file_system="local",
            file_system_params=frozendict(),
            db_type="sqlite",
            db_params=frozendict(),
            redis_params=frozendict(),
            max_persistent_estimators=50,
            persistent_mode="fs",
            store_intermediate=True,
            compress_suffix="bz2"

    ):
        file_system_params = dict(file_system_params)
        db_params = dict(db_params)
        redis_params = dict(redis_params)
        # ---file_system------------
        directory = os.path.split(generic_fs.__file__)[0]
        file_system2cls = find_components(generic_fs.__package__, directory, FileSystem)
        self.file_system_type = file_system
        if file_system not in file_system2cls:
            raise Exception(f"Invalid file_system {file_system}")
        self.file_system = file_system2cls[file_system](**file_system_params)
        if self.file_system_type == "local":
            store_path = os.path.expandvars(os.path.expanduser(store_path))
        self.store_path = store_path
        # ---data_base------------
        assert db_type in ("sqlite", "postgresql", "mysql")
        self.db_type = db_type
        self.db_params = dict(db_params)
        if db_type == "sqlite":
            assert self.file_system_type == "local"
        # ---redis----------------
        self.redis_params = redis_params
        # ---max_persistent_model---
        self.max_persistent_estimators = max_persistent_estimators
        # ---persistent_mode-------
        self.persistent_mode = persistent_mode
        assert self.persistent_mode in ("fs", "db")
        # ---store_intermediate-------
        self.store_intermediate = store_intermediate
        # ---compress_suffix------------
        self.compress_suffix = compress_suffix
        # ---post_process------------
        self.store_path = store_path
        self.file_system.mkdir(self.store_path)
        self.is_init_experiments_db = False
        self.is_init_tasks_db = False
        self.is_init_hdls_db = False
        self.is_init_trials_db = False
        self.is_init_redis = False
        self.is_master = False
        # --some specific path based on file_system---
        self.datasets_dir = self.file_system.join(self.store_path, "datasets")
        self.databases_dir = self.file_system.join(self.store_path, "databases")
        self.parent_trials_dir = self.file_system.join(self.store_path, "trials")
        self.parent_experiments_dir = self.file_system.join(self.store_path, "experiments")
        for dir_path in [self.datasets_dir, self.databases_dir, self.parent_experiments_dir, self.parent_trials_dir]:
            self.file_system.mkdir(dir_path)
        # --db-----------------------------------------
        self.Datebase = get_db_class_by_db_type(self.db_type)

    def get_runhistory_db_params(self):
        return self.update_db_params(self.get_trials_db_name())

    def update_db_params(self, database):
        db_params = dict(self.db_params)
        if self.db_type == "sqlite":
            db_params["database"] = self.file_system.join(self.databases_dir, f"{database}.db")
        elif self.db_type == "postgresql":
            pass
        elif self.db_type == "mysql":
            pass
        else:
            raise NotImplementedError
        return db_params

    def persistent_evaluated_model(self, info: Dict, trial_id):
        self.trial_dir = self.file_system.join(self.parent_trials_dir, self.task_id, self.hdl_id)
        self.file_system.mkdir(self.trial_dir)
        file_path = self.file_system.join(self.trial_dir, f"{trial_id}.{self.compress_suffix}")
        self.file_system.dump_pickle(info["models"], file_path)
        return file_path

    def load_best_estimator(self, ml_task: MLTask):
        # todo: 最后调用分析程序？
        self.connect_trials_db()
        record = self.TrialsModel.select().group_by(self.TrialsModel.loss, self.TrialsModel.cost_time).limit(1)[0]
        if self.persistent_mode == "fs":
            models = self.file_system.load_pickle(record.models_path)
        else:
            models = loads_pickle(record.models_bin)
        if ml_task.mainTask == "classification":
            estimator = VoteClassifier(models)
        else:
            estimator = MeanRegressor(models)
        return estimator

    def load_best_dhp(self):
        trial_id = self.get_best_k_trials(1)[0]
        record = self.TrialsModel.select().where(self.TrialsModel.trial_id == trial_id)[0]
        return record.dict_hyper_param

    def get_best_k_trials(self, k):
        self.connect_trials_db()
        trial_ids = []
        records = self.TrialsModel.select().group_by(self.TrialsModel.loss, self.TrialsModel.cost_time).limit(k)
        for record in records:
            trial_ids.append(record.trial_id)
        return trial_ids

    def load_estimators_in_trials(self, trials: Union[List, Tuple]) -> Tuple[List, List, List]:
        self.connect_trials_db()
        records = self.TrialsModel.select().where(self.TrialsModel.trial_id << trials)
        estimator_list = []
        y_true_indexes_list = []
        y_preds_list = []
        for record in records:
            exists = True
            if self.persistent_mode == "fs":
                if not self.file_system.exists(record.models_path):
                    exists = False
                else:
                    estimator_list.append(load(record.models_path))
            else:
                estimator_list.append(loads_pickle(record.models_bin))
            if exists:
                y_true_indexes_list.append(loads_pickle(record.y_true_indexes))
                y_preds_list.append(loads_pickle(record.y_preds))
        return estimator_list, y_true_indexes_list, y_preds_list

    def set_is_master(self, is_master):
        self.is_master = is_master

    # ----------redis------------------------------------------------------------------

    def connect_redis(self):
        if self.is_init_redis:
            return True
        try:
            self.redis_client = Redis(**self.redis_params)
            self.is_init_redis = True
            return True
        except Exception as e:
            print(f"warn:{e}")
            return False

    def close_redis(self):
        self.redis_client = None
        self.is_init_redis = False

    def clear_pid_list(self):
        self.redis_delete("hyperflow_pid_list")

    def push_pid_list(self):
        if self.connect_redis():
            self.redis_client.rpush("hyperflow_pid_list", os.getpid())

    def get_pid_list(self):
        if self.connect_redis():
            l = self.redis_client.lrange("hyperflow_pid_list", 0, -1)
            return list(map(lambda x: int(x.decode()), l))
        else:
            return []

    def redis_set(self, name, value, ex=None, px=None, nx=False, xx=False):
        if self.connect_redis():
            self.redis_client.set(name, value, ex, px, nx, xx)

    def redis_get(self, name):
        if self.connect_redis():
            return self.redis_client.get(name)
        else:
            return None

    def redis_delete(self, name):
        if self.connect_redis():
            self.redis_client.delete(name)

    # ----------experiments_model------------------------------------------------------------------
    def get_experiments_model(self) -> pw.Model:
        class Experiments(pw.Model):
            experiment_id = pw.IntegerField(primary_key=True)
            general_experiment_timestamp = pw.DateTimeField(default=datetime.datetime.now)
            current_experiment_timestamp = pw.DateTimeField(default=datetime.datetime.now)
            hdl_id = pw.CharField(default="")
            task_id = pw.CharField(default="")
            hdl_constructors = pw.TextField(default="")
            hdl_constructor = pw.TextField(default="")
            raw_hdl = pw.TextField(default="")
            hdl = pw.TextField(default="")
            tuners = pw.TextField(default="")
            tuner = pw.TextField(default="")
            all_scoring_functions = pw.BooleanField(default=True)
            data_manager_bin = pw.BitField(default=0)
            data_manager_path = pw.TextField(default="")
            column_descriptions = pw.TextField(default="")
            dataset_metadata = pw.TextField(default="")
            metric = pw.CharField(default=""),
            splitter = pw.CharField(default="")
            ml_task = pw.CharField(default="")
            user = pw.CharField(default=getuser)

            class Meta:
                database = self.experiments_db

        self.experiments_db.create_tables([Experiments])
        return Experiments

    def estimate_new_id(self, Dataset, id_field):
        try:
            records = Dataset.select(getattr(Dataset, id_field)). \
                where(getattr(Dataset, id_field)). \
                order_by(-getattr(Dataset, id_field)). \
                limit(1)
            if len(records) == 0:
                estimated_id = 1
            else:
                estimated_id = getattr(records[0], id_field) + 1
        except Exception as e:
            print(f"warn: {e}")
            estimated_id = 1
        return estimated_id

    def insert_to_experiments_db(
            self,
            general_experiment_timestamp,
            current_experiment_timestamp,
            hdl_constructors,
            hdl_constructor,
            raw_hdl,
            hdl,
            tuners,
            tuner,
            all_scoring_functions,
            data_manager,
            column_descriptions,
            dataset_metadata,
            metric,
            splitter,
    ):
        self.connect_experiments_db()
        # estimate new experiment_id
        experiment_id = self.estimate_new_id(self.ExperimentsModel, "experiment_id")
        # todo: 是否需要删除data_manager的Xy
        if self.persistent_mode == "fs":
            self.experiment_dir = self.file_system.join(self.parent_experiments_dir, str(experiment_id))
            self.file_system.mkdir(self.experiment_dir)
            data_manager_bin = 0
            data_manager_path = self.file_system.join(self.experiment_dir, f"data_manager.{self.compress_suffix}")
            self.file_system.dump_pickle(data_manager, data_manager_path)
        else:
            data_manager_path = ""
            data_manager_bin = dumps_pickle(data_manager)
        experiment_record = self.ExperimentsModel.create(
            general_experiment_timestamp=general_experiment_timestamp,
            current_experiment_timestamp=current_experiment_timestamp,
            hdl_id=self.hdl_id,
            task_id=self.task_id,
            hdl_constructors=str(hdl_constructors),
            hdl_constructor=str(hdl_constructor),
            raw_hdl=json.dumps(raw_hdl),
            hdl=json.dumps(hdl),
            tuners=str(tuners),
            tuner=str(tuner),
            all_scoring_functions=all_scoring_functions,
            data_manager_bin=data_manager_bin,
            data_manager_path=data_manager_path,
            column_descriptions=json.dumps(column_descriptions),
            dataset_metadata=json.dumps(dataset_metadata),
            metric=metric.name,
            splitter=str(splitter),
            ml_task=str(data_manager.ml_task)
        )
        fetched_experiment_id = experiment_record.experiment_id
        if fetched_experiment_id != experiment_id:
            print("warn: experiment_id")
        self.experiment_id = experiment_id

    def connect_experiments_db(self):
        if self.is_init_experiments_db:
            return
        self.is_init_experiments_db = True
        self.experiments_db: pw.Database = self.Datebase(**self.update_db_params("experiments"))
        self.ExperimentsModel = self.get_experiments_model()

    def close_experiments_db(self):
        self.is_init_experiments_db = False
        self.experiments_db = None
        self.ExperimentsModel = None

    # ----------tasks_model------------------------------------------------------------------
    def get_tasks_model(self) -> pw.Model:
        class Tasks(pw.Model):
            # task_id = md5(X_train, y_train, X_test, y_test, splitter, metric)
            task_id = pw.CharField(primary_key=True)
            metric = pw.CharField(default="")
            splitter = pw.CharField(default="")
            ml_task = pw.CharField(default="")
            # Xy_train
            Xy_train_hash = pw.CharField(default="")
            Xy_train_path = pw.TextField(default="")
            Xy_train_bin = pw.BitField(default=0)
            # Xy_test
            Xy_test_hash = pw.CharField(default="")
            Xy_test_path = pw.TextField(default="")
            Xy_test_bin = pw.BitField(default=0)

            class Meta:
                database = self.tasks_db

        self.tasks_db.create_tables([Tasks])
        return Tasks

    def insert_to_tasks_db(self, data_manager: XYDataManager, metric: Scorer, splitter):
        self.connect_tasks_db()
        Xy_train_hash = get_hash_of_Xy(data_manager.X_train, data_manager.y_train)
        Xy_test_hash = get_hash_of_Xy(data_manager.X_test, data_manager.y_test)
        Xy_train = [data_manager.X_train, data_manager.y_train]
        Xy_test = [data_manager.X_test, data_manager.y_test]
        if self.persistent_mode == "fs":
            Xy_train_path = self.file_system.join(self.datasets_dir,
                                                  f"{Xy_train_hash}.{self.compress_suffix}")
            self.file_system.dump_pickle(Xy_train, Xy_train_path)
            Xy_train_bin = 0
        else:
            Xy_train_path = ""
            Xy_train_bin = dumps_pickle(Xy_train)
        if Xy_test_hash:
            if self.persistent_mode == "fs":
                Xy_test_path = self.file_system.join(self.datasets_dir,
                                                     f"{Xy_test_hash}.{self.compress_suffix}")
                self.file_system.dump_pickle(Xy_test, Xy_test_path)
                Xy_test_bin = 0
            else:
                Xy_test_path = ""
                Xy_test_bin = dumps_pickle(Xy_test)
        else:
            Xy_test_path = ""
            Xy_test_bin = 0
        metric_str = metric.name
        splitter_str = str(splitter)
        ml_task_str = str(data_manager.ml_task)
        # task_id
        m = hashlib.md5()
        get_hash_of_Xy(data_manager.X_train, data_manager.y_train, m)
        get_hash_of_Xy(data_manager.X_test, data_manager.y_test, m)
        get_hash_of_str(metric_str, m)
        get_hash_of_str(splitter_str, m)
        get_hash_of_str(ml_task_str, m)
        task_hash = m.hexdigest()
        task_id = "task_" + task_hash
        records = self.TasksModel.select().where(self.TasksModel.task_id == task_id)
        if len(records) == 0:
            task_record, created = self.TasksModel.get_or_create(
                task_id=task_id,
                metric=metric_str,
                splitter=splitter_str,
                ml_task=ml_task_str,
                # Xy_train
                Xy_train_hash=Xy_train_hash,
                Xy_train_path=Xy_train_path,
                Xy_train_bin=Xy_train_bin,
                # Xy_test
                Xy_test_hash=Xy_test_hash,
                Xy_test_path=Xy_test_path,
                Xy_test_bin=Xy_test_bin,
            )
        self.task_id = task_id

    def connect_tasks_db(self):
        if self.is_init_tasks_db:
            return
        self.is_init_tasks_db = True
        self.tasks_db: pw.Database = self.Datebase(**self.update_db_params("tasks"))
        self.TasksModel = self.get_tasks_model()

    def close_tasks_db(self):
        self.is_init_tasks_db = False
        self.tasks_db = None
        self.TasksModel = None

    # ----------hdls_model------------------------------------------------------------------
    def get_hdls_model(self) -> pw.Model:
        class HDLs(pw.Model):
            hdl_id = pw.CharField(primary_key=True)
            hdl = pw.TextField(default="")

            class Meta:
                database = self.hdls_db

        self.hdls_db.create_tables([HDLs])
        return HDLs

    def insert_to_hdls_db(self, hdl):
        self.connect_hdls_db()
        hdl_hash = get_hash_of_dict(hdl)
        hdl_id = "hdl_" + hdl_hash
        records = self.HDLsModel.select().where(self.HDLsModel.hdl_id == hdl_id)
        if len(records) == 0:
            hdl_record, created = self.HDLsModel.get_or_create(
                hdl_id=hdl_id,
                hdl=json.dumps(hdl)
            )
        self.hdl_id = hdl_id

    def connect_hdls_db(self):
        if self.is_init_hdls_db:
            return
        self.is_init_hdls_db = True
        self.hdls_db: pw.Database = self.Datebase(**self.update_db_params(self.task_id))
        self.HDLsModel = self.get_hdls_model()

    def close_hdls_db(self):
        self.is_init_hdls_db = False
        self.hdls_db = None
        self.HDLsModel = None

    # ----------trials_model------------------------------------------------------------------
    def get_trials_db_name(self):
        return f"{self.task_id}-{self.hdl_id}"

    def get_trials_model(self) -> pw.Model:
        class Trials(pw.Model):
            trial_id = pw.IntegerField(primary_key=True)
            config_id = pw.CharField(default="")
            task_id = pw.CharField(default="")
            hdl_id = pw.CharField(default="")
            experiment_id = pw.IntegerField(default=0)
            estimator = pw.CharField(default="")
            loss = pw.FloatField(default=65535)
            losses = pw.TextField(default="")
            test_loss = pw.FloatField(default=65535)
            all_score = pw.TextField(default="")
            all_scores = pw.TextField(default="")
            test_all_score = pw.FloatField(default=0)
            models_bin = pw.BitField(default=0)
            models_path = pw.CharField(default="")
            y_true_indexes = pw.BitField(default=0)
            y_preds = pw.BitField(default=0)
            y_test_true = pw.BitField(default=0)
            y_test_pred = pw.BitField(default=0)
            smac_hyper_param = pw.BitField(default=0)
            dict_hyper_param = pw.TextField(default="")  # todo: json field
            cost_time = pw.FloatField(default=65535)
            status = pw.CharField(default="success")
            failed_info = pw.TextField(default="")
            warning_info = pw.TextField(default="")
            timestamp = pw.DateTimeField(default=datetime.datetime.now)
            user = pw.CharField(default=getuser)

            class Meta:
                database = self.trials_db

        self.trials_db.create_tables([Trials])
        return Trials

    def connect_trials_db(self):
        if self.is_init_trials_db:
            return
        self.is_init_trials_db = True
        self.trials_db: pw.Database = self.Datebase(**self.update_db_params(self.get_trials_db_name()))
        self.TrialsModel = self.get_trials_model()

    def close_trials_db(self):
        self.is_init_trials_db = False
        self.trials_db = None
        self.TrialsModel = None

    def insert_to_trials_db(self, info: Dict):
        self.connect_trials_db()
        config_id = info.get("config_id")
        if self.persistent_mode == "fs":
            models_path = self.persistent_evaluated_model(info, config_id)  # todo: 考虑更特殊的情况，不同的任务下，相同的配置
            models_bin = 0
        else:
            models_path = ""
            models_bin = dumps_pickle(info["models"])
        trial_record, created = self.TrialsModel.get_or_create(
            config_id=config_id,
            task_id=self.task_id,
            hdl_id=self.hdl_id,
            experiment_id=self.experiment_id,
            estimator=info.get("estimator", ""),
            loss=info.get("loss", 65535),
            losses=json.dumps(info.get("losses")),
            test_loss=info.get("test_loss", 65535),
            all_score=json.dumps(info.get("all_score")),
            all_scores=json.dumps(info.get("all_scores")),
            test_all_score=json.dumps(info.get("test_all_score")),
            models_bin=models_bin,
            models_path=models_path,
            y_true_indexes=dumps_pickle(info.get("y_true_indexes")),
            y_preds=dumps_pickle(info.get("y_preds")),
            y_test_true=dumps_pickle(info.get("y_test_true")),
            y_test_pred=dumps_pickle(info.get("y_test_pred")),
            smac_hyper_param=dumps_pickle(info.get("program_hyper_param")),
            dict_hyper_param=json.dumps(info.get("dict_hyper_param")),  # t,odo: json field
            cost_time=info.get("cost_time", 65535),
            status=info.get("status", "failed"),
            failed_info=info.get("failed_info", ""),
            warning_info=info.get("warning_info", ""),
            timestamp=datetime.datetime.now()
        )

    def delete_models(self):
        if hasattr(self, "sync_dict"):
            exit_processes = self.sync_dict.get("exit_processes", 3)
            records = 0
            for key, value in self.sync_dict.items():
                if isinstance(key, int):
                    records += value
            if records >= exit_processes:
                return False
        # master segment
        if not self.is_master:
            return True
        self.connect_trials_db()
        estimators = []
        for record in self.TrialsModel.select().group_by(self.TrialsModel.estimator):
            estimators.append(record.estimator)
        for estimator in estimators:
            should_delete = self.TrialsModel.select().where(self.TrialsModel.estimator == estimator).order_by(
                self.TrialsModel.loss, self.TrialsModel.cost_time).offset(self.max_persistent_estimators)
            if len(should_delete):
                if self.persistent_mode == "fs":
                    for record in should_delete:
                        models_path = record.models_path
                        print("delete:" + models_path)
                        self.file_system.delete(models_path)
                self.TrialsModel.delete().where(
                    self.TrialsModel.trial_id.in_(should_delete.select(self.TrialsModel.trial_id))).execute()
        return True


if __name__ == '__main__':
    rm = ResourceManager("/home/tqc/PycharmProjects/hyperflow/test/test_db")
    rm.init_dataset_path("default_dataset_name")
    rm.connect_trials_db()
    estimators = []
    for record in rm.TrialsModel.select().group_by(rm.TrialsModel.estimator):
        estimators.append(record.estimator)
    for estimator in estimators:
        should_delete = rm.TrialsModel.select(rm.TrialsModel.trial_id).where(
            rm.TrialsModel.estimator == estimator).order_by(
            rm.TrialsModel.loss, rm.TrialsModel.cost_time).offset(50)
        if should_delete:
            rm.TrialsModel.delete().where(rm.TrialsModel.trial_id.in_(should_delete)).execute()
