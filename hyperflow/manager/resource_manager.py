import datetime
import os
from typing import Dict, Tuple, List, Union

import json5 as json
import peewee as pw
from frozendict import frozendict
from joblib import dump, load
from redis import Redis

import generic_fs
from generic_fs import FileSystem
from generic_fs.utils import dumps_pickle, loads_pickle
from hyperflow.constants import MLTask
from hyperflow.ensemble.mean.regressor import MeanRegressor
from hyperflow.ensemble.vote.classifier import VoteClassifier
from hyperflow.hdl.hdl_constructor import HDL_Constructor
from hyperflow.utils.packages import find_components


class ResourceManager():
    '''
    ResourceManager: file_system and data_base
    '''

    def __init__(
            self,
            store_path="~",
            file_system="local",
            file_system_params=frozendict(),
            db_type="sqlite",
            db_params=frozendict(),
            redis_params=frozendict(),
            max_persistent_estimator=50,
            persistent_mode="fs",
            store_intermediate=True,
    ):
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
        self.max_persistent_estimator = max_persistent_estimator
        # ---persistent_mode-------
        self.persistent_mode = persistent_mode
        assert self.persistent_mode in ("fs", "db")
        # ---store_intermediate-------
        self.store_intermediate = store_intermediate
        # if not store_path:
        #     store_path = os.getcwd() + f'''/hyperflow-{time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())}'''
        self.store_path = store_path
        self.file_system.mkdir(self.store_path)
        self.is_init_trials_db = False
        self.is_init_experiments_db = False
        self.is_init_tasks_db = False
        self.is_init_redis = False
        self.is_master = False

    def load_hdl(self):
        persistent_data = {
            "hdl": None,
            "default_hp": None,
        }
        for name in list(persistent_data.keys()):
            txt = self.file_system.read_txt(self.hdl_dir + f"/{name}.json")
            persistent_data[name] = json.loads(txt)
        return persistent_data

    def load_dataset_path(self, dataset_name):
        # todo: 对数据做哈希进行检查, 保证前后两次使用的是同一个数据
        self.dataset_path = self.store_path + "/" + dataset_name
        if not self.file_system.isdir(self.dataset_path):
            raise NotADirectoryError()
        self.init_dataset_path(dataset_name)

    def init_dataset_path(self, dataset_name):
        # 在fit的时候调用
        self.dataset_name = dataset_name
        self.dataset_path = self.store_path + "/" + dataset_name
        self.smac_output_dir = self.dataset_path + "/smac_output"
        self.file_system.mkdir(self.smac_output_dir)
        self.trials_dir = self.dataset_path + f"/trials"
        self.file_system.mkdir(self.trials_dir)
        self.db_path = self.dataset_path + f"/trials.db"
        self.csv_path = self.dataset_path + f"/trials.csv"
        self.data_manager_path = self.dataset_path + "/data_manager.bz2"
        self.hdl_dir = self.dataset_path + "/hdl_constructor"
        self.file_system.mkdir(self.hdl_dir)
        if self.db_type == "sqlite":
            self.rh_db_args = self.dataset_path + "/runhistory.db"
            self.rh_db_kwargs = None

    def load_object(self, name):
        path = self.dataset_path + f"/{name}.bz2"
        return load(path)

    def dump_hdl(self, hdl_construct: HDL_Constructor):
        persistent_data = {
            "hdl_db": hdl_construct.hdl_db,
            # "default_hp": hdl_construct.default_hp,
            "hdl": hdl_construct.hdl,
            "params": hdl_construct.params,
        }
        for name, data in persistent_data.items():
            self.file_system.write_txt(self.hdl_dir + f"/{name}.json", json.dumps(data, indent=4))

    def dump_object(self, name, data):
        path = self.dataset_path + f"/{name}.bz2"
        dump(data, path)

    def persistent_evaluated_model(self, info: Dict):
        trial_id = info["trial_id"]
        file_name = f"{self.trials_dir}/{trial_id}.bz2"
        for model in info["models"]:
            model.resource_manager = None
        dump(info["models"], file_name)
        return file_name

    def load_best_estimator(self, ml_task: MLTask):
        # todo: 最后调用分析程序？
        self.connect_trials_db()
        record = self.TrialsModel.select().group_by(self.TrialsModel.loss, self.TrialsModel.cost_time).limit(1)[0]
        if self.persistent_mode == "fs":
            models = load(record.models_path)
        else:
            models = loads_pickle(record.models_bit)
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
            if self.persistent_mode == "fs":
                estimator_list.append(load(record.models_path))
            else:
                estimator_list.append(loads_pickle(record.models_bit))
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
            experiment_id = pw.PrimaryKeyField()
            general_task_timestamp = pw.DateTimeField(default=datetime.datetime.now)
            current_task_timestamp = pw.DateTimeField(default=datetime.datetime.now)
            HDL_list = pw.TextField(default="")
            HDL = pw.TextField(default="")
            HDL_id = pw.CharField(default="")
            Tuner_list = pw.TextField(default="")
            Tuner = pw.TextField(default="")
            task_id = pw.CharField(default="")
            # metric = pw.CharField(default="")
            all_scoring_functions = pw.BooleanField(default=True)
            # splitter = pw.CharField(default="")
            # column_descriptions = pw.TextField(default="")


            class Meta:
                database = self.experiments_db

        self.experiments_db.create_tables([Experiments])
        return Experiments

    def connect_experiments_db(self):
        if self.is_init_experiments_db:
            return
        self.is_init_experiments_db = True
        self.experiments_db: pw.Database = pw.SqliteDatabase(self.db_path)
        self.ExperimentsModel = self.get_trials_model()

    def close_experiments_db(self):
        self.is_init_experiments_db = False
        self.experiments_db = None
        self.ExperimentsModel = None
    # ----------experiments_model------------------------------------------------------------------
    def get_tasks_model(self) -> pw.Model:
        class Tasks(pw.Model):
            run_record_id = pw.PrimaryKeyField()
            general_task_timestamp = pw.DateTimeField(default=datetime.datetime.now)
            current_task_timestamp = pw.DateTimeField(default=datetime.datetime.now)
            HDL_list = pw.TextField(default="")
            HDL = pw.TextField(default="")
            HDL_id = pw.CharField(default="")
            Tuner_list = pw.TextField(default="")
            Tuner = pw.TextField(default="")
            task_id = pw.CharField(default="")
            metric = pw.CharField(default="")
            all_scoring_functions = pw.BooleanField(default=True)
            splitter = pw.CharField(default="")
            column_descriptions = pw.TextField(default="")


            class Meta:
                database = self.tasks_db

        self.tasks_db.create_tables([Tasks])
        return Tasks

    def connect_tasks_db(self):
        if self.is_init_tasks_db:
            return
        self.is_init_tasks_db = True
        self.tasks_db: pw.Database = pw.SqliteDatabase(self.db_path)
        self.TasksModel = self.get_trials_model()

    def close_tasks_db(self):
        self.is_init_tasks_db = False
        self.tasks_db = None
        self.TasksModel = None

    # ----------trials_model------------------------------------------------------------------

    def get_trials_model(self) -> pw.Model:
        class Trials(pw.Model):
            trial_id = pw.CharField(primary_key=True)
            estimator = pw.CharField(default="")
            loss = pw.FloatField(default=65535)
            losses = pw.TextField(default="")
            test_loss = pw.FloatField(default=65535)
            all_score = pw.TextField(default="")
            all_scores = pw.TextField(default="")
            test_all_score = pw.FloatField(default=0)
            models_bit = pw.BitField(default=0)
            models_path = pw.CharField(default="")
            y_true_indexes = pw.BitField(default=0)
            y_preds = pw.BitField(default=0)
            y_test_true = pw.BitField(default=0)
            y_test_pred = pw.BitField(default=0)
            program_hyper_param = pw.BitField(default=0)
            dict_hyper_param = pw.TextField(default="")  # todo: json field
            cost_time = pw.FloatField(default=65535)
            status = pw.CharField(default="success")
            failed_info = pw.TextField(default="")
            warning_info = pw.TextField(default="")
            timestamp = pw.DateTimeField(default=datetime.datetime.now)

            class Meta:
                database = self.trials_db

        self.trials_db.create_tables([Trials])
        return Trials

    def connect_trials_db(self):
        if self.is_init_trials_db:
            return
        self.is_init_trials_db = True
        # todo: 其他数据库的实现
        self.trials_db: pw.Database = pw.SqliteDatabase(self.db_path)
        self.TrialsModel = self.get_trials_model()

    def close_trials_db(self):
        self.is_init_trials_db = False
        self.trials_db = None
        self.TrialsModel = None

    def insert_to_trials_db(self, info: Dict):
        self.connect_trials_db()
        if self.persistent_mode == "fs":
            models_path = self.persistent_evaluated_model(info)
            models_bit = 0
        else:
            models_path = ""
            models_bit = dumps_pickle(info["models"])
        # TODO: 主键已存在的错误
        self.TrialsModel.create(
            trial_id=info["trial_id"],
            estimator=info.get("estimator", ""),
            loss=info.get("loss", 65535),
            losses=json.dumps(info.get("losses")),
            test_loss=info.get("test_loss", 65535),
            all_score=json.dumps(info.get("all_score")),
            all_scores=json.dumps(info.get("all_scores")),
            test_all_score=json.dumps(info.get("test_all_score")),
            models_bit=models_bit,
            models_path=models_path,
            y_true_indexes=dumps_pickle(info.get("y_true_indexes")),
            y_preds=dumps_pickle(info.get("y_preds")),
            y_test_true=dumps_pickle(info.get("y_test_true")),
            y_test_pred=dumps_pickle(info.get("y_test_pred")),
            program_hyper_param=dumps_pickle(info.get("program_hyper_param")),
            dict_hyper_param=json.dumps(info.get("dict_hyper_param")),  # t,odo: json field
            cost_time=info.get("cost_time", 65535),
            status=info.get("status", "failed"),
            failed_info=info.get("failed_info", ""),
            warning_info=info.get("warning_info", ""),
            timestamp=datetime.datetime.now()
        )

    def delete_models(self):
        # 更新记录各模型基本表现的csv，并删除表现差的模型

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
                self.TrialsModel.loss, self.TrialsModel.cost_time).offset(50)
            if should_delete:
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
