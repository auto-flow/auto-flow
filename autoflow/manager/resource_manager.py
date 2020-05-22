import datetime
import hashlib
import os
import socket
from copy import deepcopy
from getpass import getuser
from math import ceil
from typing import Dict, Tuple, List, Union, Any

import h5py
import numpy as np
import pandas as pd
# import json5 as json
import peewee as pw
from frozendict import frozendict
from playhouse.fields import PickleField
from playhouse.reflection import generate_models
from redis import Redis

from autoflow.ensemble.mean.regressor import MeanRegressor
from autoflow.ensemble.vote.classifier import VoteClassifier
from autoflow.manager.data_manager import DataManager
from autoflow.metrics import Scorer
from autoflow.utils.dataframe import replace_nan_to_None, get_unique_col_name, replace_dicts
from autoflow.utils.dict_ import update_data_structure
from autoflow.utils.hash import get_hash_of_str, get_hash_of_dict
from autoflow.utils.klass import StrSignatureMixin
from autoflow.utils.logging_ import get_logger
from autoflow.utils.ml_task import MLTask
from generic_fs import FileSystem
from generic_fs.utils.db import get_db_class_by_db_type, get_JSONField, PickleField, create_database
from generic_fs.utils.fs import get_file_system


def get_field_of_type(type_, df, column):
    if not isinstance(type_, str):
        type_ = str(type_)
    type2field = {
        "int64": pw.IntegerField(null=True),
        "int32": pw.IntegerField(null=True),
        "float64": pw.FloatField(null=True),
        "float32": pw.FloatField(null=True),
        "bool": pw.BooleanField(null=True),
    }
    if type_ in type2field:
        return type2field[type_]
    elif type_ == "object":
        try:
            series = df[column]
            N = series.str.len().max()
            if N < 128:
                return pw.CharField(max_length=255, null=True)
            else:
                return pw.TextField(null=True)
        except:
            series = df[column]
            raise NotImplementedError(f"Unsupported type in 'get_field_of_type': '{type(series[0])}'")
    else:
        raise NotImplementedError


class ResourceManager(StrSignatureMixin):
    '''
    ``ResourceManager`` is a class manager computer resources such like ``file_system`` and ``data_base``.
    '''

    def __init__(
            self,
            store_path="~/autoflow",
            file_system="local",
            file_system_params=frozendict(),
            db_type="sqlite",
            db_params=frozendict(),
            redis_params=frozendict(),
            max_persistent_estimators=50,
            compress_suffix="bz2"
    ):
        '''

        Parameters
        ----------
        store_path: str
            A path store files, such as metadata and model file and database file, which belong to AutoFlow.
        file_system: str
            Indicator-string about which file system or storage system will be used.

            Available options list below:
                * ``local``
                * ``hdfs``
                * ``s3``

            ``local`` is default value.
        file_system_params: dict
            Specific file_system configuration.
        db_type: str
            Indicator-string about which file system or storage system will be used.

            Available options list below:
                * ``sqlite``
                * ``postgresql``
                * ``mysql``

            ``sqlite`` is default value.
        db_params: dict
            Specific database configuration.
        redis_params: dict
            Redis configuration.
        max_persistent_estimators: int
            Maximal number of models can persistent in single task.

            If more than this number, the The worst performing model file will be delete,

            the corresponding database record will also be deleted.
        compress_suffix: str
            compress file's suffix, default is bz2
        '''
        # --logger-------------------
        self.logger = get_logger(self)
        # --preprocessing------------
        file_system_params = dict(file_system_params)
        db_params = dict(db_params)
        redis_params = dict(redis_params)
        # ---file_system------------
        self.file_system_type = file_system
        self.file_system: FileSystem = get_file_system(file_system)(**file_system_params)
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
        self.redis_params = dict(redis_params)
        # ---max_persistent_model---
        self.max_persistent_estimators = max_persistent_estimators
        # ---compress_suffix------------
        self.compress_suffix = compress_suffix
        # ---post_process------------
        self.store_path = store_path
        self.file_system.mkdir(self.store_path)
        self.is_init_experiment = False
        self.is_init_task = False
        self.is_init_hdl = False
        self.is_init_trial = False
        self.is_init_dataset = False
        self.is_init_redis = False
        self.is_master = False
        self.is_init_record_db = False
        self.is_init_dataset_db = False
        # --some specific path based on file_system---
        self.datasets_dir = self.file_system.join(self.store_path, "datasets")
        self.databases_dir = self.file_system.join(self.store_path, "databases")
        self.parent_trials_dir = self.file_system.join(self.store_path, "trials")
        self.parent_experiments_dir = self.file_system.join(self.store_path, "experiments")
        for dir_path in [self.datasets_dir, self.databases_dir, self.parent_experiments_dir, self.parent_trials_dir]:
            self.file_system.mkdir(dir_path)
        # --db-----------------------------------------
        self.Datebase = get_db_class_by_db_type(self.db_type)
        # --JSONField-----------------------------------------
        self.JSONField = get_JSONField(self.db_type)
        # --database_name---------------------------------
        # None means didn't create database
        self._dataset_db_name = None
        self._record_db_name = None

    def close_all(self):
        self.close_redis()
        self.close_experiment_table()
        self.close_task_table()
        self.close_hdl_table()
        self.close_trial_table()
        self.close_dataset_table()
        self.close_dataset_db()
        self.close_record_db()
        self.file_system.close_fs()

    def __reduce__(self):
        self.close_all()
        return super(ResourceManager, self).__reduce__()

    def update_db_params(self, database):
        db_params = deepcopy(self.db_params)
        if self.db_type == "sqlite":
            db_params["database"] = self.file_system.join(self.databases_dir, f"{database}.db")
        elif self.db_type == "postgresql":
            db_params["database"] = database
        elif self.db_type == "mysql":
            db_params["database"] = database
        else:
            raise NotImplementedError
        return db_params

    def forecast_new_id(self, Dataset, id_field):
        # fixme : 用来预测下一个自增主键的ID，但是感觉有问题
        try:
            records = Dataset.select(getattr(Dataset, id_field)). \
                order_by(-getattr(Dataset, id_field)). \
                limit(1)
            if len(records) == 0:
                estimated_id = 1
            else:
                estimated_id = getattr(records[0], id_field) + 1
        except Exception as e:
            self.logger.error(f"Database Error:\n{e}")
            estimated_id = 1
        return estimated_id

    def persistent_evaluated_model(self, info: Dict, model_id) -> Tuple[str, str, str]:
        y_info = {
            # 这个变量还是很有必要的，因为可能用户指定的切分器每次切的数据不一样
            "y_true_indexes": info.get("y_true_indexes"),
            "y_preds": info.get("y_preds"),
            "y_test_pred": info.get("y_test_pred")
        }
        # ----dir---------------------
        self.trial_dir = self.file_system.join(self.parent_trials_dir, self.task_id, self.hdl_id)
        self.file_system.mkdir(self.trial_dir)
        # ----get specific URL---------
        models_path = self.file_system.join(self.trial_dir, f"{model_id}_models.{self.compress_suffix}")
        y_info_path = self.file_system.join(self.trial_dir, f"{model_id}_y-info.{self.compress_suffix}")
        if info.get("finally_fit_model") is not None:
            finally_fit_model_path = self.file_system.join(self.trial_dir,
                                                           f"{model_id}_final.{self.compress_suffix}")
        else:
            finally_fit_model_path = ""
        # ----do dump---------------
        self.file_system.dump_pickle(info["models"], models_path)
        self.file_system.dump_pickle(y_info, y_info_path)
        if finally_fit_model_path:
            self.file_system.dump_pickle(info["finally_fit_model"], finally_fit_model_path)
        # ----return----------------
        return models_path, finally_fit_model_path, y_info_path

    def get_ensemble_needed_info(self, task_id) -> Tuple[MLTask, Any, Any]:
        self.task_id = task_id
        self.init_task_table()
        task_record = self.TaskModel.select().where(self.TaskModel.task_id == task_id)[0]
        ml_task_str = task_record.ml_task
        ml_task = eval(ml_task_str)
        Xy_train_path = task_record.Xy_train_path
        Xy_train = self.file_system.load_pickle(Xy_train_path)
        Xy_test_path = task_record.Xy_test_path
        Xy_test = self.file_system.load_pickle(Xy_test_path)

        return ml_task, Xy_train, Xy_test

    def load_best_estimator(self, ml_task: MLTask):
        self.init_trial_table()
        record = self.TrialsModel.select() \
            .order_by(self.TrialsModel.loss, self.TrialsModel.cost_time).limit(1)[0]
        models = self.file_system.load_pickle(record.models_path)
        if ml_task.mainTask == "classification":
            estimator = VoteClassifier(models)
        else:
            estimator = MeanRegressor(models)
        return estimator

    def load_best_dhp(self):
        # fixme: 限制hdl_id
        trial_id = self.get_best_k_trials(1)[0]
        record = self.TrialsModel.select().where(self.TrialsModel.trial_id == trial_id)[0]
        return record.dict_hyper_param

    def get_best_k_trials(self, k):
        self.init_trial_table()
        trial_ids = []
        records = self.TrialsModel.select().order_by(self.TrialsModel.loss, self.TrialsModel.cost_time).limit(k)
        for record in records:
            trial_ids.append(record.trial_id)
        return trial_ids

    def load_estimators_in_trials(self, trials: Union[List, Tuple]) -> Tuple[List, List, List]:
        self.init_trial_table()
        records = self.TrialsModel.select().where(self.TrialsModel.trial_id << trials)
        estimator_list = []
        y_true_indexes_list = []
        y_preds_list = []
        for record in records:
            exists = True
            if not self.file_system.exists(record.models_path):
                exists = False
            else:
                estimator_list.append(self.file_system.load_pickle(record.models_path))
            if exists:
                y_info = self.file_system.load_pickle(record.y_info_path)
                y_true_indexes_list.append(y_info["y_true_indexes"])
                y_preds_list.append(y_info["y_preds"])
        return estimator_list, y_true_indexes_list, y_preds_list

    def set_is_master(self, is_master):
        self.is_master = is_master

    # ----------runhistory------------------------------------------------------------------

    @property
    def runhistory_db_params(self):
        self.init_record_db()
        return self.update_db_params(self.record_db_name)

    def get_runhistory_table_name(self, hdl_id):
        return "run_history"

    @property
    def runhistory_table_name(self):
        return self.get_runhistory_table_name(self.hdl_id)

    def migrate_runhistory(self, old_task_id, old_hdl_id, new_task_id, new_hdl_id):
        # todo: 设置一个 init_instances params
        pass
        # from dsmac.runhistory.runhistory_db import RunHistoryDB
        # # 1. 把 f"task_{old_task_id}".f"runhistory_{old_hdl_id}" 的所有数据records抽出来
        # old_runhistory_db = RunHistoryDB(
        #     config_space=None,
        #     runhistory=None,
        #     db_type=self.db_type,
        #     db_params=self.get_runhistory_db_params(old_task_id),
        #     db_table_name=self.get_runhistory_table_name(old_hdl_id)
        # ).get_model()
        # new_runhistory_db = RunHistoryDB(
        #     config_space=None,
        #     runhistory=None,
        #     db_type=self.db_type,
        #     db_params=self.get_runhistory_db_params(new_task_id),
        #     db_table_name=self.get_runhistory_table_name(new_hdl_id)
        # ).get_model()
        # old_runhistory_records = old_runhistory_db.select().dicts()
        # if len(old_runhistory_records) == 0:
        #     self.logger.warning(f"SMBO Transfer learning warning: old_runhistory_db have no records!")
        # else:
        #     self.logger.info(f"SMBO Transfer learning will migrate {len(old_runhistory_records)} "
        #                      f"records from old_runhistory_records.")
        #     transfer_cnt = 0
        #     for record in old_runhistory_records:
        #         fetched = new_runhistory_db.select().where(new_runhistory_db.run_id == record["run_id"])
        #         if len(fetched) == 0:
        #             transfer_cnt += 1
        #             new_runhistory_db.create(**record)
        #         else:
        #             self.logger.warning(f'''run_id '{record["run_id"]}' in new_runhistory_db is already exist.''')
        #     if transfer_cnt:
        #         self.logger.info(
        #             f"SMBO Transfer learning successfully migrate {transfer_cnt} records to new_runhistory_db.")
        #     else:
        #         self.logger.warning(
        #             f"Unfortunately, all the migrates failed, "
        #             f"please check if you have done SMBO transfer learning in previous tasks.")

    # ----------autoflow_dataset------------------------------------------------------------------
    @property
    def dataset_db_name(self):
        if self._dataset_db_name is not None:
            return self._dataset_db_name
        self._dataset_db_name = "autoflow_dataset"
        create_database(self._dataset_db_name, self.db_type, self.db_params)
        return self._dataset_db_name

    def init_dataset_db(self):
        if self.is_init_dataset_db:
            return self.dataset_db
        else:
            self.is_init_dataset_db = True
            self.dataset_db: pw.Database = self.Datebase(**self.update_db_params(self.dataset_db_name))
            return self.dataset_db

    def close_dataset_db(self):
        self.dataset_db = None
        self.is_init_dataset_db = False

    # ----------autoflow_record------------------------------------------------------------------

    @property
    def record_db_name(self):
        if self._record_db_name is not None:
            return self._record_db_name
        self._record_db_name = "autoflow_record"
        create_database(self._record_db_name, self.db_type, self.db_params)
        return self._record_db_name

    def init_record_db(self):
        if self.is_init_record_db:
            return self.record_db
        else:
            self.is_init_record_db = True
            self.record_db: pw.Database = self.Datebase(**self.update_db_params(self.record_db_name))
            return self.record_db

    def close_record_db(self):
        self.record_db = None
        self.is_init_record_db = False

    # ----------redis------------------------------------------------------------------

    def connect_redis(self):
        if self.is_init_redis:
            return True
        try:
            self.redis_client = Redis(**self.redis_params)
            self.is_init_redis = True
            return True
        except Exception as e:
            self.logger.error(f"Redis Error:\n{e}")
            return False

    def close_redis(self):
        self.redis_client = None
        self.is_init_redis = False

    def clear_pid_list(self):
        self.redis_delete("pid_list")

    def push_pid_list(self):
        if self.connect_redis():
            self.redis_client.rpush("pid_list", os.getpid())

    def get_pid_list(self):
        if self.connect_redis():
            l = self.redis_client.lrange("pid_list", 0, -1)
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

    def redis_hset(self, name, key, value):
        if self.connect_redis():
            try:
                self.redis_client.hset(name, key, value)
            except Exception as e:
                pass

    def redis_hgetall(self,name):
        if self.connect_redis():
            return self.redis_client.hgetall(name)
        else:
            return None

    def redis_delete(self, name):
        if self.connect_redis():
            self.redis_client.delete(name)

    # ----------dataset_model------------------------------------------------------------------
    def get_dataset_model(self) -> pw.Model:
        class Dataset_meta_record(pw.Model):
            dataset_id = pw.FixedCharField(max_length=128, primary_key=True)
            dataset_metadata = self.JSONField(default={})
            dataset_path = pw.CharField(max_length=512, default="")
            upload_type = pw.CharField(max_length=32, default="")
            dataset_type = pw.CharField(max_length=32, default="")
            dataset_source = pw.CharField(max_length=32, default="")
            column_descriptions = self.JSONField(default={})
            columns_mapper = self.JSONField(default={})
            columns = self.JSONField(default={})
            create_time = pw.DateTimeField(default=datetime.datetime.now)
            modify_time = pw.DateTimeField(default=datetime.datetime.now)

            class Meta:
                database = self.record_db

        self.record_db.create_tables([Dataset_meta_record])
        return Dataset_meta_record

    def insert_to_dataset_table(
            self,
            dataset_hash,
            dataset_metadata,
            upload_type,
            dataset_source,
            column_descriptions,
            columns_mapper,
            columns
    ) -> Tuple[int, str, str]:
        self.init_dataset_table()
        records = self.DatasetModel.select().where(self.DatasetModel.dataset_id == dataset_hash)
        L = len(records)
        dataset_path = ""
        if L != 0:
            record = records[0]
            record.modify_time = datetime.datetime.now()
            record.dataset_metadata = dataset_metadata
            record.save()
        else:
            if upload_type == "fs":
                dataset_path = self.file_system.join(self.datasets_dir, f"{dataset_hash}.h5")
            record = self.DatasetModel().create(
                dataset_id=dataset_hash,
                dataset_metadata=dataset_metadata,
                dataset_path=dataset_path,
                dataset_type="dataframe",
                upload_type=upload_type,
                dataset_source=dataset_source,
                column_descriptions=column_descriptions,
                columns_mapper=columns_mapper,
                columns=columns
            )
        dataset_id = record.dataset_id

        return L, dataset_id, dataset_path

    def init_dataset_table(self):
        if self.is_init_dataset:
            return
        self.is_init_dataset = True
        self.init_record_db()
        self.DatasetModel = self.get_dataset_model()

    def close_dataset_table(self):
        self.is_init_dataset = False
        self.DatasetModel = None

    def upload_df_to_table(self, df, dataset_hash, column_mapper):
        dataset_db = self.init_dataset_db()

        class Meta:
            database = dataset_db
            table_name = f"dataset_{dataset_hash}"

        origin_columns = deepcopy(df.columns)
        df.columns = pd.Series(df.columns).map(column_mapper)
        database_id = get_unique_col_name(df.columns, "id")
        dict_ = {"Meta": Meta, database_id: pw.AutoField()}
        for col_name, dtype in zip(df.columns, df.dtypes):
            dict_[col_name] = get_field_of_type(dtype, df, col_name)
        DataframeModel = type("DataframeModel", (pw.Model,), dict_)
        dataset_db.create_tables([DataframeModel])
        sp0 = df.shape[0]
        L = 500
        N = ceil(sp0 / L)
        for i in range(N):
            sub_df = df.iloc[i * L:min(sp0, (i + 1) * L)]
            sub_df = replace_nan_to_None(sub_df)
            dicts = sub_df.to_dict('records')
            DataframeModel.insert_many(dicts).execute()
        df.columns = origin_columns

    def upload_df_to_fs(self, df: pd.DataFrame, dataset_path):
        tmp_path = f"/tmp/tmp_df_{os.getpid()}.h5"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        df.to_hdf(tmp_path, "dataset")
        self.file_system.upload(dataset_path, tmp_path)

    def upload_ndarray_to_fs(self, arr: np.ndarray, dataset_path):
        tmp_path = f"/tmp/tmp_arr_{os.getpid()}.h5"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        with h5py.File(tmp_path, 'w') as hf:
            hf.create_dataset("dataset", data=arr)
        self.file_system.upload(dataset_path, tmp_path)

    def query_dataset_record(self, dataset_hash) -> List[Dict[str, Any]]:
        self.init_dataset_table()
        records = self.DatasetModel.select().where(self.DatasetModel.dataset_id == dataset_hash).dicts()
        return list(records)

    def download_df_of_table(self, dataset_hash, columns=None):
        dataset_db = self.init_dataset_db()
        models = generate_models(dataset_db)
        table_name = f"dataset_{dataset_hash}"
        if table_name not in models:
            raise ValueError(f"Table {table_name} didn't exists.")
        model = models[table_name]
        L = 500
        offset = 0
        dataframes = []
        while True:
            dicts = list(model().select().limit(L).offset(offset).dicts())
            replace_dicts(dicts, None, np.nan)
            sub_df = pd.DataFrame(dicts)
            dataframes.append(sub_df)
            if len(dicts) < L:
                break
            offset += L
        df = pd.concat(dataframes, axis=0)
        df.index = range(df.shape[0])
        # 删除第一列数据库主键列
        database_id = df.columns[0]
        df.pop(database_id)
        if columns is not None:
            df = df[columns]
        return df

    def download_df_of_fs(self, dataset_path, columns=None):
        tmp_path = f"/tmp/tmp_df_{os.getpid()}.h5"
        self.file_system.download(dataset_path, tmp_path)
        df: pd.DataFrame = pd.read_hdf(tmp_path, "dataset")
        if columns is not None:
            df = df[columns]
        return df

    def download_arr_of_fs(self, dataset_path):
        tmp_path = f"/tmp/tmp_arr_{os.getpid()}.h5"
        self.file_system.download(dataset_path, tmp_path)
        with h5py.File(tmp_path, 'r') as hf:
            arr = hf['dataset'][:]
        return arr

    # ----------experiment_model------------------------------------------------------------------
    def get_experiment_model(self) -> pw.Model:
        class Experiment(pw.Model):
            experiment_id = pw.AutoField(primary_key=True)
            general_experiment_time = pw.DateTimeField(default=datetime.datetime.now)
            current_experiment_time = pw.DateTimeField(default=datetime.datetime.now)
            modify_time = pw.DateTimeField(default=datetime.datetime.now)
            hdl_id = pw.FixedCharField(max_length=128, default="")
            task_id = pw.FixedCharField(max_length=128, default="")
            train_set_id = pw.FixedCharField(max_length=128, default="")  # fixme
            test_set_id = pw.FixedCharField(max_length=128, default="")  # fixme
            hdl_constructors = self.JSONField(default=[])
            hdl_constructor = pw.TextField(default="")
            raw_hdl = self.JSONField(default={})
            hdl = self.JSONField(default={})
            tuners = self.JSONField(default=[])
            tuner = pw.TextField(default="")
            data_manager_path = pw.CharField(max_length=512, default="")
            resource_manager_path = pw.CharField(max_length=512, default="")
            dataset_metadata = pw.CharField(max_length=512, default={})
            metric = pw.CharField(max_length=256, default="")  # task 的冗余字段
            splitter = pw.TextField(default="")  # task 的冗余字段
            ml_task = pw.CharField(max_length=64, default="")  # task 的冗余字段
            experiment_config = self.JSONField(default={})  # 实验配置，将一些不可优化的部分存储起来 # fixme
            additional_info = self.JSONField(default={})  # trials与experiments同时存储
            user = pw.CharField(max_length=64, default=getuser)

            class Meta:
                database = self.record_db

        self.record_db.create_tables([Experiment])
        return Experiment

    def get_experiment_id_by_task_id(self, task_id):
        self.init_task_table()
        return self.TaskModel.select(self.TaskModel.experiment_id).where(self.TaskModel.task_id == task_id)[
            0].experiment_id

    def load_data_manager_by_experiment_id(self, experiment_id):
        self.init_experiment_table()
        experiment_id = int(experiment_id)
        record = self.ExperimentModel.select().where(self.ExperimentModel.experiment_id == experiment_id)[0]
        data_manager_path = record.data_manager_path
        data_manager = self.file_system.load_pickle(data_manager_path)
        return data_manager

    def insert_to_experiment_table(
            self,
            general_experiment_time,
            current_experiment_time,
            hdl_constructors,
            hdl_constructor,
            raw_hdl,
            hdl,
            tuners,
            tuner,
            data_manager,
            dataset_metadata,
            metric,
            splitter,
            experiment_config,
            additional_info,
            set_id=True
    ):
        # todo: 不再保存data_manager
        # self.close_all()
        # copied_resource_manager = deepcopy(self)
        # data_manager = data_manager.copy()
        self.init_experiment_table()
        # estimate new experiment_id
        experiment_id = self.forecast_new_id(self.ExperimentModel, "experiment_id")
        # todo: 是否需要删除data_manager的Xy
        # data_manager.X_train = None
        # data_manager.X_test = None
        # data_manager.y_train = None
        # data_manager.y_test = None
        self.experiment_dir = self.file_system.join(self.parent_experiments_dir, str(experiment_id))
        self.file_system.mkdir(self.experiment_dir)
        data_manager_path = self.file_system.join(self.experiment_dir, f"data_manager.{self.compress_suffix}")
        resource_manager_path = self.file_system.join(self.experiment_dir, f"resource_manager.{self.compress_suffix}")
        # self.file_system.dump_pickle(data_manager, data_manager_path)
        # self.file_system.dump_pickle(copied_resource_manager, resource_manager_path)
        self.additional_info = additional_info
        experiment_record = self.ExperimentModel.create(
            general_experiment_time=general_experiment_time,
            current_experiment_time=current_experiment_time,
            hdl_id=self.hdl_id,
            task_id=self.task_id,
            train_set_id=data_manager.train_set_hash,
            test_set_id=data_manager.test_set_hash,
            hdl_constructors=[str(item) for item in hdl_constructors],
            hdl_constructor=str(hdl_constructor),
            raw_hdl=raw_hdl,
            hdl=hdl,
            tuners=[str(item) for item in tuners],
            tuner=str(tuner),
            data_manager_path=data_manager_path,
            resource_manager_path=resource_manager_path,
            dataset_metadata=dataset_metadata,
            metric=metric.name,
            splitter=str(splitter),
            ml_task=str(data_manager.ml_task),
            experiment_config=experiment_config,
            additional_info=additional_info,
        )
        fetched_experiment_id = experiment_record.experiment_id
        if fetched_experiment_id != experiment_id:
            self.logger.warning("fetched_experiment_id != experiment_id")
        if set_id:
            self.experiment_id = experiment_id

    def init_experiment_table(self):
        if self.is_init_experiment:
            return
        self.is_init_experiment = True
        self.init_record_db()
        self.ExperimentModel = self.get_experiment_model()

    def close_experiment_table(self):
        self.is_init_experiment = False
        self.ExperimentModel = None

    # ----------tasks_model------------------------------------------------------------------
    def get_task_model(self) -> pw.Model:
        class Task(pw.Model):
            task_id = pw.FixedCharField(max_length=128, primary_key=True)
            metric = pw.CharField(max_length=64, default="")
            splitter = pw.TextField(default="")
            ml_task = pw.CharField(max_length=64, default="")
            specific_task_token = pw.CharField(max_length=128, default="")
            train_set_id = pw.FixedCharField(max_length=128, default="")
            test_set_id = pw.FixedCharField(max_length=128, default="")
            sub_sample_indexes = self.JSONField()
            sub_feature_indexes = self.JSONField()
            create_time = pw.DateTimeField(default=datetime.datetime.now)
            modify_time = pw.DateTimeField(default=datetime.datetime.now)
            # meta info
            meta_data = self.JSONField(default={})

            class Meta:
                database = self.record_db

        self.record_db.create_tables([Task])
        return Task

    def insert_to_tasks_table(self, data_manager: DataManager,
                              metric: Scorer, splitter,
                              specific_task_token, dataset_metadata,
                              task_metadata, sub_sample_indexes, sub_feature_indexes, set_id=True):
        self.init_task_table()
        train_set_id = data_manager.train_set_hash
        test_set_id = data_manager.test_set_hash
        metric_str = metric.name
        splitter_str = str(splitter)
        ml_task_str = str(data_manager.ml_task)
        if sub_sample_indexes is None:
            sub_sample_indexes = []
        if sub_feature_indexes is None:
            sub_feature_indexes = []
        if not isinstance(sub_sample_indexes, list):
            sub_sample_indexes = list(sub_sample_indexes)
        if not isinstance(sub_feature_indexes, list):
            sub_feature_indexes = list(sub_feature_indexes)
        sub_sample_indexes_str = str(sub_sample_indexes)
        sub_feature_indexes_str = str(sub_feature_indexes)
        # ---task_id----------------------------------------------------
        m = hashlib.md5()
        get_hash_of_str(train_set_id, m)
        get_hash_of_str(test_set_id, m)
        get_hash_of_str(metric_str, m)
        get_hash_of_str(splitter_str, m)
        get_hash_of_str(ml_task_str, m)
        get_hash_of_str(sub_sample_indexes_str, m)
        get_hash_of_str(sub_feature_indexes_str, m)
        get_hash_of_str(specific_task_token, m)
        task_hash = m.hexdigest()
        task_id = task_hash
        records = self.TaskModel.select().where(self.TaskModel.task_id == task_id)
        meta_data = dict(
            dataset_metadata=dataset_metadata, **task_metadata
        )
        # ---store_task_record----------------------------------------------------
        if len(records) == 0:
            self.TaskModel.create(
                task_id=task_id,
                metric=metric_str,
                splitter=splitter_str,
                ml_task=ml_task_str,
                specific_task_token=specific_task_token,
                train_set_id=train_set_id,
                test_set_id=test_set_id,
                sub_sample_indexes=sub_sample_indexes,
                sub_feature_indexes=sub_feature_indexes,
                meta_data=meta_data
            )
        else:
            record = records[0]
            old_meta_data = record.meta_data
            new_meta_data = update_data_structure(old_meta_data, meta_data)
            record.meta_data = new_meta_data
            record.save()
        if set_id:
            self.task_id = task_id

    def init_task_table(self):
        if self.is_init_task:
            return
        self.is_init_task = True
        self.init_record_db()
        self.TaskModel = self.get_task_model()

    def close_task_table(self):
        self.is_init_task = False
        self.TaskModel = None

    # ----------hdl_model------------------------------------------------------------------
    def get_hdl_model(self) -> pw.Model:
        class Hdl(pw.Model):
            task_id = pw.FixedCharField(max_length=128, default="")
            hdl_id = pw.FixedCharField(max_length=128, default="")
            hdl = self.JSONField(default={})
            meta_data = self.JSONField(default={})
            create_time = pw.DateTimeField(default=datetime.datetime.now)
            modify_time = pw.DateTimeField(default=datetime.datetime.now)

            class Meta:
                database = self.record_db
                primary_key = pw.CompositeKey('task_id', 'hdl_id')

        self.record_db.create_tables([Hdl])
        return Hdl

    def insert_to_hdl_table(self, hdl, hdl_metadata, set_id=True):
        self.init_hdl_table()
        hdl_hash = get_hash_of_dict(hdl)
        hdl_id = hdl_hash
        records = self.HdlModel.select().where(
            (self.HdlModel.task_id == self.task_id) & (self.HdlModel.hdl_id == hdl_id))
        if len(records) == 0:
            self.HdlModel.create(
                task_id=self.task_id,
                hdl_id=hdl_id,
                hdl=hdl,
                meta_data=hdl_metadata
            )
        else:
            record = records[0]
            old_meta_data = record.meta_data
            new_meta_data = update_data_structure(old_meta_data, hdl_metadata)
            record.meta_data = new_meta_data
            record.save()
        if set_id:
            self.hdl_id = hdl_id

    def init_hdl_table(self):
        if self.is_init_hdl:
            return
        self.is_init_hdl = True
        self.init_record_db()
        self.HdlModel = self.get_hdl_model()

    def close_hdl_table(self):
        self.is_init_hdl = False
        self.HdlModel = None

    # ----------trial_model------------------------------------------------------------------

    def get_trial_model(self) -> pw.Model:
        class Trial(pw.Model):
            trial_id = pw.AutoField(primary_key=True,
                                    help_text="Trial ID. One experiment consists of many experiments.")
            config_id = pw.FixedCharField(max_length=128, default="",
                                          help_text="Configuration ID. Calculated by hash value of Configuration array.")
            task_id = pw.FixedCharField(max_length=128, default="",
                                        help_text="Task ID. Task ID is calculated by hash(TrainSet, TestSet, metric, splitter, ml_task, specific_task_token)",
                                        index=True)  # 加索引
            hdl_id = pw.FixedCharField(max_length=128, default="",
                                       help_text="Hyper-param Descriptions Language ID, calculated by hash(hdl)")
            experiment_id = pw.IntegerField(default=0,
                                            help_text="Experiment ID. The operation of a program is called an experiment.")
            estimator = pw.CharField(max_length=256, default="", help_text="Estimator. For instance: CNN, LSTM, SVM.")
            loss = pw.FloatField(default=65535,
                                 help_text="Loss value, is calculated by metric's current-value and 'optimal-value' and 'greater-is-better'. \n"
                                           "For instance, r2 is 'greater-is-better' and 'optimal-value'=1, now the r2=0.55, so loss=0.45 . ")
            losses = self.JSONField(default=[])
            test_loss = self.JSONField(default=[])  # 测试集
            all_score = self.JSONField(default={})
            all_scores = self.JSONField(default=[])
            test_all_score = self.JSONField(default={})  # 测试集
            models_path = pw.CharField(max_length=512, default="")
            final_model_path = pw.CharField(max_length=512, default="")
            y_info_path = pw.CharField(max_length=512, default="")
            # ------------被附加的额外信息---------------
            additional_info = self.JSONField(default={})
            # -------------------------------------
            smac_hyper_param = PickleField(default=0)
            dict_hyper_param = self.JSONField(default={})
            cost_time = pw.FloatField(default=65535)
            status = pw.CharField(max_length=32, default="SUCCESS")
            failed_info = pw.TextField(default="")
            warning_info = pw.TextField(default="")
            # todo: 改用数据集存储？变成Json字段，item是dataset ID
            # intermediate_result_path = pw.TextField(default="")
            intermediate_results = self.JSONField(default=[])
            create_time = pw.DateTimeField(default=datetime.datetime.now)
            modify_time = pw.DateTimeField(default=datetime.datetime.now)
            user = pw.CharField(max_length=32, default=getuser)
            pid = pw.IntegerField(default=os.getpid)
            host = pw.CharField(max_length=32, default=socket.gethostname)

            class Meta:
                database = self.record_db

        self.record_db.create_tables([Trial])
        return Trial

    def init_trial_table(self):
        if self.is_init_trial:
            return
        self.is_init_trial = True
        self.init_record_db()
        self.TrialsModel = self.get_trial_model()

    def close_trial_table(self):
        self.is_init_trial = False
        self.TrialsModel = None

    def insert_to_trial_table(self, info: Dict):
        self.init_trial_table()
        config_id = info.get("config_id")
        models_path, finally_fit_model_path, y_info_path = \
            self.persistent_evaluated_model(info, config_id)
        additional_info = deepcopy(self.additional_info)
        additional_info.update(info["additional_info"])
        self.TrialsModel.create(
            config_id=config_id,
            task_id=self.task_id,
            hdl_id=self.hdl_id,
            experiment_id=self.experiment_id,
            estimator=info.get("component", ""),
            loss=info.get("loss", 65535),
            losses=info.get("losses", []),
            test_loss=info.get("test_loss", 65535),
            all_score=info.get("all_score", {}),
            all_scores=info.get("all_scores", []),
            test_all_score=info.get("test_all_score", {}),
            models_path=models_path,
            final_model_path=finally_fit_model_path,
            y_info_path=y_info_path,
            additional_info=additional_info,
            smac_hyper_param=info.get("program_hyper_param"),
            dict_hyper_param=info.get("dict_hyper_param", {}),
            cost_time=info.get("cost_time", 65535),
            status=info.get("status", "failed"),
            failed_info=info.get("failed_info", ""),
            warning_info=info.get("warning_info", ""),
            intermediate_results=info.get("intermediate_results", []),
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
        self.init_trial_table()
        # estimators = []
        # for record in self.TrialsModel.select(self.TrialsModel.trial_id, self.TrialsModel.component).group_by(
        #         self.TrialsModel.component):
        #     estimators.append(record.component)
        # for component in estimators:
        # .where(self.TrialsModel.component == component)
        should_delete = self.TrialsModel.select().where(self.TrialsModel.task_id == self.task_id).order_by(
            self.TrialsModel.loss, self.TrialsModel.cost_time).offset(self.max_persistent_estimators)
        if len(should_delete):
            for record in should_delete:
                models_path = record.models_path
                self.logger.info(f"Delete expire Model in path : {models_path}")
                self.file_system.delete(models_path)
            self.TrialsModel.delete().where(
                self.TrialsModel.trial_id.in_(should_delete.select(self.TrialsModel.trial_id))).execute()
        return True


if __name__ == '__main__':
    rm = ResourceManager("/home/tqc/PycharmProjects/autoflow/test/test_db")
    rm.init_dataset_path("default_dataset_name")
    rm.init_trial_table()
    estimators = []
    for record in rm.TrialsModel.select().group_by(rm.TrialsModel.estimator):
        estimators.append(record.estimator)
    for estimator in estimators:
        should_delete = rm.TrialsModel.select(rm.TrialsModel.trial_id).where(
            rm.TrialsModel.estimator == estimator).order_by(
            rm.TrialsModel.loss, rm.TrialsModel.cost_time).offset(50)
        if should_delete:
            rm.TrialsModel.delete().where(rm.TrialsModel.trial_id.in_(should_delete)).execute()
