import os
import sqlite3
import time
from typing import Dict, Tuple, List, Union

import json5 as json
import pandas as pd
import peewee as pw
from joblib import dump, load

from hyperflow.constants import Task
from hyperflow.ensemble.mean.regressor import MeanRegressor
from hyperflow.ensemble.vote.classifier import VoteClassifier
from hyperflow.hdl.hdl_constructor import HDL_Constructor
from generic_fs import LocalFS
from generic_fs.utils import dumps_pickle, loads_pickle


class ResourceManager():
    '''
    资源管理： 文件系统与数据库
    '''

    def __init__(
            self,
            project_path=None,
            file_system=None,
            max_persistent_model=50,
            persistent_mode="fs",
            db_type="sqlite"
    ):
        self.persistent_mode = persistent_mode
        assert self.persistent_mode in ("fs", "db")
        self.db_type = db_type
        self.max_persistent_model = max_persistent_model
        if not file_system:
            file_system = LocalFS()
        self.file_system = file_system
        if not project_path:
            project_path = os.getcwd() + f'''/hyperflow-{time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())}'''
        self.project_path = project_path
        self.file_system.mkdir(self.project_path)

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
        self.dataset_path = self.project_path + "/" + dataset_name
        if not self.file_system.isdir(self.dataset_path):
            raise NotADirectoryError()
        self.init_dataset_path(dataset_name)

    def init_dataset_path(self, dataset_name):
        # 在fit的时候调用
        self.dataset_name = dataset_name
        self.dataset_path = self.project_path + "/" + dataset_name
        self.smac_output_dir = self.dataset_path + "/smac_output"
        self.file_system.mkdir(self.smac_output_dir)
        self.trials_dir = self.dataset_path + f"/trials"
        self.file_system.mkdir(self.trials_dir)
        self.db_path = self.dataset_path + f"/trials.db"
        self.csv_path = self.dataset_path + f"/trials.csv"
        self.data_manager_path = self.dataset_path + "/data_manager.bz2"
        self.hdl_dir = self.dataset_path + "/hdl_constructor"
        self.file_system.mkdir(self.hdl_dir)
        self.is_init_db = False
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
        dump(info["models"], file_name)
        return file_name

    def load_best_estimator(self, task: Task):
        # todo: 最后调用分析程序？
        self.init_db()
        record = self.Model.select().group_by(self.Model.loss, self.Model.cost_time).limit(1)[0]
        if self.persistent_mode == "fs":
            models = load(record.models_path)
        else:
            models = loads_pickle(record.models_bit)
        if task.mainTask == "classification":
            estimator = VoteClassifier(models)
        else:
            estimator = MeanRegressor(models)
        return estimator

    def load_best_dhp(self):
        trial_id=self.get_best_k_trials(1)[0]
        record = self.Model.select().where(self.Model.trial_id ==trial_id)[0]
        return record.dict_hyper_param

    def get_best_k_trials(self, k):
        self.init_db()
        trial_ids = []
        records = self.Model.select().group_by(self.Model.loss, self.Model.cost_time).limit(k)
        for record in records:
            trial_ids.append(record.trial_id)
        return trial_ids

    def load_estimators_in_trials(self, trials: Union[List, Tuple]) -> Tuple[List, List, List]:
        self.init_db()
        records = self.Model.select().where(self.Model.trial_id << trials)
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
        self._is_master = is_master

    @property
    def is_master(self):
        return self._is_master

    def get_model(self) -> pw.Model:
        class TrialModel(pw.Model):
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

            class Meta:
                database = self.db

        self.db.create_tables([TrialModel])
        return TrialModel

    def init_db(self):
        if self.is_init_db:
            return
        self.is_init_db = True
        # todo: 其他数据库的实现
        self.db: pw.Database = pw.SqliteDatabase(self.db_path)
        self.Model = self.get_model()

    def close_db(self):
        self.is_init_db=False
        self.db=None
        self.Model=None

    def insert_to_db(self, info: Dict):
        self.init_db()
        if self.persistent_mode == "fs":
            models_path = self.persistent_evaluated_model(info)
            models_bit = 0
        else:
            models_path = ""
            models_bit = dumps_pickle(info["models"])
        # TODO: 主键已存在的错误
        self.Model.create(
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
            warning_info=info.get("warning_info", "")
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
        self.init_db()
        estimators = []
        for record in self.Model.select().group_by(self.Model.estimator):
            estimators.append(record.estimator)
        for estimator in estimators:
            should_delete = self.Model.select().where(self.Model.estimator == estimator).order_by(
                self.Model.loss, self.Model.cost_time).offset(50)
            if should_delete:
                if self.persistent_mode == "fs":
                    for record in should_delete:
                        models_path = record.models_path
                        print("delete:" + models_path)
                        self.file_system.delete(models_path)
                self.Model.delete().where(self.Model.trial_id.in_(should_delete.select(self.Model.trial_id))).execute()
        return True

    def dump_db_to_csv(self):
        # todo: 做成一个分析工具，而不是每次都运行
        # db的内容都是随跑随写的，trails中的模型文件也一样。
        # 但是csv中的内容是运行了一段时间后生成的，如果任务突然中断，就会数据丢失
        data = []
        columns = ["trial_id", "estimator", "loss", "cost_time"]
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        fetched = cur.execute(f"select trial_id,estimator,loss,cost_time"
                              f" from record;")
        for row in fetched:
            data.append(row)
        cur.close()
        conn.close()
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(self.csv_path, index=False)


if __name__ == '__main__':
    rm = ResourceManager("/home/tqc/PycharmProjects/hyperflow/test/test_db")
    rm.init_dataset_path("default_dataset_name")
    rm.init_db()
    estimators = []
    for record in rm.Model.select().group_by(rm.Model.estimator):
        estimators.append(record.estimator)
    for estimator in estimators:
        should_delete = rm.Model.select(rm.Model.trial_id).where(rm.Model.estimator == estimator).order_by(
            rm.Model.loss, rm.Model.cost_time).offset(50)
        if should_delete:
            rm.Model.delete().where(rm.Model.trial_id.in_(should_delete)).execute()
