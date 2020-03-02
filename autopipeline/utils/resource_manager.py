import os
import sqlite3
import time
from typing import Dict

import joblib
import json5 as json
import pandas as pd
from joblib import dump, load

from autopipeline.constants import Task
from autopipeline.ensemble.mean.regressor import MeanRegressor
from autopipeline.ensemble.vote.classifier import VoteClassifier
from autopipeline.hdl.hdl_constructor import HDL_Constructor
from general_fs import LocalFS


class ResourceManager():
    def __init__(self, project_path=None, file_system=None, max_persistent_model=50):
        self.max_persistent_model = max_persistent_model
        if not file_system:
            file_system = LocalFS()
        self.file_system = file_system
        if not project_path:
            project_path = os.getcwd() + f'''/auto-pipeline-{time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())}'''
        self.project_path = project_path
        self.file_system.mkdir(self.project_path)

    def load_hdl(self):
        persistent_data={
            "hdl":None,
            "default_hp":None,
        }
        for name in list(persistent_data.keys()):
            txt=self.file_system.read_txt(self.hdl_dir + f"/{name}.json")
            persistent_data[name]=json.loads(txt)
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
        self.trials_dir = self.dataset_path + f"/trials"
        self.file_system.mkdir(self.trials_dir)
        self.db_path = self.dataset_path + f"/trials.db"
        self.csv_path = self.dataset_path + f"/trials.csv"
        self.data_manager_path = self.dataset_path + "/data_manager.bz2"
        self.hdl_dir = self.dataset_path + "/hdl_constructor"
        self.file_system.mkdir(self.hdl_dir)
        self.init_db()

    def load_object(self, name):
        path = self.dataset_path + f"/{name}.bz2"
        return load(path)

    def dump_hdl(self, hdl_construct: HDL_Constructor):
        persistent_data = {
            "hdl_db": hdl_construct.hdl_db,
            "default_hp": hdl_construct.default_hp,
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
        dump(info, file_name)

    def load_best_estimator(self, task: Task):
        df = pd.read_csv(self.csv_path)
        df.sort_values(by=["loss", "cost_time"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        assert len(df) > 0
        path = self.trials_dir + "/" + df["trial_id"][0] + ".bz2"
        models = joblib.load(path)["models"]
        if task.mainTask == "classification":
            estimator = VoteClassifier(models)
        else:
            estimator = MeanRegressor(models)
        return estimator

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "create table if not exists record(trial_id char(100),estimator char(32),loss real,cost_time real);")
        conn.commit()
        cur.close()
        conn.close()

    def insert_to_db(self, trial_id, estimator, loss, cost_time):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(f"insert into record values({repr(trial_id)},{repr(estimator)},{loss},{cost_time});")
        conn.commit()
        cur.close()
        conn.close()

    def fetch_all_from_db(self):
        data = []
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        fetched = cur.execute(f"select * from record;")
        for row in fetched:
            data.append(row)
        cur.close()
        conn.close()
        return data

    def delete_models(self):
        # 更新记录各模型基本表现的csv，并删除表现差的模型
        estimators = []
        data = []
        should_delete = []
        columns = ["trial_id", "estimator", "loss", "cost_time"]
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        fetched = cur.execute(f"select estimator from  record group by estimator;")
        for row in fetched:
            estimators.append(row[0])
        for estimator in estimators:
            fetched = cur.execute(f"select trial_id,estimator,loss,cost_time"
                                  f" from record where estimator=='{estimator}' "
                                  f"order by loss,cost_time limit {self.max_persistent_model};")
            for row in fetched:
                data.append(row)
            fetched = cur.execute(f"select trial_id from record where estimator=='{estimator}'"
                                  f" order by loss,cost_time limit -1 offset {self.max_persistent_model}")
            for row in fetched:
                should_delete.append(row[0])
            cur.execute(f"delete from record where trial_id in ("
                        f"select trial_id from record where estimator=='{estimator}'"
                        f" order by loss,cost_time limit -1 offset {self.max_persistent_model}"
                        f");")
            conn.commit()
        cur.close()
        conn.close()
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(self.csv_path, index=False)
        print("should_delete")
        print(should_delete)
        for trail_id in should_delete:
            self.file_system.delete(self.trials_dir + "/" + trail_id + ".bz2")

    def dump_db_to_csv(self):
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
        df.to_csv(self.csv_path,index=False)
