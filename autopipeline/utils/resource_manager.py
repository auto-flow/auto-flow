import os
import sqlite3
import time
from typing import Dict

import pandas as pd
from joblib import dump

from general_fs import LocalFS


class ResourceManager():
    def __init__(self, file_system=None, project_path=None, max_persistent_model=50):
        self.max_persistent_model = max_persistent_model
        if not file_system:
            file_system = LocalFS()
        self.file_system = file_system
        if not project_path:
            project_path = os.getcwd() + f'''/auto-pipeline-{time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())}'''
        self.project_path = project_path
        self.file_system.mkdir(self.project_path)
        self.smac_output_dir = self.project_path + "/smac_output"
        self.trials_dir = self.project_path + f"/trials"
        self.file_system.mkdir(self.trials_dir)
        self.db_path = self.project_path + f"/trials.db"
        self.csv_path = self.project_path + f"/trials.csv"
        self.init_db()

    def persistent_evaluated_model(self, info: Dict):
        trial_id = info["trial_id"]
        file_name = f"{self.trials_dir}/{trial_id}.bz2"
        dump(info, file_name)

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
