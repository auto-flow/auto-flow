#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import datetime
import json
from typing import Dict, Any, List

import requests

from autoflow import ResourceManager
from autoflow.utils.klass import get_valid_params_in_kwargs


class DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return str(obj)
        else:
            return json.JSONEncoder.default(self, obj)


class HttpResourceManager(ResourceManager):

    def init_dataset_db(self):
        return None

    def init_record_db(self):
        return None

    def init_dataset_table(self):
        return None

    def init_hdl_table(self):
        return None

    def init_experiment_table(self):
        return None

    def init_task_table(self):
        return None

    def init_trial_table(self):
        return None

    def _insert_to_dataset_table(
            self,
            user_id: int,
            dataset_hash: str,
            dataset_metadata: Dict[str, Any],
            upload_type: str,
            dataset_source: str,
            column_descriptions: Dict[str, Any],
            columns_mapper: Dict[str, str],
            columns: List[str]
    ):
        local = get_valid_params_in_kwargs(self._insert_to_dataset_table, locals())
        target = "dataset"
        response = requests.post(self.db_params["url"] + "/" + target, headers=self.db_params["headers"],
                                 data=json.dumps(local, cls=DateEncoder))
        return response.json()

    def _insert_to_experiment_table(
            self, user_id: int, hdl_id: str, task_id: str,
            experiment_type: str,
            experiment_config: Dict[str, Any], additional_info: Dict[str, Any]
    ):
        local = get_valid_params_in_kwargs(self._insert_to_experiment_table, locals())
        target = "experiment"
        response = requests.post(self.db_params["url"] + "/" + target, headers=self.db_params["headers"],
                                 data=json.dumps(local, cls=DateEncoder))
        return response.json()["experiment_id"]

    def _finish_experiment_update_info(self, experiment_id: int, final_model_path: str, log_path: str,
                                       end_time: datetime.datetime):
        local = get_valid_params_in_kwargs(self._finish_experiment_update_info, locals())
        target = "experiment_finish"
        response = requests.post(self.db_params["url"] + "/" + target, headers=self.db_params["headers"],
                                 data=json.dumps(local, cls=DateEncoder))

    def _insert_to_hdl_table(self, task_id: str, hdl_id: str, user_id: int, hdl: dict, hdl_metadata: Dict[str, Any]):
        local = get_valid_params_in_kwargs(self._insert_to_hdl_table, locals())
        target = "hdl"
        response = requests.post(self.db_params["url"] + "/" + target, headers=self.db_params["headers"],
                                 data=json.dumps(local, cls=DateEncoder))
        return response.json()["hdl_id"]

    def _insert_to_task_table(self, task_id: str, user_id: int,
                              metric_str: str, splitter_str: str, ml_task_str: str,
                              train_set_id: str, test_set_id: str, train_label_id: str, test_label_id: str,
                              specific_task_token: str, task_metadata: Dict[str, Any], sub_sample_indexes: List[str],
                              sub_feature_indexes: List[str]):
        local = get_valid_params_in_kwargs(self._insert_to_task_table, locals())
        target = "task"
        response = requests.post(self.db_params["url"] + "/" + target, headers=self.db_params["headers"],
                                 data=json.dumps(local, cls=DateEncoder))
        return response.json()["task_id"]

    def _insert_to_trial_table(self, user_id: int, task_id: str, hdl_id: str, experiment_id: int, info: dict):
        local = get_valid_params_in_kwargs(self._insert_to_trial_table, locals())
        target = "trial"
        response = requests.post(self.db_params["url"] + "/" + target, headers=self.db_params["headers"],
                                 data=json.dumps(local, cls=DateEncoder))
        return response.json()["trial_id"]
