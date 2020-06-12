#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import datetime
import json
from typing import Dict, Any, List

import requests

from autoflow import ResourceManager
from autoflow.utils.json_ import CustomJsonEncoder
from autoflow.utils.klass import get_valid_params_in_kwargs


class HttpResourceManager(ResourceManager):
    ##################################################################
    #########################  disable db connection #################
    ##################################################################
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

    ##################################################################
    ############################  utils code #########################
    ##################################################################

    def judge_state_code(self, response: requests.Response):
        assert response.status_code == 200

    def post_requests(self, target, data) -> requests.Response:
        response = requests.post(self.db_params["url"] + "/" + target, headers=self.db_params["headers"],
                                 data=json.dumps(data, cls=CustomJsonEncoder))
        self.judge_state_code(response)
        return response

    ##################################################################
    ############################  dataset ############################
    ##################################################################

    def _insert_dataset_record(
            self,
            user_id: int,
            dataset_id: str,
            dataset_metadata: Dict[str, Any],
            dataset_path:str,
            upload_type: str,
            dataset_source: str,
            column_descriptions: Dict[str, Any],
            columns_mapper: Dict[str, str],
            columns: List[str]
    ):
        local = get_valid_params_in_kwargs(self._insert_dataset_record, locals())
        target = "insert_dataset_record"
        response = self.post_requests(target, local)
        return response.json()

    def _get_dataset_records(self, dataset_id, user_id) -> List[Dict[str, Any]]:
        local = get_valid_params_in_kwargs(self._get_dataset_records, locals())
        target = "get_dataset_records"
        response = self.post_requests(target, local)
        return response.json()

    ##################################################################
    ###########################  experiment ##########################
    ##################################################################

    def _insert_experiment_record(
            self, user_id: int, hdl_id: str, task_id: str,
            experiment_type: str,
            experiment_config: Dict[str, Any], additional_info: Dict[str, Any]
    ):
        local = get_valid_params_in_kwargs(self._insert_experiment_record, locals())
        target = "insert_experiment_record"
        response = self.post_requests(target, local)
        return response.json()["experiment_id"]

    def _finish_experiment_update_info(self, experiment_id: int, final_model_path: str, log_path: str,
                                       end_time: datetime.datetime):
        local = get_valid_params_in_kwargs(self._finish_experiment_update_info, locals())
        target = "finish_experiment_update_info"
        response = self.post_requests(target, local)

    ##################################################################
    ############################   task    ###########################
    ##################################################################

    def _insert_task_record(self, task_id: str, user_id: int,
                            metric_str: str, splitter_dict: Dict[str, str], ml_task_dict: Dict[str, str],
                            train_set_id: str, test_set_id: str, train_label_id: str, test_label_id: str,
                            specific_task_token: str, task_metadata: Dict[str, Any], sub_sample_indexes: List[str],
                            sub_feature_indexes: List[str]):
        local = get_valid_params_in_kwargs(self._insert_task_record, locals())
        target = "insert_task_record"
        response = self.post_requests(target, local)
        return response.json()["task_id"]

    def _get_task_records(self, task_id: str, user_id: int):
        local = get_valid_params_in_kwargs(self._get_task_records, locals())
        target = "get_task_records"
        response = self.post_requests(target, local)
        return response.json()

    ##################################################################
    ############################   hdl     ###########################
    ##################################################################

    def _insert_hdl_record(self, task_id: str, hdl_id: str, user_id: int, hdl: dict, hdl_metadata: Dict[str, Any]):
        local = get_valid_params_in_kwargs(self._insert_hdl_record, locals())
        target = "insert_hdl_record"
        response = self.post_requests(target, local)
        return response.json()["hdl_id"]

    ##################################################################
    ############################   trial   ###########################
    ##################################################################

    def _insert_trial_record(self, user_id: int, task_id: str, hdl_id: str, experiment_id: int, info: Dict[str, Any]):
        local = get_valid_params_in_kwargs(self._insert_trial_record, locals())
        target = "insert_trial_record"
        response = self.post_requests(target, local)
        return response.json()["trial_id"]

    def _get_sorted_trial_records(self, task_id, user_id, limit):
        local = get_valid_params_in_kwargs(self._get_sorted_trial_records, locals())
        target = "get_sorted_trial_records"
        response = self.post_requests(target, local)
        return response.json()

    def _get_trial_records_by_id(self, trial_id, k=0):
        local = get_valid_params_in_kwargs(self._get_trial_records_by_id, locals())
        target = "get_trial_records_by_id"
        response = self.post_requests(target, local)
        return response.json()

    def _get_trial_records_by_ids(self, trial_ids, k=0):
        local = get_valid_params_in_kwargs(self._get_trial_records_by_ids, locals())
        target = "get_trial_records_by_ids"
        response = self.post_requests(target, local)
        return response.json()

    def _get_best_k_trial_ids(self, task_id, user_id, k):
        local = get_valid_params_in_kwargs(self._get_best_k_trial_ids, locals())
        target = "get_best_k_trial_ids"
        response = self.post_requests(target, local)
        return response.json()
