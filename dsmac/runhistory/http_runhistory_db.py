import datetime
import inspect
import json
from typing import Tuple, Optional, Dict, Any

import peewee as pw
import requests

from dsmac.runhistory.runhistory_db import RunHistoryDB
from generic_fs.utils.utils import CustomJsonEncoder


def get_valid_params_in_kwargs(func, kwargs: Dict[str, Any]):
    validated = {}
    for key, value in kwargs.items():
        if key in inspect.signature(func).parameters.keys():
            validated[key] = value
    return validated


class HttpRunHistoryDB(RunHistoryDB):

    def get_db_cls(self):
        return None

    def get_db(self):
        return None

    def get_model(self) -> Optional[pw.Model]:
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
    #########################   run_history   ########################
    ##################################################################

    def _appointment_config(self, run_id, instance_id) -> Tuple[bool, Optional[pw.Model]]:
        local = get_valid_params_in_kwargs(self._appointment_config, locals())
        target = "appointment_config"
        response = self.post_requests(target, local)
        json_response = response.json()
        return json_response["ok"], json_response["record"]

    def _insert_runhistory_record(
            self, run_id, config_id, config, config_origin, cost: float, time: float,
            status: int, instance_id: str,
            seed: int,
            additional_info: dict,
            origin: int,
            pid: int,
    ):
        additional_info = dict(additional_info)
        modify_time = datetime.datetime.now()
        local = get_valid_params_in_kwargs(self._insert_runhistory_record, locals())
        target = "insert_runhistory_record"
        response = self.post_requests(target, local)

    def _fetch_new_runhistory(self, instance_id, pid, timestamp, is_init):
        local = get_valid_params_in_kwargs(self._fetch_new_runhistory, locals())
        target = "fetch_new_runhistory"
        response = self.post_requests(target, local)
        return response.json()
