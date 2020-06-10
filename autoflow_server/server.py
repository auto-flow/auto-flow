import os
import sys

try:
    import autoflow
except Exception:
    sys.path.insert(0, os.path.abspath(".."))

###########################################################

from typing import Dict, Any, List

from fastapi import FastAPI, Body
from autoflow.resource_manager.base import ResourceManager
from dsmac.runhistory.runhistory_db import RunHistoryDB

# from pydantic import BaseModel

app = FastAPI()
db_type = "sqlite"
db_params = {}
resource_manager = ResourceManager(db_type=db_type, db_params=db_params)
runhistory_db = RunHistoryDB(None, None, db_type, resource_manager.runhistory_db_params,
                             resource_manager.runhistory_table_name, "")
resource_manager.init_dataset_table()
resource_manager.init_experiment_table()
resource_manager.init_hdl_table()
# resource_manager.init_()
resource_manager.init_task_table()
resource_manager.init_trial_table()


##################################################################
############################  dataset ############################
##################################################################
@app.post("/dataset")
async def insert_to_dataset_table(
        column_descriptions: Dict[str, Any],
        columns_mapper: Dict[str, str],
        columns: List[str],
        user_id: int = Body(...),
        dataset_hash: str = Body(...),
        dataset_metadata: Dict[str, Any] = Body(...),
        upload_type: str = Body(...),
        dataset_source: str = Body(...),

):
    return resource_manager._insert_to_dataset_table(
        user_id,
        dataset_hash,
        dataset_metadata,
        upload_type,
        dataset_source,
        column_descriptions,
        columns_mapper,
        columns
    )


@app.post("/get_dataset_records")
async def get_dataset_records(dataset_id: str = Body(...), user_id: int = Body(...)) -> List[Dict[str, Any]]:
    return resource_manager._get_dataset_records(dataset_id, user_id)


##################################################################
##########################  experiment ###########################
##################################################################

@app.post("/experiment")
async def insert_to_experiment_table(
        experiment_config: Dict[str, Any], additional_info: Dict[str, Any],
        user_id: int = Body(...), hdl_id: str = Body(...), task_id: str = Body(...),
        experiment_type: str = Body(...),
):
    experiment_id = resource_manager._insert_to_experiment_table(
        user_id, hdl_id, task_id, experiment_type, experiment_config, additional_info
    )
    return {
        "experiment_id": experiment_id
    }


@app.post("/experiment_finish")
async def finish_experiment_update_info(
        end_time: str = Body(...),
        experiment_id: int = Body(...), final_model_path: str = Body(...), log_path: str = Body(...),
):
    resource_manager._finish_experiment_update_info(experiment_id, final_model_path, log_path, end_time)
    return {"msg": "ok"}


##################################################################
############################   task    ###########################
##################################################################

@app.post("/task")
async def insert_to_task_table(
        task_metadata: Dict[str, Any], sub_sample_indexes: List[str],
        sub_feature_indexes: List[str],
        task_id: str = Body(...), user_id: int = Body(...),
        metric_str: str = Body(...), splitter_str: str = Body(...), ml_task_str: str = Body(...),
        train_set_id: str = Body(...), test_set_id: str = Body(...), train_label_id: str = Body(...),
        test_label_id: str = Body(...), specific_task_token: str = Body(...),
):
    resource_manager._insert_to_task_table(
        task_id, user_id,
        metric_str, splitter_str, ml_task_str,
        train_set_id, test_set_id, train_label_id, test_label_id,
        specific_task_token, task_metadata, sub_sample_indexes, sub_feature_indexes)
    return {"task_id": task_id}


@app.post("/get_task_records")
async def get_task_records(task_id: str = Body(...), user_id: int = Body(...)):
    return resource_manager._get_task_records(task_id, user_id)


##################################################################
############################   hdl     ###########################
##################################################################

@app.post("/hdl")
async def insert_to_hdl_table(
        hdl_metadata: Dict[str, Any],
        task_id: str = Body(...), hdl_id: str = Body(...), user_id: int = Body(...), hdl: dict = Body(...),
):
    resource_manager._insert_to_hdl_table(task_id, hdl_id, user_id, hdl, hdl_metadata)
    return {"hdl_id": hdl_id}


##################################################################
############################   trial   ###########################
##################################################################

@app.post("/trial")
async def insert_to_trial_table(
        info: Dict[str, Any], user_id: int = Body(...),
        task_id: str = Body(...), hdl_id: str = Body(...), experiment_id: int = Body(...)):
    trial_id = resource_manager._insert_to_trial_table(user_id, task_id, hdl_id, experiment_id, info)
    return {"trial_id": trial_id}


@app.post("/get_sorted_trial_records")
async def get_sorted_trial_records(task_id: str = Body(...), user_id: int = Body(...), limit: int = Body(...)):
    return resource_manager._get_sorted_trial_records(task_id, user_id, limit)


@app.post("/get_trial_records_by_id")
async def get_trial_records_by_id(trial_id: int = Body(...)):
    return resource_manager._get_trial_records_by_id(trial_id)


@app.post("/get_trial_records_by_ids")
async def get_trial_records_by_ids(trial_ids=Body(...), k=Body(...)):
    return resource_manager._get_trial_records_by_ids(trial_ids)


@app.post("/get_best_k_trial_ids")
async def get_best_k_trial_ids(task_id: str = Body(...), user_id: int = Body(...), k: int = Body(...)):
    return resource_manager._get_best_k_trial_ids(task_id, user_id, k)


##################################################################
#########################   run_history   ########################
##################################################################

@app.post("/appointment_config")
async def _appointment_config(run_id: str = Body(...), instance_id: str = Body(...)):
    ok, record = runhistory_db._appointment_config(run_id, instance_id)
    response = {"ok": ok, "record": record}
    # encoded_response = jsonable_encoder(response)
    return response


@app.post("/runhistory")
async def insert_runhistory(
        run_id: str = Body(...), config_id: str = Body(...),
        config: Dict[str, Any] = Body(...),
        config_origin=Body(...), cost: float = Body(...), time: float = Body(...),
        status: int = Body(...), instance_id: str = Body(...),
        seed: int = Body(...),
        additional_info: Dict[str, Any] = Body(...),
        origin: int = Body(...),
        pid: int = Body(...),
):
    runhistory_db._insert_runhistory(
        run_id, config_id, config, config_origin, cost, time,
        status, instance_id, seed,
        additional_info,
        origin, pid
    )
    return {"msg": "ok"}


@app.post("/fetch_new_runhistory")
async def fetch_new_runhistory(
        instance_id: str = Body(...), pid: int = Body(...), timestamp: str = Body(...),
        is_init: bool = Body(...)):
    return runhistory_db._fetch_new_runhistory(instance_id, pid, timestamp, is_init)
