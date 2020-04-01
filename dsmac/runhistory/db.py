from dsmac.runhistory.runhistory import RunHistoryDB
from dsmac.runhistory.utils import get_id_of_config
from dsmac.tae.execute_ta_run import StatusType

if __name__ == '__main__':
    from ConfigSpace import ConfigurationSpace
    import joblib
    from dsmac.optimizer.objective import average_cost
    from dsmac.runhistory.runhistory import RunHistory

    runhistory = RunHistory(average_cost,db_args="test.db")
    cs: ConfigurationSpace = joblib.load("/home/tqc/PycharmProjects/hyperflow/test/php.bz2")
    runhistory.load_json(
        "/home/tqc/PycharmProjects/hyperflow/test/test_runhistory/default_dataset_name/smac_output/runhistory.json",
        cs)
    all_configs = (runhistory.get_all_configs())
    config = all_configs[0]
    config_id = get_id_of_config(config)
    cost = runhistory.get_cost(config)
    db = RunHistoryDB(cs, runhistory, "test.db")
    db.delete_all()
    ans = db.appointment_config(config)
    print(ans)
    db.insert_runhistory(config, cost, 0.1, StatusType.SUCCESS)
    db2 = RunHistoryDB(cs, runhistory, "test.db")
    db2.insert_runhistory(all_configs[1], runhistory.get_cost(all_configs[1]), 0.1, StatusType.SUCCESS)
    db.fetch_new_runhistory()
