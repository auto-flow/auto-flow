from autoflow.resource_manager.base import ResourceManager
from autoflow.utils.klass import StrSignatureMixin


class TrialsFetcher(StrSignatureMixin):
    def __init__(
            self,
            resource_manager: ResourceManager,
            task_id: str,
            hdl_id: str,
    ):
        self.hdl_id = hdl_id
        self.task_id = task_id
        self.resource_manager = resource_manager

    def fetch(self):
        raise NotImplementedError


class GetBestK(TrialsFetcher):
    def __init__(
            self,
            resource_manager: ResourceManager,
            task_id: str,
            hdl_id: str,
            k: int
    ):
        super(GetBestK, self).__init__(resource_manager, task_id, hdl_id)
        self.k = k

    def fetch(self):
        self.resource_manager.task_id = self.task_id
        self.resource_manager.hdl_id = self.hdl_id
        self.resource_manager.init_trial_table()
        fetched = self.resource_manager._get_best_k_trial_ids(
            self.task_id, self.resource_manager.user_id, self.k
        )
        # self.resource_manager.close_trials_db()
        return fetched


class GetSpecificTrials(TrialsFetcher):
    def __init__(
            self,
            resource_manager: ResourceManager,
            task_id: str,
            hdl_id: str,
            trial_ids: int
    ):
        super(GetSpecificTrials, self).__init__(resource_manager, task_id, hdl_id)
        self.trial_ids = trial_ids

    def fetch(self):
        # todo: 校验？
        self.resource_manager.task_id = self.task_id
        self.resource_manager.hdl_id = self.hdl_id
        return self.trial_ids