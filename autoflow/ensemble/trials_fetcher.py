from autoflow.manager.resource_manager import ResourceManager
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
        fetched = self.resource_manager.get_best_k_trials(self.k)
        # self.resource_manager.close_trials_db()
        return fetched
