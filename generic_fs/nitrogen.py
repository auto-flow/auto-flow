import logging
import os
from uuid import uuid4

import requests

from generic_fs.utils.http import send_requests

logger = logging.getLogger(__name__)

from joblib import dump, load

from generic_fs import FileSystem


class NitrogenFS(FileSystem):
    def __init__(self, db_params):
        self.db_params = db_params

    def mkdir(self, path, **kwargs):
        pass

    def get_url_of_dataset_id(self, dataset_id):
        response = send_requests(self.db_params, "signature/download",
                                 params={"object": str(dataset_id)}, method="get")
        data = response.json()["data"]
        if len(data) > 0 and "url" in data[0]:
            return data[0]["url"]
        return None

    def exists(self, path):
        return self.get_url_of_dataset_id(path) is not None

    def delete(self, path):
        pass

    def dump_pickle(self, data, path):
        tmp_path = self.join("/tmp", uuid4().hex + ".bz2")
        dump(data, tmp_path)
        dataset_id=self.upload(path, tmp_path)
        os.remove(tmp_path)
        return dataset_id

    def load_pickle(self, dataset_id):
        tmp_path = self.join("/tmp", uuid4().hex + ".bz2")
        self.download(dataset_id, tmp_path)
        return load(tmp_path)

    def upload(self, path, local_path) -> str:
        """

        Parameters
        ----------
        path
        local_path

        Returns
        -------
        dataset_id:int
        """
        response = send_requests(
            self.db_params, "signature/upload",
            params={"object": path},
            method="get"
        )
        data = response.json()["data"]
        url = data.get("url")
        if url is None: # path already exists
            return str(response.json()["message"][0])
        s3_header = {'Content-Type': 'multipart/form-data'}
        with open(local_path, 'rb') as f:
            requests.put(url, data=f, headers=s3_header)
        response = send_requests(
            self.db_params, "dataset",
            params={"file_name": path, "file_size": str(os.path.getsize(local_path))},
            method="put"
        )
        data = response.json()["data"]
        dataset_id = data["id"]
        return str(dataset_id)

    def download(self, dataset_id, local_path):
        url = self.get_url_of_dataset_id(dataset_id)
        assert url is not None, ValueError(f"dataste_id='{dataset_id}' is invalid")
        chunk_size = 1024 * 1024 * 128
        # todo: 这里报警告
        download_data = requests.get(url, stream=True)
        with open(local_path, 'wb') as f:
            for chunk in download_data.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        # self.get_small_file(url,local_path)

    def get_small_file(self,url, localfile):
        with open(localfile, 'wb') as f:
            r = requests.get(url, allow_redirects=True)
            f.write(r.content)
