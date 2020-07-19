import logging
import os
import shutil
from glob import glob
from pathlib import Path

import pandas as pd
from joblib import dump, load

from generic_fs import FileSystem

logger = logging.getLogger(__name__)


class LocalFS(FileSystem):
    def listdir(self, parent, **kwargs):
        return os.listdir(parent)

    def read_txt(self, path):
        return Path(path).read_text()

    def write_txt(self, path, txt, append=False):
        if append:
            mode = 'a+'
        else:
            mode = 'w'
        with open(path, mode) as f:
            f.write(txt)

    def mkdir(self, path, **kwargs):
        Path(path).mkdir(exist_ok=kwargs.get('exist_ok', True), parents=kwargs.get('parents', True))

    def glob(self, pattern):
        return glob(pattern)

    def isdir(self, path):
        return os.path.isdir(path)

    def isfile(self, path):
        return os.path.isfile(path)

    def exists(self, path):
        return os.path.exists(path)

    def delete(self, path):
        if self.isfile(path):
            os.remove(path)
        elif self.isdir(path):
            os.rmdir(path)
        else:
            pass

    def dump_pickle(self, data, path):
        dump(data, path)
        return path

    def load_pickle(self, path):
        return load(path)

    def dump_csv(self, data: pd.DataFrame, path, **kwargs):
        data.to_csv(path, **kwargs)

    def load_csv(self, path, **kwargs) -> pd.DataFrame:
        return pd.read_csv(path, **kwargs)

    def upload(self, path, local_path):
        if os.path.exists(path):
            logger.warning(f"{path} already exists, don't upload.")
            return
        shutil.move(local_path, path)
        return path

    def download(self, path, local_path):
        if os.path.exists(local_path):
            os.remove(local_path)
        shutil.copy(path, local_path)
