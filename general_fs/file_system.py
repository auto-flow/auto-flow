import logging
import os
from fnmatch import fnmatchcase
from glob import glob
from pathlib import Path

import hdfs


class FileSystem():
    def listdir(self, parent, **kwargs):
        raise NotImplementedError

    def join(self, path, *paths):
        return os.path.join(path, *paths)

    def read_txt(self, path):
        raise NotImplementedError

    def write_txt(self, path, txt, append=False):
        raise NotImplementedError

    def isdir(self, path):
        raise NotImplementedError

    def isfile(self, path):
        raise NotImplementedError

    def mkdir(self, path, **kwargs):
        raise NotImplementedError

    def glob(self, pattern):
        raise NotImplementedError

    def splitext(self, path):
        return os.path.splitext(path)

    def basename(self, path):
        return os.path.basename(path)

    def exists(self, path):
        raise NotImplementedError


class HDFS(FileSystem):
    def __init__(self, url='http://0.0.0.0:50070'):
        self.client = hdfs.client.Client(url)  # , level=logging.WARN

    def listdir(self, parent, **kwargs):
        return self.client.list(parent, kwargs.get('status', False))

    def read_txt(self, path):
        with self.client.read(path) as f:
            b: bytes = f.read()
            txt = b.decode(encoding='utf-8')
            return txt

    def exists(self, path):
        return self.client.status(path, strict=False) is not None

    def write_txt(self, path, txt, append=False):
        if not self.exists(path):
            append = False
        if append:
            self.client.write(path, txt, overwrite=False, append=True)
        else:
            self.client.write(path, txt, overwrite=True, append=False)

    def mkdir(self, path, **kwargs):
        self.client.makedirs(path)

    def glob(self, pattern: str):
        if '/' not in pattern:
            raise Exception('don not support relative path in HDFS mode.')
        r = pattern.rfind("/")
        prefix = pattern[:r]
        suffix = pattern[r + 1:]
        file_list = self.client.list(prefix)
        return [f'{prefix}/{file_name}' for file_name in file_list if fnmatchcase(file_name, suffix)]

    def isdir(self, path):
        try:
            if self.client.status(path)['type'] == 'DIRECTORY':
                return True
        except:
            return False

    def isfile(self, path):
        try:
            if self.client.status(path)['type'] == 'FILE':
                return True
        except:
            return False


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


if __name__ == '__main__':
    file_system = HDFS()
    res = file_system.glob('/user/tqc/test_dir')
    print(res)
