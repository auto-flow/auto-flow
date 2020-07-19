import logging
import os
from fnmatch import fnmatchcase

from generic_fs.utils.utils import remove_None_value

logger = logging.getLogger(__name__)

import hdfs
from joblib import dump, load

from generic_fs import FileSystem


class HDFS(FileSystem):
    def __init__(self, url='http://0.0.0.0:50070', user="dr.who", root=None, proxy=None, timeout=None, session=None):
        self.user = user
        self.session = session
        self.timeout = timeout
        self.proxy = proxy
        self.root = root
        self.url = url
        self.is_init = False

    def connect_fs(self):
        if self.is_init:
            return
        self.is_init = True
        kwargs = {
            "root": self.root,
            "proxy": self.proxy,
            "timeout": self.timeout,
            "session": self.session
        }
        kwargs = remove_None_value(kwargs)
        # todo 调研参数
        self.client = hdfs.client.InsecureClient(
            self.url, self.user,
            **kwargs
        )

    def close_fs(self):
        self.is_init = False
        self.client = None

    def listdir(self, parent, **kwargs):
        self.connect_fs()
        return self.client.list(parent, kwargs.get('status', False))

    def read_txt(self, path):
        self.connect_fs()
        with self.client.read(path) as f:
            b: bytes = f.read()
            txt = b.decode(encoding='utf-8')
            return txt

    def write_txt(self, path, txt, append=False):
        self.connect_fs()
        if not self.exists(path):
            append = False
        if append:
            self.client.write(path, txt, overwrite=False, append=True)
        else:
            self.client.write(path, txt, overwrite=True, append=False)

    def isdir(self, path):
        self.connect_fs()
        try:
            if self.client.status(path)['type'] == 'DIRECTORY':
                return True
        except:
            return False

    def isfile(self, path):
        self.connect_fs()
        try:
            if self.client.status(path)['type'] == 'FILE':
                return True
        except:
            return False

    def mkdir(self, path, **kwargs):
        self.connect_fs()
        self.client.makedirs(path)

    def glob(self, pattern: str):
        self.connect_fs()
        if '/' not in pattern:
            raise Exception('don not support relative path in HDFS mode.')
        r = pattern.rfind("/")
        prefix = pattern[:r]
        suffix = pattern[r + 1:]
        file_list = self.client.list(prefix)
        return [f'{prefix}/{file_name}' for file_name in file_list if fnmatchcase(file_name, suffix)]

    def exists(self, path):
        self.connect_fs()
        return self.client.status(path, strict=False) is not None

    def delete(self, path):
        self.connect_fs()
        self.client.delete(path, True)

    def dump_pickle(self, data, path):
        # todo : 对比远程与本地的MD5 建立缓存系统？
        # fixme: 记录一个异常，每次第一次运行都会出现下列警告
        # todo: 是应该直接退出吗？还是应该删除？可能这是个老数据？
        if self.exists(path):
            logger.warning(f"{path} already exists, don't do dump_pickle.")
            return
        self.connect_fs()
        tmp_path = self.join("/tmp", self.basename(path))
        dump(data, tmp_path)
        self.client.upload_remote(path, tmp_path)
        os.remove(tmp_path)
        return path

    def load_pickle(self, path):
        # todo : 对比远程与本地的MD5 建立缓存系统？
        self.connect_fs()
        tmp_path = self.join("/tmp", self.basename(path))
        self.client.download(path, tmp_path, overwrite=True)
        return load(tmp_path)

    def upload(self, path, local_path):
        if self.exists(path):
            logger.warning(f"{path} already exists, don't upload.")
            return
        self.connect_fs()
        self.client.upload_remote(path, local_path)
        os.remove(local_path)
        return path

    def download(self, path, local_path):
        self.connect_fs()
        self.client.download(path, local_path, overwrite=True)


if __name__ == '__main__':
    from pickle import loads, dumps

    fs = HDFS(url='http://0.0.0.0:50070')
    loads(dumps(fs))
