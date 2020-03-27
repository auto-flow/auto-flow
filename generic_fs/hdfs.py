from fnmatch import fnmatchcase

import hdfs

from generic_fs import FileSystem


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