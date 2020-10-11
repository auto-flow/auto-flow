import ast
import cgitb
import datetime
import json
import os
import socket
import sys
import traceback
from pathlib import Path

from tabulate import tabulate

from autoflow.utils.logging_ import get_logger

logger = get_logger(__name__)


class Global:
    '''
    跨文件全局变量类
    '''

    @classmethod
    def get(cls, key, default=None):
        return cls.__dict__.get(key, default)


def get_log_dir():
    return Global.get('log_dir', os.getenv('SAVEDPATH', '/home/tqc/Desktop') + "/log_dir")


def get_trance_back_msg():
    try:
        txt = cgitb.text(sys.exc_info(), context=12)
    except Exception as e2:
        logger.error(f"cgitb.text Exception:\n{e2}")
        txt = traceback.format_exc()
    logger.error(f"ERROR occur:\n{txt}")
    return txt


class HandleException():
    '''
    异常处理记录类
    '''

    @classmethod
    def handle(cls, e: Exception):
        cls.log_dir = get_log_dir()
        print('Error: ', e)
        try:
            txt = cgitb.text(sys.exc_info(), context=12)  # fixme: 可能出错？
        except Exception as e2:
            print('cgitb.text Exception')
            print(e2)
            txt = traceback.format_exc()
        Path(cls.log_dir).mkdir(parents=True, exist_ok=True)
        file_name = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S_%f') + '.log'
        log_file = Path(cls.log_dir) / file_name
        log_file.write_text(txt)
        print(str(log_file))


def get_info():
    print(__name__)
    print(sys._getframe().f_code.co_name)


def print_run(cmd):
    print(cmd)
    os.system(cmd)


def get_ip():
    hostname = socket.gethostname()
    IPAddr = socket.gethostbyname(hostname)
    print("Your Computer Name is:" + hostname)
    print("Your Computer IP Address is:" + IPAddr)


class EnvUtils:
    def __init__(self):
        self.env_items = []
        self.variables = {}

    def from_json(self, json_path):
        env_items = json.loads(Path(json_path).read_text())
        for env_item in env_items:
            self.add_env(**env_item)

    def add_env(self, name, default, description=""):
        self.env_items.append({
            "name": name,
            "default": default,
            "description": description,
        })
        self.variables[name] = default

    def __getitem__(self, item):
        return self.variables[item]

    def __getattr__(self, item):
        return self.variables[item]

    def update(self):
        for item in self.env_items:
            name = item["name"]
            value = os.getenv(name)
            if value is not None:
                value = value.strip()
                parsed_value = self.parse(value)
                if parsed_value is not None:
                    self.variables[name] = parsed_value

    def parse(self, value: str):
        if value.lower() in ("null", "none", "nan"):
            return None
        try:
            return ast.literal_eval(value)
        except:
            try:
                return json.loads(value)
            except:
                return value

    def get_data(self):
        data = []
        long_data = []
        for k, v in self.variables.items():
            sv = str(v)
            if len(sv) < 20:
                data.append([k, sv])
            else:
                long_data.append([k, v])
        return data, long_data

    def __str__(self):
        data, self.long_data = self.get_data()
        return tabulate(data, headers=["name", "value"])

    def print(self, logger=None):
        if logger is None:
            func = print
        else:
            func = logger.info
        func("\n" + str(self))
        if len(self.long_data) > 0:
            func("--------------------")
            func("| Complex variable |")
            func("--------------------")

            for k, v in self.long_data:
                func(k + " : " + type(v).__name__)
                func(v)
                func("-" * 50)

    __repr__ = __str__
