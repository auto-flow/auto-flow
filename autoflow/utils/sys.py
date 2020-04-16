import cgitb
import datetime
import os
import sys
import traceback
from pathlib import Path

from autoflow.utils.logging import get_logger

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
