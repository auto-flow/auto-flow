#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import os
import re
import shutil
import unittest
from pathlib import Path
from typing import Iterator, Tuple

from autoflow.tests.mock import get_mock_resource_manager


class LocalResourceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        super(LocalResourceTestCase, self).setUp()
        self.mock_resource_manager = get_mock_resource_manager()

    def tearDown(self) -> None:
        shutil.rmtree(self.mock_resource_manager.store_path)


class LogTestCase(LocalResourceTestCase):
    visible_levels = None
    log_name = None

    def setUp(self) -> None:
        super(LogTestCase, self).setUp()
        self.log_file = os.getcwd() + "/" + self.log_name
        self.pattern = re.compile("\[(" + "|".join(self.visible_levels) + ")\]\s\[.*:(.*)\](.*)$", re.MULTILINE)
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def iter_log_items(self)->Iterator[Tuple[str,str,str]]:
        '''
        iterate log items
        Returns
        -------
        result:Iterator[Tuple[str,str,str]]
        (level, logger, msg)
        like: "INFO", "peewee", "SELECT * FROM table;"
        '''
        log_content = Path(self.log_file).read_text()

        for item in self.pattern.finditer(log_content):
            level = item.group(1)
            logger = item.group(2)
            msg = item.group(3)
            msg = msg.strip()
            yield (level, logger, msg)
