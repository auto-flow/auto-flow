#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import shutil
import unittest

from autoflow.tests.mock import get_mock_resource_manager


class LocalResourceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        super(LocalResourceTestCase, self).setUp()
        self.mock_resource_manager = get_mock_resource_manager()

    def tearDown(self) -> None:
        shutil.rmtree(self.mock_resource_manager.store_path)
