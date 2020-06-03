#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from autoflow.tests.base import LocalResourceTestCase
from autoflow.utils.data import is_target_need_label_encode


class TestData(LocalResourceTestCase):
    def test_is_target_need_label_encode(self):
        self.assertTrue(is_target_need_label_encode([1,2,3,3,3,2]))
        self.assertTrue(is_target_need_label_encode([1,2,3,3,3,100]))
        self.assertFalse(is_target_need_label_encode([1,2,3,3,3,0,1,0,2]))
        self.assertTrue(is_target_need_label_encode(["apple", "pear", "banana"]))