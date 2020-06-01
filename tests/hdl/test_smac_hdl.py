import logging
import unittest
from autoflow.hdl import smac as smac_hdl

class TestSmacHDL(unittest.TestCase):
    def test_encode(self):
        before = {"name": ["tqc", "dsy"], "type": {(1, 2): {1, 2, 3}}}
        encoded = smac_hdl._encode(before)
        logging.info(encoded)
        after = smac_hdl._decode(encoded)
        logging.info(after)
        self.assertEqual(before,after)