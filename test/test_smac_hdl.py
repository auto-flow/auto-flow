import unittest
from hyperflow.hdl import smac as smac_hdl

class TestSmacHDL(unittest.TestCase):
    def test_encode(self):
        before = {"name": ["tqc", "dsy"], "type": {(1, 2): {1, 2, 3}}}
        encoded = smac_hdl._encode(before)
        print(encoded)
        after = smac_hdl._decode(encoded)
        print(after)
        self.assertEqual(before,after)