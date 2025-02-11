#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""
This module provides unit-level tests for the XGB forecast model specification
file registry.
"""

import unittest

from xgboost import Booster

from wqf import xgb


class RegistryTest(unittest.TestCase):

    def test_registry(self):
        reg = xgb.registry()

        self.assertTrue("default" in reg)
        self.assertIsInstance(reg.model("default"), Booster)

        self.assertTrue("ns-central" in reg)
        self.assertIsInstance(reg.model("ns-central"), Booster)

        self.assertTrue("ns-coastal" in reg)
        self.assertIsInstance(reg.model("ns-coastal"), Booster)

        self.assertTrue("ns-natural" in reg)
        self.assertIsInstance(reg.model("ns-natural"), Booster)

        self.assertFalse("unregistered" in reg)
        self.assertRaises(KeyError, reg.model, "unregistered")


if __name__ == "__main__":
    unittest.main()
