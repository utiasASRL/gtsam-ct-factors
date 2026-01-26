"""
GTSAM Copyright 2010-2026, Georgia Tech Research Corporation,
Atlanta, Georgia 30332-0415
All Rights Reserved

See LICENSE for the license information

PseudorangeFactor python binding unit tests.
Author: Sammy Guo
"""
import unittest

import numpy as np

import gtsam
from gtsam.utils.test_case import GtsamTestCase


class TestPriorFactor(GtsamTestCase):

    def test_PseudorangeFactor(self):
        self.assertEqual(1, 0)


if __name__ == "__main__":
    unittest.main()
