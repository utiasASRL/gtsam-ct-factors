"""
GTSAM Copyright 2010-2026, Georgia Tech Research Corporation,
Atlanta, Georgia 30332-0415
All Rights Reserved

See LICENSE for the license information

Non-comprehensive Python wrapper smoke tests for constrained optimization.
"""
# pylint: disable=invalid-name, no-name-in-module

import unittest

import numpy as np

import gtsam
from gtsam.symbol_shorthand import X


class TestConstrainedWrappers(unittest.TestCase):
    """Smoke tests for the LP, QP, and QCQP Python bindings."""

    def test_lp_problem_wrapper(self):
        """Construct and solve a tiny LP."""
        x = X(0)
        model = gtsam.noiseModel.Unit.Create(1)

        problem = gtsam.LpProblem()
        problem.addCost(
            gtsam.JacobianFactor(x, np.array([[-1.0]]), np.zeros(1), model)
        )
        problem.addConstraint(
            gtsam.LinearConstraint.LessEqual(
                gtsam.JacobianFactor(x, np.eye(1), np.array([1.0]), model)
            )
        )
        problem.addConstraint(
            gtsam.LinearConstraint.GreaterEqual(
                gtsam.JacobianFactor(x, np.eye(1), np.array([0.0]), model)
            )
        )

        initial = gtsam.Values()
        initial.insert(x, np.array([0.0]))
        result = problem.optimize(initial)

        np.testing.assert_allclose(result.atVector(x), np.array([1.0]))
        self.assertAlmostEqual(problem.objective(result), -1.0)

    def test_qp_problem_wrapper(self):
        """Construct and solve a tiny QP."""
        x = X(0)
        model = gtsam.noiseModel.Unit.Create(2)

        problem = gtsam.QpProblem()
        problem.addCost(
            gtsam.HessianFactor(x, np.eye(2), np.array([2.0, 3.0]), 13.0)
        )
        problem.addConstraint(
            gtsam.LinearConstraint.LessEqual(
                gtsam.JacobianFactor(x, np.eye(2), np.ones(2), model)
            )
        )

        initial = gtsam.Values()
        initial.insert(x, np.zeros(2))
        result = problem.optimize(initial, gtsam.QpSolverType.Sparse)

        np.testing.assert_allclose(result.atVector(x), np.ones(2))
        self.assertEqual(problem.dim(), (1, 0, 2))

    def test_qcqp_problem_wrapper(self):
        """Construct and evaluate a tiny QCQP."""
        x = X(0)
        Q = np.eye(2)

        block_matrix = gtsam.SymmetricBlockMatrix([2], Q)
        cost = gtsam.QpCost(gtsam.KeyVector([x]), block_matrix)
        constraint = gtsam.QuadraticConstraint.Equal(x, Q, 1.0)

        problem = gtsam.QcqpProblem()
        problem.addCost(cost)
        problem.addConstraint(constraint)

        values = gtsam.Values()
        values.insert(x, np.array([1.0, 0.0]))

        self.assertEqual(block_matrix.nBlocks(), 1)
        self.assertTrue(constraint.isEquality())
        self.assertEqual(problem.dim(), (1, 1, 0))
        self.assertEqual(problem.evaluate(values), (0.5, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
