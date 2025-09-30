/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file testNonlinearLikelihood.cpp
 * @date September 30, 2025
 * @author Frank Dellaert
 * @brief unit tests for NonlinearLikelihood factor
 */

#include <gtsam/nonlinear/NonlinearLikelihood.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/base/Testable.h>
#include <gtsam/base/numericalDerivative.h>
#include <CppUnitLite/TestHarness.h>

using namespace std; 
using namespace gtsam;

//******************************************************************************
TEST(NonlinearLikelihood, Constructor) {
  Key key(1);
  Pose2 prior(1, 2, 0.3);
  auto model = noiseModel::Isotropic::Sigma(3, 0.5);
  NonlinearLikelihood<Pose2> factor(key, prior, model);
}

//******************************************************************************
TEST(NonlinearLikelihood, ConstructorWithMean) {
  Key key(1);
  Pose2 prior(1, 2, 0.3);
  auto model = noiseModel::Isotropic::Sigma(3, 0.5);
  Vector mean = (Vector(3) << 0.1, 0.2, 0.3).finished();
  NonlinearLikelihood<Pose2> factor(key, prior, model, mean);
}

//******************************************************************************
TEST(NonlinearLikelihood, Error) {
  Key key(1);
  Pose2 prior(1, 2, 0.3);
  auto model = noiseModel::Isotropic::Sigma(3, 0.5);
  NonlinearLikelihood<Pose2> factor(key, prior, model);

  Pose2 x(1.1, 2.2, 0.3);
  Vector expected_error = (Vector(3) << 0.15463769, 0.161515277, 0.0).finished();
  Vector actual_error = factor.evaluateError(x);
  EXPECT(assert_equal(expected_error, actual_error, 1e-9));
}

//******************************************************************************
TEST(NonlinearLikelihood, ErrorWithMean) {
  Key key(1);
  Pose2 prior(1, 2, 0.3);
  auto model = noiseModel::Isotropic::Sigma(3, 0.5);
  Vector mean = (Vector(3) << 0.1, 0.2, 0.05).finished();
  NonlinearLikelihood<Pose2> factor(key, prior, model, mean);

  Pose2 x(1.1, 2.2, 0.3);
  Vector expected_error = (Vector(3) << 0.05463769, -0.038484723, -0.05).finished();
  Vector actual_error = factor.evaluateError(x);
  EXPECT(assert_equal(expected_error, actual_error, 1e-9));
}

//******************************************************************************
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
//******************************************************************************
