/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file testNonlinearDensity.cpp
 * @date September 30, 2025
 * @author Frank Dellaert
 * @brief Unit tests for NonlinearDensity
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/Testable.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/nonlinear/NonlinearDensity.h>

using namespace std;
using namespace gtsam;

//******************************************************************************
TEST(NonlinearDensity, Pose2) {
  Key key(1);
  Pose2 origin(1, 2, 0.3);
  SharedNoiseModel model = noiseModel::Diagonal::Sigmas(Vector3(0.1, 0.2, 0.3));
  NonlinearDensity<Pose2> factor(key, origin, model);

  // Test error
  Pose2 x = origin;
  Values values;
  values.insert(key, x);
  EXPECT_DOUBLES_EQUAL(0.0, factor.error(values), 1e-9);

  // Test logProbability and evaluate
  double expected_log_prob =
      -0.5 * 3 * log(2 * M_PI) + log(1.0 / (0.1 * 0.2 * 0.3));
  EXPECT_DOUBLES_EQUAL(expected_log_prob, factor.logProbability(x), 1e-9);
  EXPECT_DOUBLES_EQUAL(exp(expected_log_prob), factor.evaluate(x), 1e-9);

  // Test with non-Gaussian noise model
  SharedNoiseModel robust_model = noiseModel::Robust::Create(
      noiseModel::mEstimator::Huber::Create(1.345), model);
  NonlinearDensity<Pose2> robust_factor(key, origin, robust_model);
  CHECK_EXCEPTION(robust_factor.logProbability(x), std::runtime_error);
}

//******************************************************************************
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
//******************************************************************************
