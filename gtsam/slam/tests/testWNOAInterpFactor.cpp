/**
 * @file    testWNOAFactor.cpp
 * @brief   Unit test for WNOA Interpolation Factor
 * @author  Connor Holmes
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/WNOAFactor.h>
#include <gtsam/slam/WNOAInterpolationFactor.h>

using namespace std;
using namespace gtsam;

static Vector Q_p1 = Vector1::Ones();
static Vector Q_p2 = Vector2::Ones();
static Vector Q_p3 = Vector3::Ones();
static Vector Q_se2 = Vector3::Ones();
static Vector Q_se3 = Vector6::Ones();
static double timestep = 0.1;

/**** Point3 Test Variables*****/
Point3 v0_p3(5.0, 0.0, 0.0);
Vector3 p0_p3(0.0, 0.0, 0.0);
// Define Second Pose with 2x the velocity (to get an error)
Point3 p1_p3 = p0_p3 + timestep * v0_p3;
Vector3 v1_p3 = 2.0 * v0_p3;

using symbol_shorthand::P;
using symbol_shorthand::V;

TEST(WNOAInterp, evalErr) {
  // Define interpolation data
  InterpData interp_data_p3;
  interp_data_p3.state_keys = {P(0), P(1)};
  interp_data_p3.vel_keys = {V(0), V(1)};
  interp_data_p3.times = {0, 1};
  interp_data_p3.interp_time = 0.5;

  // Create a factor
  Point3 priorValue(0.5, 0.0, 0.0);
  auto model = noiseModel::Diagonal::Sigmas(Vector3::Ones());
  auto prior = PriorFactor<Point3>(P(0), priorValue, model);
  // Embed in wrapper factor
  auto interp_prior =
      WNOAInterpFactor<typename gtsam::PriorFactor<Point3>, Point3, 0>(
          prior, interp_data_p3);
  Values values;
  values.insert(P(0), p0_p3);
  auto residual1 = interp_prior.unwhitenedError(values);
  auto residual2 = prior.evaluateError(p0_p3);
  CHECK(assert_equal(residual1, residual2, 1e-12));
}

/* *************************************************************************
 */

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}