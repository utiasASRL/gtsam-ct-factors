/**
 * @file    testWNOAFactor.cpp
 * @brief   Unit test for WNOA Interpolation Factor
 * @author  Connor Holmes
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/WNOAFactor.h>
#include <gtsam/nonlinear/WNOAInterpFactor.h>
#include <gtsam/slam/BetweenFactor.h>

using namespace std;
using namespace gtsam;

// Symbol shorthands
using symbol_shorthand::P;
using symbol_shorthand::V;

static Vector Q_p1 = Vector1::Ones();
static Vector Q_p2 = Vector2::Ones();
static Vector Q_p3 = Vector3::Ones();
static Vector Q_se2 = Vector3::Ones();
static Vector Q_se3 = Vector6::Ones();
static double timestep = 0.1;

/**** Point3 Test Variables*****/
// First pose, moving forward in x with velocity 1
Point3 v0_p3(1.0, 0.0, 0.0);
Vector3 p0_p3(0.0, 0.0, 0.0);
// define next two poses along same trajectory
Point3 p1_p3 = p0_p3 + timestep * v0_p3;
Vector3 v1_p3 = v0_p3;
Point3 p2_p3 = p0_p3 + 2 * timestep * v0_p3;
Vector3 v2_p3 = v0_p3;

/**** SE3 Test Variables *****/
// Define First Pose  with a general velocity
static Pose3 p0_se3 = Pose3::Expmap(Vector6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
static Vector6 v0_se3(1, 0.0, 0.5, 0.1, 0.0, 0.0);
// Define Second Pose with same velocity (to get an error)
static Pose3 p1_se3 = p0_se3.expmap(timestep * v0_se3);
static Vector6 v1_se3 = v0_se3;
// Define Third Pose with same vel
static Pose3 p2_se3 = p0_se3.expmap(2 * timestep * v0_se3);
static Vector6 v2_se3 = v0_se3;

// Default interpolation scheme
static vector<StateData> border = {StateData(P(0), V(0), 0.0),
                                   StateData(P(2), V(2), 2 * timestep)};
static vector<StateData> interp = {StateData(P(1), V(1), timestep)};

// Constructor test
TEST(WNOAInterp, Constructor) {
  // Create a factor
  auto model = noiseModel::Diagonal::Sigmas(Vector3::Ones());
  auto prior = PriorFactor<Point3>(P(1), p0_p3, model);

  // Construct factor
  const auto factor = WNOAInterpFactor<Point3>(prior, border, interp, Q_p3);
  // get factor keys
  KeyVector inner_keys = prior.keys();
  KeyVector outer_keys = factor.keys();
  // Make sure keys defined properly
  CHECK(find(outer_keys.begin(), outer_keys.end(), P(0)) != outer_keys.end());
  CHECK(find(outer_keys.begin(), outer_keys.end(), P(2)) != outer_keys.end());
  CHECK(find(outer_keys.begin(), outer_keys.end(), V(0)) != outer_keys.end());
  CHECK(find(outer_keys.begin(), outer_keys.end(), V(2)) != outer_keys.end());
  CHECK(find(outer_keys.begin(), outer_keys.end(), P(1)) == outer_keys.end());
  CHECK(find(outer_keys.begin(), outer_keys.end(), V(1)) == outer_keys.end());
}

TEST(WNOAInterp, Print) {
  // Create a factor
  Point3 priorValue(0.5, 0.0, 0.0);
  auto model = noiseModel::Diagonal::Sigmas(Vector3::Ones());
  auto prior = PriorFactor<Point3>(P(1), priorValue, model);

  // Construct factor
  const auto factor = WNOAInterpFactor<Point3>(prior, border, interp, Q_p3);

  factor.print();
  cout << endl << endl;
}

TEST(WNOAInterp, EvalErrorP3Unary) {
  // SAME POSE CASE
  // Create a prior factor and interpolated version
  const auto model = noiseModel::Diagonal::Sigmas(Vector3::Ones());
  // prior at first pose
  auto prior = PriorFactor<Point3>(P(1), p0_p3, model);
  auto factor =
      boost::make_shared<WNOAInterpFactor<Point3>>(prior, border, interp, Q_p3);
  Values values;

  values.insert(P(0), p0_p3);
  values.insert(P(2), p0_p3);
  values.insert(V(0), Vector3::Zero().eval());  // zero vel
  values.insert(V(2), Vector3::Zero().eval());
  auto residual1 = factor->unwhitenedError(values);
  auto residual2 = prior.evaluateError(p0_p3);  // same as start pose
  auto res_actual = Vector3::Zero().eval();
  CHECK(assert_equal(residual1, res_actual, 1e-12));
  // CHECK(assert_equal(residual2, res_actual, 1e-12));

  // DIFFERENT POSE, ZERO VELOCITY
  // prior at pose 1 (between 0 and 2)
  prior = PriorFactor<Point3>(P(1), p1_p3, model);
  factor =
      boost::make_shared<WNOAInterpFactor<Point3>>(prior, border, interp, Q_p3);
  values.update(P(0), p0_p3);
  values.update(P(2), p2_p3);
  values.update(V(0), Vector3::Zero().eval());
  values.update(V(2), Vector3::Zero().eval());
  // Evaluate
  residual1 = factor->unwhitenedError(values);
  residual2 = prior.evaluateError(p1_p3);
  CHECK(assert_equal(residual1, res_actual, 1e-12));
  // CHECK(assert_equal(residual2, res_actual, 1e-12));

  // DIFFERENT POSE, WITH VELOCITY
  // prior at pose 1 (between 0 and 2)
  prior = PriorFactor<Point3>(P(1), p1_p3, model);
  factor =
      boost::make_shared<WNOAInterpFactor<Point3>>(prior, border, interp, Q_p3);
  values.update(P(0), p0_p3);
  values.update(P(2), p2_p3);
  values.update(V(0), v0_p3);
  values.update(V(2), v0_p3);
  // Evaluate
  residual1 = factor->unwhitenedError(values);
  residual2 = prior.evaluateError(p1_p3);
  CHECK(assert_equal(residual1, res_actual, 1e-12));
  // CHECK(assert_equal(residual2, res_actual, 1e-12));
}

/* *************************************************************************
 */

TEST(WNOAInterp, EvalErrorSE3UnaryPose) {
  // Model
  const auto model = noiseModel::Diagonal::Sigmas(Vector3::Ones());
  const auto prior_pose = PriorFactor<Pose3>(P(1), p1_se3, model);
  // Construct factor for interpolated pose and velocity
  const auto factor_pose =
      WNOAInterpFactor<Pose3>(prior_pose, border, interp, Q_se3);

  // Set up values
  Values values;
  values.insert(P(0), p0_se3);
  values.insert(P(2), p2_se3);
  values.insert(V(0), v0_se3);
  values.insert(V(2), v2_se3);

  // Check pose residuals
  const auto res_zero = Vector6::Zero();
  auto residual = factor_pose.unwhitenedError(values);  // new
  CHECK(assert_equal(residual, res_zero, 1e-12));
  residual = prior_pose.evaluateError(p1_se3);  // original
  CHECK(assert_equal(residual, res_zero, 1e-12));
}

/* *************************************************************************
 */
TEST(WNOAInterp, EvalErrorSE3UnaryVel) {
  // Model
  const auto model = noiseModel::Diagonal::Sigmas(Vector3::Ones());
  using VelType = traits<Pose3>::TangentVector;
  const auto prior_vel = PriorFactor<VelType>(V(1), v1_se3, model);
  // Construct factor for interpolated pose and velocity
  const auto factor_vel =
      WNOAInterpFactor<Pose3>(prior_vel, border, interp, Q_se3);

  // Set up values
  Values values;
  values.insert(P(0), p0_se3);
  values.insert(P(2), p2_se3);
  values.insert(V(0), v0_se3);
  values.insert(V(2), v2_se3);

  // Check vel residuals
  const auto res_zero = Vector6::Zero();
  auto residual = factor_vel.unwhitenedError(values);  // new
  CHECK(assert_equal(residual, res_zero, 1e-12));
  residual = prior_vel.evaluateError(v1_se3);  // original
  CHECK(assert_equal(residual, res_zero, 1e-12));
}

/* *************************************************************************
 */

TEST(WNOAInterp, EvalErrorSE3BetweenPose) {
  // Model
  const auto model = noiseModel::Diagonal::Sigmas(Vector3::Ones());
  // construct relative pose
  const Pose3 p01_se3 = p0_se3.inverse().compose(p1_se3);
  const auto between_factor = BetweenFactor<Pose3>(P(0), P(1), p01_se3, model);
  // Construct factor for interpolated pose and velocity
  const auto factor =
      WNOAInterpFactor<Pose3>(between_factor, border, interp, Q_se3);

  factor.print();
  // Set up values
  Values values;
  values.insert(P(0), p0_se3);
  values.insert(P(2), p2_se3);
  values.insert(V(0), v0_se3);
  values.insert(V(2), v2_se3);

  // Check pose residuals
  const auto res_zero = Vector6::Zero();
  auto residual = factor.unwhitenedError(values);  // new
  CHECK(assert_equal(residual, res_zero, 1e-12));
  residual = between_factor.evaluateError(p0_se3, p1_se3);  // original
  CHECK(assert_equal(residual, res_zero, 1e-12));
}

/* *************************************************************************
 */

TEST(WNOAInterp, EvalErrorSE3BtwnInterp) {
  // Same as between above, but using two interpolated states with different
  // boundaries
  vector<StateData> border = {StateData(P(0), V(0), 0.0),
                              StateData(P(2), V(2), 2 * timestep),
                              StateData(P(4), V(4), 4 * timestep)};
  vector<StateData> interp = {StateData(P(1), V(1), timestep),
                              StateData(P(3), V(3), 3 * timestep)};
  const Pose3 p3_se3 = p0_se3.expmap(3 * timestep * v0_se3);
  const Pose3 p4_se3 = p0_se3.expmap(4 * timestep * v0_se3);
  Vector6 v2_se3 = v0_se3;
  // Model
  const auto model = noiseModel::Diagonal::Sigmas(Vector3::Ones());
  const auto between_factor =
      BetweenFactor<Pose3>(P(1), P(3), Pose3::Identity(), model);
  // Construct factor for interpolated pose and velocity
  const auto factor =
      WNOAInterpFactor<Pose3>(between_factor, border, interp, Q_se3);

  factor.print();
  // Set up values
  Values values;
  values.insert(P(0), p0_se3);
  values.insert(P(2), p2_se3);
  values.insert(P(4), p4_se3);
  values.insert(V(0), v0_se3);
  values.insert(V(2), v2_se3);
  values.insert(V(4), v0_se3);

  // construct relative pose
  const Pose3 p13_se3 = p1_se3.inverse().compose(p3_se3);
  const Vector6 res_actual = Pose3::Logmap(p13_se3);
  auto residual = between_factor.evaluateError(p1_se3, p3_se3);  // original
  // Check pose residuals
  CHECK(assert_equal(residual, res_actual, 1e-12));
  residual = factor.unwhitenedError(values);  // new
  CHECK(assert_equal(residual, res_actual, 1e-12));
}

/* *************************************************************************
 */

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}