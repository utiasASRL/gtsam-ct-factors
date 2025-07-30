/**
 * @file    testWNOAFactor.cpp
 * @brief   Unit test for WNOA Factor
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

using namespace std;
using namespace gtsam;

static Vector Q2 = Vector3::Ones();
static Vector Q3 = Vector6::Ones();
static double timestep = 0.1;

/**** SE(2) Test Variables*****/
Vector3 v0_se2(1.0, 2.0, 0.1);
Pose2 p0_se2(1.0, 0.0, 0.5);
// Define Second Pose with 2x the velocity (to get an error)
Pose2 p1_se2 = p0_se2.expmap(timestep * v0_se2);
Vector3 v1_se2 = 2 * v0_se2;

/***** SE(3) Test Variables******/
// Define First Pose  with a general velocity
Vector6 v0_se3(0.1, 0.0, 0.0, 1.0, 0.0, 2.0);
Pose3 p0_se3 = Pose3::Expmap(Vector6(0.5, 0.0, 0.2, 1.0, 0.0, 0.0));
// Define Second Pose with 2x the velocity (to get an error)
Pose3 p1_se3 = p0_se3.expmap(timestep * v0_se3);
Vector6 v1_se3 = 2 * v0_se3;

using symbol_shorthand::P;
using symbol_shorthand::V;

/* ************************************************************************* */
TEST(WNOAFactor, Constructor) {
  WNOAMotionFactor<Pose2> factorSE2(P(1), V(1), P(2), V(2), timestep, Q2);
  WNOAMotionFactor<Pose3> factorSE3(P(1), V(1), P(2), V(2), timestep, Q3);
}

/* *************************************************************************
 */
TEST(WNOAFactor, Equals) {
  // Create two identical factors and make sure they're equal
  WNOAMotionFactor<Pose2> factor1(P(1), V(1), P(2), V(2), timestep, Q2);
  WNOAMotionFactor<Pose2> factor2(P(1), V(1), P(2), V(2), timestep, Q2);

  CHECK(assert_equal(factor1, factor2));
}

/* *************************************************************************
 */

TEST(WNOAFactor, EvalErrorSE2) {
  // Define factor
  WNOAMotionFactor<Pose2> factorSE2(P(1), V(1), P(2), V(2), timestep, Q2);
  // compute error
  Vector actualError(factorSE2.evaluateError(p0_se2, v0_se2, p1_se2, v1_se2));
  // expected is zero
  Vector6 expectedError;
  expectedError << Vector3(0.0, 0.0, 0.0), v0_se2;

  // Actual error depends on the level of accuracy we are using for the expmap
  double tol;
#ifdef GTSAM_SLOW_BUT_CORRECT_EXPMAP
  tol = 1e-9;
#else
  tol = 1e-3;
#endif

  // Verify we get the expected error
  CHECK(assert_equal(expectedError, actualError, tol));
}

/* *************************************************************************
 */

TEST(WNOAFactor, EvalErrorSE3) {
  // Define factor (keys don't matter here)
  WNOAMotionFactor<Pose3> factorSE3(P(1), V(1), P(2), V(2), timestep, Q3);
  // compute error
  Vector actualError(factorSE3.evaluateError(p0_se3, v0_se3, p1_se3, v1_se3));
  // expected is zero
  Vector12 expectedError;
  expectedError << Vector6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), v0_se3;

  // Actual error depends on the level of accuracy we are using for the expmap
  double tol;
#ifdef GTSAM_SLOW_BUT_CORRECT_EXPMAP
  tol = 1e-9;
#else
  tol = 1e-3;
#endif

  // Verify we get the expected error
  CHECK(assert_equal(expectedError, actualError, tol));
}

/* *************************************************************************
 */

TEST(WNOAFactor, ErrorJacobianSE2) {
  // Get analytical derivatives
  Matrix J_p0, J_v0, J_p1, J_v1;
  WNOAMotionFactor<Pose2> factorSE2(P(1), V(1), P(2), V(2), timestep, Q2);
  auto residual = factorSE2.evaluateError(p0_se2, v0_se2, p1_se2, v1_se2, J_p0,
                                          J_v0, J_p1, J_v1);

  // define lambda function for derivatives
  auto f = [&](auto& p0, auto& v0, auto& p1, auto& v1) {
    return factorSE2.evaluateError(p0, v0, p1, v1);
  };

  // Compute numerical derivatives
  Matrix J_p0_num =
      numericalDerivative41<Vector, Pose2, Vector3, Pose2, Vector3>(
          f, p0_se2, v0_se2, p1_se2, v1_se2);
  Matrix J_v0_num =
      numericalDerivative42<Vector, Pose2, Vector3, Pose2, Vector3>(
          f, p0_se2, v0_se2, p1_se2, v1_se2);
  Matrix J_p1_num =
      numericalDerivative43<Vector, Pose2, Vector3, Pose2, Vector3>(
          f, p0_se2, v0_se2, p1_se2, v1_se2);
  Matrix J_v1_num =
      numericalDerivative44<Vector, Pose2, Vector3, Pose2, Vector3>(
          f, p0_se2, v0_se2, p1_se2, v1_se2);

  double tol = 1e-6;
  EXPECT(assert_equal(J_p0, J_p0_num, tol));
  EXPECT(assert_equal(J_v0, J_v0_num, tol));
  EXPECT(assert_equal(J_p1, J_p1_num, tol));
  EXPECT(assert_equal(J_v1, J_v1_num, tol));
}

/* *************************************************************************
 */

TEST(WNOAFactor, ErrorJacobianSE3) {
  // Get analytical derivatives
  Matrix J_p0, J_v0, J_p1, J_v1;
  WNOAMotionFactor<Pose3> factorSE3(P(1), V(1), P(2), V(2), timestep, Q3);
  auto residual = factorSE3.evaluateError(p0_se3, v0_se3, p1_se3, v1_se3, J_p0,
                                          J_v0, J_p1, J_v1);

  // define lambda function for derivatives
  auto f = [&](auto& p0, auto& v0, auto& p1, auto& v1) {
    return factorSE3.evaluateError(p0, v0, p1, v1);
  };

  // Compute numerical derivatives
  Matrix J_p0_num =
      numericalDerivative41<Vector, Pose3, Vector6, Pose3, Vector6>(
          f, p0_se3, v0_se3, p1_se3, v1_se3);
  Matrix J_v0_num =
      numericalDerivative42<Vector, Pose3, Vector6, Pose3, Vector6>(
          f, p0_se3, v0_se3, p1_se3, v1_se3);
  Matrix J_p1_num =
      numericalDerivative43<Vector, Pose3, Vector6, Pose3, Vector6>(
          f, p0_se3, v0_se3, p1_se3, v1_se3);
  Matrix J_v1_num =
      numericalDerivative44<Vector, Pose3, Vector6, Pose3, Vector6>(
          f, p0_se3, v0_se3, p1_se3, v1_se3);

  double tol = 1e-4;
  EXPECT(assert_equal(J_p0, J_p0_num, tol));
  EXPECT(assert_equal(J_v0, J_v0_num, tol));
  EXPECT(assert_equal(J_p1, J_p1_num, tol));
  EXPECT(assert_equal(J_v1, J_v1_num, tol));
}

/* *************************************************************************
 */

TEST(WNOAFactor, OptimizeSE2) {
  // Create a factor graph
  NonlinearFactorGraph graph;
  // Lock the first pose
  graph.emplace_shared<NonlinearEquality<Pose2>>(P(0), p0_se2);
  graph.emplace_shared<NonlinearEquality<Vector3>>(V(0), v0_se2);

  // add WNOA factor
  graph.add(WNOAMotionFactor<Pose2>(P(0), V(0), P(1), V(1), timestep, Q2));
  // Set up initial geuss
  Values values;
  values.insert(P(0), p0_se2);
  values.insert(V(0), v0_se2);
  values.insert(P(1), p1_se2);
  values.insert(V(1), v0_se2);

  // Set up optimizer
  GaussNewtonOptimizer optimizer(graph, values);

  // We expect the initial to be zero because config is the ground truth
  DOUBLES_EQUAL(0.0, optimizer.error(), 1e-9);

  // Iterate once, and the config should not have changed
  optimizer.iterate();
  DOUBLES_EQUAL(0.0, optimizer.error(), 1e-9);

  // Complete solution
  optimizer.optimize();
  DOUBLES_EQUAL(0.0, optimizer.error(), 1e-6);
}

/* ************************************************************************* */

TEST(WNOAFactor, OptimizeSE3) {
  // Create a factor graph
  NonlinearFactorGraph graph;
  // Lock the first pose
  graph.emplace_shared<NonlinearEquality<Pose3>>(P(0), p0_se3);
  graph.emplace_shared<NonlinearEquality<Vector6>>(V(0), v0_se3);

  // add WNOA factor
  graph.add(WNOAMotionFactor<Pose3>(P(0), V(0), P(1), V(1), timestep, Q3));
  // Set up initial geuss
  Values values;
  values.insert(V(0), v0_se3);
  values.insert(P(0), p0_se3);
  values.insert(P(1), p1_se3);
  values.insert(V(1), v0_se3);

  // Set up optimizer
  GaussNewtonOptimizer optimizer(graph, values);

  // We expect the initial to be zero because config is the ground truth
  DOUBLES_EQUAL(0.0, optimizer.error(), 1e-9);

  // Iterate once, and the config should not have changed
  optimizer.iterate();
  DOUBLES_EQUAL(0.0, optimizer.error(), 1e-9);

  // Complete solution
  optimizer.optimize();
  DOUBLES_EQUAL(0.0, optimizer.error(), 1e-6);
}

/* ************************************************************************* */

TEST(WNOAFactor, OptimizeSE3Pert) {
  // Create a factor graph
  NonlinearFactorGraph graph;
  // Lock the first pose
  graph.emplace_shared<NonlinearEquality<Pose3>>(P(0), p0_se3);
  graph.emplace_shared<NonlinearEquality<Vector6>>(V(0), v0_se3);

  // add WNOA factor
  graph.add(WNOAMotionFactor<Pose3>(P(0), V(0), P(1), V(1), timestep, Q3));
  // Set up initial geuss
  Values values;
  values.insert(V(0), v0_se3);
  values.insert(P(0), p0_se3);

  //Perturb the pose and velocity
  values.insert(V(1), v0_se3 + Vector6(1.0,1.0,1.0,1.0,1.0,1.0)*0.1);
  values.insert(P(1), p1_se3.expmap(Vector6(0.001,0.001,0.001,0.1,0.1,0.1)));

  // Set up optimizer
  GaussNewtonOptimizer optimizer(graph, values);

  // Check that we converge to solution
  optimizer.optimize();
  DOUBLES_EQUAL(0.0, optimizer.error(), 1e-6);
}

/* ************************************************************************* */


int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */
