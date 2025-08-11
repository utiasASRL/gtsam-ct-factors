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

// Constructor test
TEST(WNOAInterp, Constructor) {
  // Interpolation data
  InterpData data;
  data.border_states[0] = StateData(P(0), V(0), 0.0);
  data.border_states[1] = StateData(P(2), V(2), 1.0);
  data.interp_states.push_back(StateData(P(1), V(1), 0.5));
  const auto Q_psd = Vector3::Ones().eval();

  // Create a factor
  Point3 priorValue(0.5, 0.0, 0.0);
  auto model = noiseModel::Diagonal::Sigmas(Vector3::Ones());
  auto prior = PriorFactor<Point3>(P(1), priorValue, model);

  // Construct factor
  const auto factor = WNOAInterpFactor<Point3>(prior, data, Q_psd);
  
}

TEST(WNOAInterp, Print) {
  // Interpolation data
  InterpData data;
  data.border_states[0] = StateData(P(0), V(0), 0.0);
  data.border_states[1] = StateData(P(2), V(2), 1.0);
  data.interp_states.push_back(StateData(P(1), V(1), 0.5));
  const auto Q_psd = Vector3::Ones().eval();

  // Create a factor
  Point3 priorValue(0.5, 0.0, 0.0);
  auto model = noiseModel::Diagonal::Sigmas(Vector3::Ones());
  auto prior = PriorFactor<Point3>(P(1), priorValue, model);

  // Construct factor
  const auto factor = WNOAInterpFactor<Point3>(prior, data, Q_psd);
  

  factor.print();
}

TEST(WNOAInterp, EvalErrorP3Unary) {
  // Interpolation data
  InterpData data;
  data.border_states[0] = StateData(P(0), V(0), 0.0);
  data.border_states[1] = StateData(P(2), V(2), 1.0);
  data.interp_states.push_back(StateData(P(1), V(1), 0.5));
  const auto Q_psd = Vector3::Ones().eval();
  // Create a prior factor
  Point3 priorValue(0.5, 0.0, 0.0);
  const auto model = noiseModel::Diagonal::Sigmas(Vector3::Ones());
  const auto prior = PriorFactor<Point3>(P(1), priorValue, model);
  // Construct factor
  const auto factor = WNOAInterpFactor<Point3>(prior, data, Q_psd);

  // zero velocity case
  Values values;
  values.insert(P(0), p0_p3);
  values.insert(P(2), p0_p3);
  values.insert(V(0), Vector3::Zero().eval());
  values.insert(V(2), Vector3::Zero().eval());
  auto residual1 = factor.unwhitenedError(values);
  auto residual2 = prior.evaluateError(p0_p3);
  CHECK(assert_equal(residual1, residual2, 1e-12));
}

/* *************************************************************************
 */

//  TEST(WNOAInterp, EvalErrorSE3Unary) {
//   // Interpolation data
//   InterpData data;
//   data.border_states[0] = StateData(P(0), V(0), 0.0);
//   data.border_states[1] = StateData(P(2), V(2), 1.0);
//   data.interp_states.push_back(StateData(P(1), V(1), 0.5));
//   const auto Q_psd = Vector6::Ones().eval();
//   // Define First Pose  with a general velocity
//   Pose3 p0_se3 = Pose3::Expmap(Vector6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
//   Vector6 v0_se3(1, 0.0, 0.0, 0.0, 0.0, 0.0);
//   double del_t = 0.1;
//   // Define Second Pose with same velocity (to get an error)
//   Pose3 p1_se3 = p0_se3.expmap(timestep * v0_se3);
//   Vector6 v1_se3 = v0_se3;
//   // Define Third Pose with same vel
//   Pose3 p2_se3 = p0_se3.expmap(2 * timestep * v0_se3);
//   Vector6 v2_se3 = v0_se3;
  
//   // Model 
//   const auto model = noiseModel::Diagonal::Sigmas(Vector3::Ones());
//   const auto prior = PriorFactor<Pose3>(P(1), p1_se3, model);
//   // Construct factor
//   const auto factor =
//       WNOAInterpFactor<typename gtsam::PriorFactor<Point3>, Point3>(prior, data,
//                                                                     Q_psd);

//   // zero velocity case
//   Values values;
//   values.insert(P(0), p0_p3);
//   values.insert(P(2), p0_p3);
//   values.insert(V(0), Vector3::Zero().eval());
//   values.insert(V(2), Vector3::Zero().eval());
//   auto residual1 = factor.unwhitenedError(values);
//   auto residual2 = prior.evaluateError(p0_p3);
//   CHECK(assert_equal(residual1, residual2, 1e-12));
// }



/* *************************************************************************
 */

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}