/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 *  @file  testLocBearingRangeFactor.cpp
 *  @brief Unit tests for LocBearingRangeFactor Class
 *  @author Connor Holmes
 *  @date July 2025
 */

#include <gtsam/sam/LocBearingRangeFactor.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/factorTesting.h>
#include <gtsam/nonlinear/expressionTesting.h>

#include <CppUnitLite/TestHarness.h>

using namespace std;
using namespace gtsam;

namespace {
Key poseKey(1);
Point2 point2(-4.0, 11.0);
Point3 point3(1, 0, 0);

typedef LocBearingRangeFactor<Pose2, Point2> BearingRangeFactor2D;
static SharedNoiseModel model2D(noiseModel::Isotropic::Sigma(2, 0.5));
BearingRangeFactor2D factor2D(poseKey, point2, 1, 2, model2D);

typedef LocBearingRangeFactor<Pose3, Point3> BearingRangeFactor3D;
static SharedNoiseModel model3D(noiseModel::Isotropic::Sigma(3, 0.5));
BearingRangeFactor3D factor3D(poseKey, point3,
                              Pose3().bearing(Point3(1, 0, 0)), 1, model3D);
}

/* ************************************************************************* */
TEST(LocBearingRangeFactor, 2D) {
  // Set the linearization point
  Values values;
  values.insert(poseKey, Pose2(1.0, 2.0, 0.57));

  EXPECT_CORRECT_EXPRESSION_JACOBIANS(factor2D.expression({poseKey}),
                                      values, 1e-7, 1e-5);
  EXPECT_CORRECT_FACTOR_JACOBIANS(factor2D, values, 1e-7, 1e-5);
}

/* ************************************************************************* */
// TODO(frank): this test is disabled (for now) because the macros below are
// incompatible with the Unit3 localCoordinates. See testBearingFactor...
//TEST(BearingRangeFactor, 3D) {
//  // Serialize the factor
//  std::string serialized = serializeXML(factor3D);
//
//  // And de-serialize it
//  BearingRangeFactor3D factor;
//  deserializeXML(serialized, factor);
//
//  // Set the linearization point
//  Values values;
//  values.insert(poseKey, Pose3());
//
//  EXPECT_CORRECT_EXPRESSION_JACOBIANS(factor.expression({poseKey, pointKey}),
//                                      values, 1e-7, 1e-5);
//  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-7, 1e-5);
//}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */
