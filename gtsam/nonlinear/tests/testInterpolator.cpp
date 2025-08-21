/**
 * @file    testInterpolator.cpp
 * @brief   Unit tests for post-solve interpolator
 * @author  Zi Cong Guo
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/WNOAFactor.h>
#include <gtsam/nonlinear/Interpolator.h>

using namespace std;
using namespace gtsam;

static Vector1 Q_p1(0.9);
static Vector2 Q_p2(0.9, 0.8);
static Vector3 Q_p3(0.9, 0.8, 0.7);
static Vector3 Q_se2(0.9, 0.8, 0.7);
static Vector6 Q_se3(0.9, 0.8, 0.7, 0.6, 0.5, 0.4);
static double timestep = 0.1;


/**** Point1 Test Variables*****/
Point1 p0_p1(1.0);
Vector1 v0_p1(1.0);
Point1 p1_p1(-2.0);
Vector1 v1_p1(3.0);

/**** Point2 Test Variables*****/
Point2 p0_p2(1.0, 2.0);
Vector2 v0_p2(1.0, 2.0);
Point2 p1_p2(-3.0, -4.0);
Vector2 v1_p2(-2.0, -1.0);

/**** Point3 Test Variables*****/
Point3 p0_p3(1.0, 2.0, 3.0);
Vector3 v0_p3(1.0, 2.0, 3.0);
Point3 p1_p3(-4.0, -5.0, -6.0);
Vector3 v1_p3(-2.0, -1.0, 0.0);

/**** SE(2) Test Variables*****/
Pose2 p0_se2(0.3, 0.2, 0.5);
Vector3 v0_se2(0.8, 0.4, 0.1);
Pose2 p1_se2(-0.2, 0.1, 0.6);
Vector3 v1_se2(0.5, -0.6, 0.2);

/***** SE(3) Test Variables******/
Pose3 p0_se3(Rot3::RzRyRx(0.1, 0.2, 0.3), Point3(0.3, 0.2, 0.1));
Vector6 v0_se3(0.1, 0.2, 0.3, 0.4, 0.5, 0.6);
Pose3 p1_se3(Rot3::RzRyRx(0.4, 0.5, 0.6), Point3(-0.1, -0.2, -0.3));
Vector6 v1_se3(0.2, -0.1, 0.4, 0.6, -0.5, 0.3);

using symbol_shorthand::P;
using symbol_shorthand::V;

/* ************************************************************************* */
TEST(Interpolator, Constructor) {
  // Test that constructors work for different types of poses
  Interpolator<Point1> interpolatorP1(Q_p1);
  Interpolator<Point2> interpolatorP2(Q_p2);
  Interpolator<Point3> interpolatorP3(Q_p3);
  Interpolator<Pose2> interpolatorSE2(Q_se2);
  Interpolator<Pose3> interpolatorSE3(Q_se3);
}

/* ************************************************************************* */
/* COMMON-VELOCITY TESTS
    * These tests check that if the velocities are the same at the two endpoints,
    * and the second pose is obtained by applying the common velocity to the first,
    * then the interpolated velocity should be the same as well, and the pose
    * should be linear interpolation in the tangent space.
*/

TEST(Interpolator, InterpolatePoseAndVelocityP1) {
  Interpolator<Point1> interpolator(Q_p1);
  Vector1 v_common(2.0);
  Point1 p0_p1_common_v = p0_p1;
  Point1 p1_p1_common_v = p0_p1_common_v + timestep * v_common;
  for (double ratio = 0.0; ratio <= 1.0; ratio += 0.1) {
    auto pvtau = interpolator.interpolatePoseAndVelocity(
        std::make_pair(p0_p1_common_v, v_common), 0.0,
        std::make_pair(p1_p1_common_v, v_common), timestep, timestep * ratio);
    Point1 expectedPose = p0_p1_common_v + ratio * timestep * v_common;
    CHECK(assert_equal(expectedPose, pvtau.first));
    CHECK(assert_equal(v_common, pvtau.second));
  }
}
/* ************************************************************************* */
TEST(Interpolator, InterpolatePoseAndVelocityP2) {
  Interpolator<Point2> interpolator(Q_p2);
  Vector2 v_common(0.5, -0.5);
  Point2 p0_p2_common_v = p0_p2;
  Point2 p1_p2_common_v = p0_p2_common_v + timestep * v_common;
  for (double ratio = 0.0; ratio <= 1.0; ratio += 0.1) {
    auto pvtau = interpolator.interpolatePoseAndVelocity(
        std::make_pair(p0_p2_common_v, v_common), 0.0,
        std::make_pair(p1_p2_common_v, v_common), timestep, timestep * ratio);
    Point2 expectedPose = p0_p2_common_v + ratio * timestep * v_common;
    CHECK(assert_equal(expectedPose, pvtau.first));
    CHECK(assert_equal(v_common, pvtau.second));
  }
}
/* ************************************************************************* */
TEST(Interpolator, InterpolatePoseAndVelocityP3) {
  Interpolator<Point3> interpolator(Q_p3);
  Vector3 v_common(0.5, -0.5, 0.5);
  Point3 p0_p3_common_v = p0_p3;
  Point3 p1_p3_common_v = p0_p3_common_v + timestep * v_common;
  for (double ratio = 0.0; ratio <= 1.0; ratio += 0.1) {
    auto pvtau = interpolator.interpolatePoseAndVelocity(
        std::make_pair(p0_p3_common_v, v_common), 0.0,
        std::make_pair(p1_p3_common_v, v_common), timestep, timestep * ratio);
    Point3 expectedPose = p0_p3_common_v + ratio * timestep * v_common;
    CHECK(assert_equal(expectedPose, pvtau.first));
    CHECK(assert_equal(v_common, pvtau.second));
  }
}
/* ************************************************************************* */

TEST(Interpolator, InterpolatePoseAndVelocitySE2) {
  Interpolator<Pose2> interpolator(Q_se2);
  Vector3 v_common(0.65, -0.1, 0.15);
  Pose2 p0_se2_common_v = p0_se2;
  Pose2 p1_se2_common_v = p0_se2_common_v.expmap(timestep * v_common);
  for (double ratio = 0.0; ratio <= 1.0; ratio += 0.1) {
    auto pvtau = interpolator.interpolatePoseAndVelocity(
        std::make_pair(p0_se2_common_v, v_common), 0.0,
        std::make_pair(p1_se2_common_v, v_common), timestep, timestep * ratio);
    Pose2 expectedPose = p0_se2_common_v.expmap(ratio * timestep * v_common);
    CHECK(assert_equal(expectedPose, pvtau.first));
    CHECK(assert_equal(v_common, pvtau.second));
  }
}

/* ************************************************************************* */
TEST(Interpolator, InterpolatePoseAndVelocitySE3) {
  Interpolator<Pose3> interpolator(Q_se3);
  Vector6 v_common(0.15, -0.25, 0.35, 0.45, -0.55, 0.65);
  Pose3 p0_se3_common_v = p0_se3;
  Pose3 p1_se3_common_v = p0_se3_common_v.expmap(timestep * v_common);
  for (double ratio = 0.0; ratio <= 1.0; ratio += 0.1) {
    auto pvtau = interpolator.interpolatePoseAndVelocity(
        std::make_pair(p0_se3_common_v, v_common), 0.0,
        std::make_pair(p1_se3_common_v, v_common), timestep, timestep * ratio);
    Pose3 expectedPose = p0_se3_common_v.expmap(ratio * timestep * v_common);
    double tol = 1e-8;  // larger tolerance since Lie groups have approximations
    CHECK(assert_equal(expectedPose, pvtau.first, tol));
    CHECK(assert_equal(v_common, pvtau.second, tol));
  }
}

/* ************************************************************************* */
/* FORWARD-BACKWARD TESTS
    * These tests check that interpolating forward is the same as interpolating backward with negative velocities.
    * Should be exact for vector spaces (Point1, Point2, Point3), and approximate for Lie groups (Pose2, Pose3).
 */
TEST(Interpolator, ForwardBackwardP1) {
  Interpolator<Point1> interpolator(Q_p1);
  for (double ratio = 0.0; ratio <= 1.0; ratio += 0.1) {
    auto pvtau1 = interpolator.interpolatePoseAndVelocity(
        std::make_pair(p0_p1, v0_p1), 0.0,
        std::make_pair(p1_p1, v1_p1), timestep, timestep * ratio);
    // swap poses and velocities, make velocities negative
    auto pvtau2 = interpolator.interpolatePoseAndVelocity(
        std::make_pair(p1_p1, -v1_p1), 0.0,
        std::make_pair(p0_p1, -v0_p1), timestep, timestep * (1 - ratio));
    double tol = 1e-8;
    CHECK(assert_equal(pvtau1.first, pvtau2.first, tol));
    CHECK(assert_equal(pvtau1.second, (-pvtau2.second).eval(), tol));
  }
}
/* ************************************************************************* */
TEST(Interpolator, ForwardBackwardP2) {
  Interpolator<Point2> interpolator(Q_p2);
  for (double ratio = 0.0; ratio <= 1.0; ratio += 0.1) {
    auto pvtau1 = interpolator.interpolatePoseAndVelocity(
        std::make_pair(p0_p2, v0_p2), 0.0,
        std::make_pair(p1_p2, v1_p2), timestep, timestep * ratio);
    // swap poses and velocities, make velocities negative
    auto pvtau2 = interpolator.interpolatePoseAndVelocity(
        std::make_pair(p1_p2, -v1_p2), 0.0,
        std::make_pair(p0_p2, -v0_p2), timestep, timestep * (1 - ratio));
    double tol = 1e-8;
    CHECK(assert_equal(pvtau1.first, pvtau2.first, tol));
    CHECK(assert_equal(pvtau1.second, (-pvtau2.second).eval(), tol));
  }
}
/* ************************************************************************* */
TEST(Interpolator, ForwardBackwardP3) {
  Interpolator<Point3> interpolator(Q_p3);
  for (double ratio = 0.0; ratio <= 1.0; ratio += 0.1) {
    auto pvtau1 = interpolator.interpolatePoseAndVelocity(
        std::make_pair(p0_p3, v0_p3), 0.0,
        std::make_pair(p1_p3, v1_p3), timestep, timestep * ratio);
    // swap poses and velocities, make velocities negative
    auto pvtau2 = interpolator.interpolatePoseAndVelocity(
        std::make_pair(p1_p3, -v1_p3), 0.0,
        std::make_pair(p0_p3, -v0_p3), timestep, timestep * (1 - ratio));
    double tol = 1e-8;
    CHECK(assert_equal(pvtau1.first, pvtau2.first, tol));
    CHECK(assert_equal(pvtau1.second, (-pvtau2.second).eval(), tol));
  }
}
/* ************************************************************************* */
TEST(Interpolator, ForwardBackwardSE2) {
  Interpolator<Pose2> interpolator(Q_se2);
  for (double ratio = 0.0; ratio <= 1.0; ratio += 0.1) {
    auto pvtau1 = interpolator.interpolatePoseAndVelocity(
        std::make_pair(p0_se2, v0_se2), 0.0,
        std::make_pair(p1_se2, v1_se2), timestep, timestep * ratio);
    // swap poses and velocities, make velocities negative
    auto pvtau2 = interpolator.interpolatePoseAndVelocity(
        std::make_pair(p1_se2, -v1_se2), 0.0,
        std::make_pair(p0_se2, -v0_se2), timestep, timestep * (1 - ratio));
    double tol = 1e-3;  // larger tolerance since Lie groups have approximations
    CHECK(assert_equal(pvtau1.first, pvtau2.first, tol));
    CHECK(assert_equal(pvtau1.second, (-pvtau2.second).eval(), tol));
  }
}
/* ************************************************************************* */
TEST(Interpolator, ForwardBackwardSE3) {
  Interpolator<Pose3> interpolator(Q_se3);
  for (double ratio = 0.0; ratio <= 1.0; ratio += 0.1) {
    auto pvtau1 = interpolator.interpolatePoseAndVelocity(
        std::make_pair(p0_se3, v0_se3), 0.0,
        std::make_pair(p1_se3, v1_se3), timestep, timestep * ratio);
    // swap poses and velocities, make velocities negative
    auto pvtau2 = interpolator.interpolatePoseAndVelocity(
        std::make_pair(p1_se3, -v1_se3), 0.0,
        std::make_pair(p0_se3, -v0_se3), timestep, timestep * (1 - ratio));
    double tol = 1e-2;  // larger tolerance since Lie groups have approximations
    CHECK(assert_equal(pvtau1.first, pvtau2.first, tol));
    CHECK(assert_equal(pvtau1.second, (-pvtau2.second).eval(), tol));
  }
}

/* ************************************************************************* */
/* FRAME TRANSFORMATION TESTS
    * These tests check that the interpolator is agnostic to the frame of reference.
    * The test is done by transforming the poses and velocities to a different frame and checking if the interpolation is consistent.
    * Note: we separate rotation and translation in SE2 and SE3 to avoid coupling effects.
 */
TEST(Interpolator, FrameTranslationP1) {
  Point1 T_frame(0.5);  // simple translation
  Point1 p0_transformed = p0_p1 + T_frame;
  Point1 p1_transformed = p1_p1 + T_frame;
  Interpolator<Point1> interpolator(Q_p1);
  
  for (double ratio = 0.0; ratio <= 1.0; ratio += 0.1) {
    // pvtau1: Interpolate in the original frame
    auto pvtau1 = interpolator.interpolatePoseAndVelocity(
        std::make_pair(p0_p1, v0_p1), 0.0,
        std::make_pair(p1_p1, v1_p1), timestep, timestep * ratio);

    // pvtau2: Interpolate in the transformed frame
    auto pvtau2 = interpolator.interpolatePoseAndVelocity(
        std::make_pair(p0_transformed, v0_p1), 0.0,
        std::make_pair(p1_transformed, v1_p1), timestep, timestep * ratio);

    // Transform pvtau2 back to the original frame
    Point1 pvtau2_pose = pvtau2.first - T_frame;
    Vector1 pvtau2_velocity = pvtau2.second; // body-frame velocity is invariant

    double tol = 1e-8;  // should be exact for Point1
    CHECK(assert_equal(pvtau1.first, pvtau2_pose, tol));
    CHECK(assert_equal(pvtau1.second, pvtau2_velocity, tol));
  }
}
/* ************************************************************************* */
TEST(Interpolator, FrameTranslationP2) {
  Point2 T_frame(0.5, -0.3);  // simple translation
  Point2 p0_transformed = p0_p2 + T_frame;
  Point2 p1_transformed = p1_p2 + T_frame;
  Interpolator<Point2> interpolator(Q_p2);
    for (double ratio = 0.0; ratio <= 1.0; ratio += 0.1) {
        // pvtau1: Interpolate in the original frame
        auto pvtau1 = interpolator.interpolatePoseAndVelocity(
            std::make_pair(p0_p2, v0_p2), 0.0,
            std::make_pair(p1_p2, v1_p2), timestep, timestep * ratio);
    
        // pvtau2: Interpolate in the transformed frame
        auto pvtau2 = interpolator.interpolatePoseAndVelocity(
            std::make_pair(p0_transformed, v0_p2), 0.0,
            std::make_pair(p1_transformed, v1_p2), timestep, timestep * ratio);
        // Transform pvtau2 back to the original frame
        Point2 pvtau2_pose = pvtau2.first - T_frame;
        Vector2 pvtau2_velocity = pvtau2.second; // body-frame velocity is invariant
        double tol = 1e-8;  // should be exact for Point2
        CHECK(assert_equal(pvtau1.first, pvtau2_pose, tol));
        CHECK(assert_equal(pvtau1.second, pvtau2_velocity, tol));
    }
}
/* ************************************************************************* */
TEST(Interpolator, FrameTranslationP3) {
  Point3 T_frame(0.5, -0.3, 0.2);  // simple translation
  Point3 p0_transformed = p0_p3 + T_frame;
  Point3 p1_transformed = p1_p3 + T_frame;
  Interpolator<Point3> interpolator(Q_p3);
    for (double ratio = 0.0; ratio <= 1.0; ratio += 0.1) {
        // pvtau1: Interpolate in the original frame
        auto pvtau1 = interpolator.interpolatePoseAndVelocity(
            std::make_pair(p0_p3, v0_p3), 0.0,
            std::make_pair(p1_p3, v1_p3), timestep, timestep * ratio);
        // pvtau2: Interpolate in the transformed frame
        auto pvtau2 = interpolator.interpolatePoseAndVelocity(
            std::make_pair(p0_transformed, v0_p3), 0.0,
            std::make_pair(p1_transformed, v1_p3), timestep, timestep * ratio);
        // Transform pvtau2 back to the original frame
        Point3 pvtau2_pose = pvtau2.first - T_frame;
        Vector3 pvtau2_velocity = pvtau2.second; // body-frame velocity is invariant
        double tol = 1e-8;  // should be exact for Point3
        CHECK(assert_equal(pvtau1.first, pvtau2_pose, tol));
        CHECK(assert_equal(pvtau1.second, pvtau2_velocity, tol));
    }
}
/* ************************************************************************* */
TEST(Interpolator, FrameTranslationSE2) {
  Point2 T_frame(0.5, -0.3);  // simple translation
  Pose2 p0_transformed = Pose2(Rot2::Identity(), T_frame).compose(p0_se2);
  Pose2 p1_transformed = Pose2(Rot2::Identity(), T_frame).compose(p1_se2);
  Interpolator<Pose2> interpolator(Q_se2);
    for (double ratio = 0.0; ratio <= 1.0; ratio += 0.1) {
        // pvtau1: Interpolate in the original frame
        auto pvtau1 = interpolator.interpolatePoseAndVelocity(
            std::make_pair(p0_se2, v0_se2), 0.0,
            std::make_pair(p1_se2, v1_se2), timestep, timestep * ratio);
        // pvtau2: Interpolate in the transformed frame
        auto pvtau2 = interpolator.interpolatePoseAndVelocity(
            std::make_pair(p0_transformed, v0_se2), 0.0,
            std::make_pair(p1_transformed, v1_se2), timestep, timestep * ratio);
        // Transform pvtau2 back to the original frame
        Pose2 pvtau2_pose = Pose2(Rot2::Identity(), -T_frame).compose(pvtau2.first);
        Vector3 pvtau2_velocity = pvtau2.second; // body-frame velocity is invariant
        double tol = 1e-8;  // should be exact for SE(2) translation
        CHECK(assert_equal(pvtau1.first, pvtau2_pose, tol));
        CHECK(assert_equal(pvtau1.second, pvtau2_velocity, tol));
    }
}
/* ************************************************************************* */
TEST(Interpolator, FrameTranslationSE3) {
  Point3 T_frame(0.5, -0.3, 0.2);  // simple translation
  Pose3 p0_transformed = Pose3(Rot3::Identity(), T_frame).compose(p0_se3);
  Pose3 p1_transformed = Pose3(Rot3::Identity(), T_frame).compose(p1_se3);
  Interpolator<Pose3> interpolator(Q_se3);
    for (double ratio = 0.0; ratio <= 1.0; ratio += 0.1) {
        // pvtau1: Interpolate in the original frame
        auto pvtau1 = interpolator.interpolatePoseAndVelocity(
            std::make_pair(p0_se3, v0_se3), 0.0,
            std::make_pair(p1_se3, v1_se3), timestep, timestep * ratio);
        // pvtau2: Interpolate in the transformed frame
        auto pvtau2 = interpolator.interpolatePoseAndVelocity(
            std::make_pair(p0_transformed, v0_se3), 0.0,
            std::make_pair(p1_transformed, v1_se3), timestep, timestep * ratio);
        // Transform pvtau2 back to the original frame
        Pose3 pvtau2_pose = Pose3(Rot3::Identity(), -T_frame).compose(pvtau2.first);
        Vector6 pvtau2_velocity = pvtau2.second; // body-frame velocity is invariant
        double tol = 1e-8;  // should be exact for SE(3) translation
        CHECK(assert_equal(pvtau1.first, pvtau2_pose, tol));
        CHECK(assert_equal(pvtau1.second, pvtau2_velocity, tol));
    }
}
/* ************************************************************************* */
TEST(Interpolator, FrameRotationSE2) {
  Pose2 T_frame(Rot2(0.5), Point2(0.0, 0.0));
  Pose2 p0_transformed = T_frame.compose(p0_se2);
  Pose2 p1_transformed = T_frame.compose(p1_se2);
  Vector3 Q_se2_isotropic = Vector3::Ones() * 0.8;  // isotropic noise for SE(2)
  // Use an isotropic noise to avoid a non-isotropic noise in the new frame
  Interpolator<Pose2> interpolator(Q_se2_isotropic);

    for (double ratio = 0.0; ratio <= 1.0; ratio += 0.1) {
        // pvtau1: Interpolate in the original frame
        auto pvtau1 = interpolator.interpolatePoseAndVelocity(
            std::make_pair(p0_se2, v0_se2), 0.0,
            std::make_pair(p1_se2, v1_se2), timestep, timestep * ratio);

        // pvtau2: Interpolate in the transformed frame
        auto pvtau2 = interpolator.interpolatePoseAndVelocity(
            std::make_pair(p0_transformed, v0_se2), 0.0,
            std::make_pair(p1_transformed, v1_se2), timestep, timestep * ratio);

        // Transform pvtau2 back to the original frame
        Pose2 pvtau2_pose = T_frame.inverse().compose(pvtau2.first);
        Vector3 pvtau2_velocity = pvtau2.second; // body-frame velocity is invariant

        double tol = 1e-3;  // larger tolerance since Lie groups have approximations
        CHECK(assert_equal(pvtau1.first, pvtau2_pose, tol));
        CHECK(assert_equal(pvtau1.second, pvtau2_velocity, tol));
    }
}

/* ************************************************************************* */
TEST(Interpolator, FrameRotationSE3) {
  Pose3 T_frame(Rot3::RzRyRx(0.3, 0.2, 0.1), Point3(0.0, 0.0, 0.0));
  Pose3 p0_transformed = T_frame.compose(p0_se3);
  Pose3 p1_transformed = T_frame.compose(p1_se3);
  Vector6 Q_se3_isotropic = Vector6::Ones() * 0.8;  // isotropic noise for SE(3)
  // Use an isotropic noise to avoid a non-isotropic noise in the new frame
  Interpolator<Pose3> interpolator(Q_se3_isotropic);
    for (double ratio = 0.0; ratio <= 1.0; ratio += 0.1) {
        // pvtau1: Interpolate in the original frame
        auto pvtau1 = interpolator.interpolatePoseAndVelocity(
            std::make_pair(p0_se3, v0_se3), 0.0,
            std::make_pair(p1_se3, v1_se3), timestep, timestep * ratio);

        // pvtau2: Interpolate in the transformed frame
        auto pvtau2 = interpolator.interpolatePoseAndVelocity(
            std::make_pair(p0_transformed, v0_se3), 0.0,
            std::make_pair(p1_transformed, v1_se3), timestep, timestep * ratio);
        // Transform pvtau2 back to the original frame
        Pose3 pvtau2_pose = T_frame.inverse().compose(pvtau2.first);
        Vector6 pvtau2_velocity = pvtau2.second; // body-frame velocity is invariant
        double tol = 1e-2;  // larger tolerance since Lie groups have approximations
        CHECK(assert_equal(pvtau1.first, pvtau2_pose, tol));
        CHECK(assert_equal(pvtau1.second, pvtau2_velocity, tol));
    }
}

/* ************************************************************************* */
/* Todo: write tests for covariance interpolation,
check that computation of psi and lambda using Eq. (11.41) in the book is consistent
with Eq. (5.23) in the paper. */

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */
