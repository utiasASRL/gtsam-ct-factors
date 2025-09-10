/* ----------------------------------------------------------------------------
 * @file Interpolator.cpp
 * -------------------------------------------------------------------------- */

#include "Interpolator.h"

namespace gtsam {

// ---- Constructors ----
template <typename PoseType>
Interpolator<PoseType>::Interpolator(
    const VectorN& Q_psd, std::function<Matrix(double dt)> transitionFunction,
    std::function<Matrix(double dt, const VectorN& Q_psd)> covarianceFunction,
    std::function<Matrix(double dt, const VectorN& Q_psd)>
        inverseCovarianceFunction,
    std::function<Matrix(const std::pair<PoseType, VelocityType>&,
                         const std::pair<PoseType, VelocityType>&, double)>
        computeJacobianPrev,
    std::function<Matrix(const std::pair<PoseType, VelocityType>&,
                         const std::pair<PoseType, VelocityType>&, double)>
        computeJacobianNext)
    : Q_psd_(Q_psd),
      transitionFunction_(transitionFunction),
      covarianceFunction_(covarianceFunction),
      inverseCovarianceFunction_(inverseCovarianceFunction),
      computeJacobianPrev_(computeJacobianPrev),
      computeJacobianNext_(computeJacobianNext) {}

template <typename PoseType>
Interpolator<PoseType>::Interpolator(const VectorN& Q_psd)
    : Interpolator(Q_psd, WNOAMotionFactor<PoseType>::transitionFunction,
                   WNOAMotionFactor<PoseType>::buildWNOACovariance,
                   WNOAMotionFactor<PoseType>::buildInverseWNOACovariance,
                   WNOAMotionFactor<PoseType>::computeJacobianPrev,
                   WNOAMotionFactor<PoseType>::computeJacobianNext) {}

// ---- Member Functions ----
template <typename PoseType>
typename Interpolator<PoseType>::PoseVel
Interpolator<PoseType>::extrapolatePoseAndVelocity(
    const std::optional<TimestampedPoseVel>& tPoseVel_k,
    const std::optional<TimestampedPoseVel>& tPoseVel_kp1,
    double t_tau,
    OptionalMatrixVecType H,
    const std::shared_ptr<Matrix>& mainSolveMarginalMatrix,
    Matrix* covarianceOut) const {
  
  auto poseVel_ex =
      (!tPoseVel_kp1.has_value() || t_tau < tPoseVel_k.value().timestamp)
      ? tPoseVel_k.value().poseVel : tPoseVel_kp1.value().poseVel;
  double t_diff;
  if (!tPoseVel_kp1.has_value() || t_tau < tPoseVel_k.value().timestamp) { // lower than the earliest time
    t_diff = t_tau - tPoseVel_k.value().timestamp;
  } else if (!tPoseVel_k.has_value() || t_tau > tPoseVel_kp1.value().timestamp) { // greater than the latest time
    t_diff = t_tau - tPoseVel_kp1.value().timestamp;
  } else {  // shouldn't happen if this function is called
    throw std::runtime_error(
        "Unexpected case in extrapolatePoseAndVelocity");
  }
  // follow (11.5) in the book
  auto [T_ex, varpi_ex] = poseVel_ex;
  Vector2N gamma_k;
  gamma_k.topRows(dim).setZero();
  gamma_k.bottomRows(dim) = varpi_ex;
  auto Psi = transitionFunction_(t_diff);
  auto gamma_ex = Psi * gamma_k;
  auto T_tau = traits<PoseType>::Compose(
      T_ex, traits<PoseType>::Expmap(gamma_ex.topRows(dim), nullptr));
  auto varpi_tau = gamma_ex.bottomRows(dim);

  if (mainSolveMarginalMatrix) {
    // compute covariance of the extrapolated pose and velocity
    // assume that mainSolveMarginalMatrix corresponds to the covariance of
    // Tvarpi_extrapolate_point
    assert(mainSolveMarginalMatrix->rows() == 2 * dim &&
            mainSolveMarginalMatrix->cols() == 2 * dim);
    Matrix2N Sigma = covarianceFunction_(t_diff, Q_psd_);
    // (11.5) in the book
    *covarianceOut =
        Sigma + Psi * *mainSolveMarginalMatrix * Psi.transpose();
  }
  return PoseVel{T_tau, varpi_tau};
}

template <typename PoseType>
typename Interpolator<PoseType>::PoseVel
Interpolator<PoseType>::interpolatePoseAndVelocity(
    const TimestampedPoseVel& tPoseVel_k,
    const TimestampedPoseVel& tPoseVel_kp1, double t_tau,
    OptionalMatrixVecType H,
    const std::shared_ptr<Matrix>& mainSolveMarginalMatrix,
    Matrix* covarianceOut) const {
  // unpack inputs
  const auto& [poseVel_k, t_k] = tPoseVel_k;
  const auto& [poseVel_kp1, t_kp1] = tPoseVel_kp1;

  // if t_tau is equal to t_k or t_kp1, return the corresponding pose and
  // velocity
  if (t_tau == t_k) {
    // t_tau equal to left boundary time
    return interpolateBoundaryLeft(poseVel_k, H, mainSolveMarginalMatrix,
                                   covarianceOut);

  } else if (t_tau == t_kp1) {
    // t_tau equal to right boundary time
    return interpolateBoundaryRight(poseVel_kp1, H, mainSolveMarginalMatrix,
                                    covarianceOut);

  } else if (t_tau < t_k || std::isinf(t_kp1)) {
    // Extrapolation from left boundary
    double t_diff = t_tau - t_k;
    return extrapolatePoseAndVelocity(poseVel_k, t_diff, H,
                                      mainSolveMarginalMatrix, covarianceOut);
  } else if (t_tau > t_kp1 || std::isinf(t_k)) {
    // Extrapolation from right boundary
    double t_diff = t_tau - t_kp1;
    return extrapolatePoseAndVelocity(poseVel_kp1, t_diff, H,
                                      mainSolveMarginalMatrix, covarianceOut);
  } else {
    // only remaining case is that t_tau is within border states
    // call protected overload of interpolate function
    return interpolatePoseAndVelocity_(tPoseVel_k, tPoseVel_kp1, t_tau, H,
                                       mainSolveMarginalMatrix, covarianceOut);
  }
}

template <typename PoseType>
typename Interpolator<PoseType>::PoseVel
Interpolator<PoseType>::interpolateBoundaryLeft(
    const PoseVelocity<PoseType>& poseVel_k, OptionalMatrixVecType H,
    const std::shared_ptr<Matrix>& mainSolveMarginalMatrix,
    Matrix* covarianceOut) const {
  if (H) {
    // dTtau_dTk
    (*H)[0] = MatrixN::Identity();
    // dTtau_dvarpik
    (*H)[1] = MatrixN::Zero();
    // dTtau_dTkp1
    (*H)[2] = MatrixN::Zero();
    // dTtau_dvarpikp1
    (*H)[3] = MatrixN::Zero();
    // dvarpitau_dTk
    (*H)[4] = MatrixN::Zero();
    // dvarpitau_dvarpik
    (*H)[5] = MatrixN::Identity();
    // dvarpitau_dTkp1
    (*H)[6] = MatrixN::Zero();
    // dvarpitau_dvarpikp1
    (*H)[7] = MatrixN::Zero();
  }
  if (covarianceOut && mainSolveMarginalMatrix) {
    // if t_tau == t_k, then the covariance is the same as that of Tvarpi_k
    *covarianceOut = mainSolveMarginalMatrix->topLeftCorner(dim * 2, dim * 2);
  }
  return poseVel_k;
}

template <typename PoseType>
typename Interpolator<PoseType>::PoseVel
Interpolator<PoseType>::interpolateBoundaryRight(
    const PoseVelocity<PoseType>& poseVel_kp1, OptionalMatrixVecType H,
    const std::shared_ptr<Matrix>& mainSolveMarginalMatrix,
    Matrix* covarianceOut) const {
  if (H) {
    // dTtau_dTk
    (*H)[0] = MatrixN::Zero();
    // dTtau_dvarpik
    (*H)[1] = MatrixN::Zero();
    // dTtau_dTkp1
    (*H)[2] = MatrixN::Identity();
    // dTtau_dvarpikp1
    (*H)[3] = MatrixN::Zero();
    // dvarpitau_dTk
    (*H)[4] = MatrixN::Zero();
    // dvarpitau_dvarpik
    (*H)[5] = MatrixN::Zero();
    // dvarpitau_dTkp1
    (*H)[6] = MatrixN::Zero();
    // dvarpitau_dvarpikp1
    (*H)[7] = MatrixN::Identity();
  }
  if (covarianceOut && mainSolveMarginalMatrix) {
    // if t_tau == t_kp1, then the covariance is the same as that of Tvarpi_kp1
    *covarianceOut =
        mainSolveMarginalMatrix->bottomRightCorner(dim * 2, dim * 2);
  }
  return poseVel_kp1;
}

template <typename PoseType>
typename Interpolator<PoseType>::PoseVel
Interpolator<PoseType>::extrapolatePoseAndVelocity(
    const PoseVelocity<PoseType>& poseVel_ex, double t_diff,
    OptionalMatrixVecType H,
    const std::shared_ptr<Matrix>& mainSolveMarginalMatrix,
    Matrix* covarianceOut) const {
  // follows (11.5) in the book
  const auto& [T_ex, varpi_ex] = poseVel_ex;
  Vector2N gamma_k;
  gamma_k.topRows(dim).setZero();
  gamma_k.bottomRows(dim) = varpi_ex;
  auto Psi = transitionFunction_(t_diff);
  auto gamma_ex = Psi * gamma_k;
  auto T_tau = traits<PoseType>::Compose(
      T_ex, traits<PoseType>::Expmap(gamma_ex.topRows(dim), nullptr));
  auto varpi_tau = gamma_ex.bottomRows(dim);

  if (mainSolveMarginalMatrix) {
    // compute covariance of the extrapolated pose and velocity
    // assume that mainSolveMarginalMatrix corresponds to the covariance of
    // Tvarpi_extrapolate_point
    assert(mainSolveMarginalMatrix->rows() == 2 * dim &&
           mainSolveMarginalMatrix->cols() == 2 * dim);
    Matrix2N Sigma = covarianceFunction_(t_diff, Q_psd_);
    // (11.5) in the book
    *covarianceOut = Sigma + Psi * *mainSolveMarginalMatrix * Psi.transpose();
  }
  return PoseVel{T_tau, varpi_tau};
}

template <typename PoseType>
typename Interpolator<PoseType>::PoseVel
Interpolator<PoseType>::interpolatePoseAndVelocity_(
    const TimestampedPoseVel& tPoseVel_k,
    const TimestampedPoseVel& tPoseVel_kp1, double t_tau,
    OptionalMatrixVecType H,
    const std::shared_ptr<Matrix>& mainSolveMarginalMatrix,
    Matrix* covarianceOut) const {
  // unpack poses and velocities
  const auto& [poseVel_k, t_k] = tPoseVel_k;
  const auto& [poseVel_kp1, t_kp1] = tPoseVel_kp1;
  const auto& [T_k, varpi_k] = poseVel_k;
  const auto& [T_kp1, varpi_kp1] = poseVel_kp1;

  // Retrieve interpolation matrices
  Matrix2N Lambda, Psi;
  std::tie(Lambda, Psi) = getLambdaPsi(t_k, t_kp1, t_tau);

  // form quantities for Eq. (11.45) in the book, (5.13) in the paper
  VectorN xi_k, xi_dot_k;
  xi_k.setZero();
  xi_dot_k = varpi_k;

  // Note that p1 = T(t_k), p2 = T(t_{k+1})
  //  compute xi = log(T_k^-1 T_{k+1})^check
  MatrixN dxi_dTk;
  MatrixN dxi_dTkp1;
  VectorN xi;
  MatrixN right_jac_inv;
  if (H) {
    MatrixN dbetween_Tk;
    MatrixN dbetween_Tkp1;
    xi = traits<PoseType>::Logmap(
        traits<PoseType>::Between(T_k, T_kp1, &dbetween_Tk, &dbetween_Tkp1),
        &right_jac_inv);
    dxi_dTk = right_jac_inv * dbetween_Tk;
    dxi_dTkp1 = right_jac_inv * dbetween_Tkp1;
  } else {
    xi = traits<PoseType>::Logmap(traits<PoseType>::Between(T_k, T_kp1),
                                  &right_jac_inv);
  }

  VectorN xi_kp1, xi_dot_kp1;
  xi_kp1 << xi;
  xi_dot_kp1 = right_jac_inv * varpi_kp1;

  VectorN xi_tau =
      Lambda(0, dim) * xi_dot_k + Psi(0, 0) * xi_kp1 +
      Psi(0, dim) * xi_dot_kp1;  // Dropping xi_k term here since it's zero
  VectorN xidot_tau =
      Lambda(dim, dim) * xi_dot_k + Psi(dim, 0) * xi_kp1 +
      Psi(dim, dim) * xi_dot_kp1;  // Dropping xi_k term here since it's zero
  // Eq. (11.45) in Barfoot 2025
  MatrixN right_jac_tau;
  MatrixN dTtau_dTk;
  MatrixN dTtau_dxitau;
  PoseType T_tau;
  if (H) {
    T_tau = traits<PoseType>::Compose(
        T_k, traits<PoseType>::Expmap(xi_tau, &right_jac_tau), &dTtau_dTk,
        &dTtau_dxitau);
    dTtau_dxitau = dTtau_dxitau * right_jac_tau;
  } else {
    T_tau = traits<PoseType>::Compose(
        T_k, traits<PoseType>::Expmap(xi_tau, &right_jac_tau));
  }
  auto varpi_tau = right_jac_tau * xidot_tau;

  // Compute Jacobians
  if (H) {
    // Derivative of right Jacobians
    // Zero for vector spaces, use an approximation for Lie groups
    MatrixN dxidot_dxi;
    MatrixN dvarpitau_dxitau;
    if constexpr (std::is_same_v<typename traits<PoseType>::structure_category,
                                 vector_space_tag>) {
      dxidot_dxi.setZero();
      dvarpitau_dxitau.setZero();
    } else {
      // For Lie groups
      dxidot_dxi = -PoseType::adjointMap(varpi_kp1) / 2.0;
      dvarpitau_dxitau = PoseType::adjointMap(xidot_tau) / 2.0;
    }
    // dgammakp1
    Eigen::Matrix<double, dim, dim> dxidot_dTk;
    dxidot_dTk << dxidot_dxi * dxi_dTk;
    Eigen::Matrix<double, dim, dim> dxidot_dTkp1;
    dxidot_dTkp1 << dxidot_dxi * dxi_dTkp1;
    Eigen::Matrix<double, dim, dim> dxidotkp1_dvarpikp1;
    dxidotkp1_dvarpikp1 << right_jac_inv;
    // dxitau
    MatrixN dxitau_dTk = Psi(0, 0) * dxi_dTk + Psi(0, dim) * dxidot_dTk;
    MatrixN dxitau_dvarpik =
        Lambda(0, dim) * MatrixN::Identity();  // xi does not depend on varpi_k
                                               // and xi_dot is exactly varpi_k
    MatrixN dxitau_dTkp1 = Psi(0, 0) * dxi_dTkp1 + Psi(0, dim) * dxidot_dTkp1;
    MatrixN dxitau_dvarpikp1 =
        Psi(0, dim) *
        dxidotkp1_dvarpikp1;  // dxikp1 does not depend on varpi_kp1
    // dxidottau
    MatrixN dxidottau_dTk = Psi(dim, 0) * dxi_dTk + Psi(dim, dim) * dxidot_dTk;
    MatrixN dxidottau_dvarpik =
        Lambda(dim, dim) *
        MatrixN::Identity();  // xi does not depend on varpi_k and xi_dot is
                              // exactly varpi_k
    MatrixN dxidottau_dTkp1 =
        Psi(dim, 0) * dxi_dTkp1 + Psi(dim, dim) * dxidot_dTkp1;
    MatrixN dxidottau_dvarpikp1 =
        Psi(dim, dim) *
        dxidotkp1_dvarpikp1;  // dxikp1 does not depend on varpi_kp1

    // dTtau_dTk
    (*H)[0] = dTtau_dTk + dTtau_dxitau * dxitau_dTk;
    // dTtau_dvarpik
    (*H)[1] = dTtau_dxitau * dxitau_dvarpik;
    // dTtau_dTkp1
    (*H)[2] = dTtau_dxitau * dxitau_dTkp1;
    // dTtau_dvarpikp1
    (*H)[3] = dTtau_dxitau * dxitau_dvarpikp1;
    // dvarpitau_dTk
    (*H)[4] = right_jac_tau * dxidottau_dTk + dvarpitau_dxitau * dxitau_dTk;
    // dvarpitau_dvarpik
    (*H)[5] =
        right_jac_tau * dxidottau_dvarpik + dvarpitau_dxitau * dxitau_dvarpik;
    // dvarpitau_dTkp1
    (*H)[6] = right_jac_tau * dxidottau_dTkp1 + dvarpitau_dxitau * dxitau_dTkp1;
    // dvarpitau_dvarpikp1
    (*H)[7] = right_jac_tau * dxidottau_dvarpikp1 +
              dvarpitau_dxitau * dxitau_dvarpikp1;
  }

  // Output pair
  auto poseVel_tau = PoseVel{T_tau, varpi_tau};

  // compute covariance of the interpolated pose and velocity, if required
  if (mainSolveMarginalMatrix) {
    Eigen::Matrix<double, 2 * dim, 4 * dim> LambdaPsi;

    // uncomment this block to recompute Lambda and Psi using (5.23) in paper
    // Matrix Lambda_paper(2*dim, 2*dim), Psi_paper(2*dim, 2*dim);
    // Matrix2N Sigma = computeConditionalCov(Tvarpi_k, Tvarpi_kp1,
    // Tvarpi_tau, t_k, t_kp1, t_tau,
    //                                       &Lambda_paper, &Psi_paper);
    // LambdaPsi << Lambda_paper, Psi_paper;

    // use existing Lambda and Psi computed from (11.41) in the book
    Matrix2N Sigma = computeConditionalCov(
        tPoseVel_k, tPoseVel_kp1, TimestampedPoseVel{poseVel_tau, t_tau});
    LambdaPsi << Lambda, Psi;
    *covarianceOut =
        Sigma + LambdaPsi * *mainSolveMarginalMatrix * LambdaPsi.transpose();
  }
  return poseVel_tau;
}

template <typename PoseType>
std::map<StateDataInterval, std::shared_ptr<Matrix>>
Interpolator<PoseType>::computeJointMarginals(
    const std::map<StateDataInterval, std::vector<StateData>>& queryBuckets,
    const std::unique_ptr<Marginals>& marginals) {
  std::cout << "Computing joint marginals for " << queryBuckets.size() << " intervals." << std::endl;
  std::map<StateDataInterval, std::shared_ptr<Matrix>>
    intervalJointMarginals;  // JointMarginal matrices for each interval
  std::unordered_set<Key> allBoundaryKeys;

  // Lambda function for forming the boundary key vector
  auto formBoundaryKeyVector = [](const StateDataInterval& stateDataBorders) {
    KeyVector boundaryKeyVector;
    if (stateDataBorders.first.has_value()) {
      boundaryKeyVector.push_back(stateDataBorders.first->pose); // p1
      boundaryKeyVector.push_back(stateDataBorders.first->vel); // v1
    }
    if (stateDataBorders.second.has_value()) {
      boundaryKeyVector.push_back(stateDataBorders.second->pose); // p2
      boundaryKeyVector.push_back(stateDataBorders.second->vel); // v2
    }
    return boundaryKeyVector;
  };

  for (const auto& [stateDataBorders, stateDataInterpVec] : queryBuckets) {
    KeyVector boundaryKeyVector = formBoundaryKeyVector(stateDataBorders);
    allBoundaryKeys.insert(boundaryKeyVector.begin(), boundaryKeyVector.end());

    // Method 1: compute JointMarginal for each interval separately
    // faster if there are not too many intervals
    // ----------------------------------
    // JointMarginal mainSolveMarginal =
    // marginals->jointMarginalCovariance(boundaryKeyVector);
    // // avoid using JointMarginal.fullMatrix() as it returns covariance
    // // in alphabetical order of the keys...
    // auto mainSolveMarginalMatrix =
    //   std::make_shared<Matrix>(constructMatrixFromJointMarginal(
    //   mainSolveMarginal, boundaryKeyVector, dim));
    // intervalJointMarginals[stateDataBorders] = mainSolveMarginalMatrix;
    // ----------------------------------
  }

  // Method 2: compute JointMarginal for all boundary keys at once
  // faster if there are many intervals with shared boundary keys
  // ----------------------------------
  JointMarginal allBoundaryMarginal =
      marginals->jointMarginalCovariance(
          KeyVector(allBoundaryKeys.begin(), allBoundaryKeys.end()));
  
  for (const auto& [stateDataBorders, stateDataInterpVec] : queryBuckets) {
    KeyVector boundaryKeyVector = formBoundaryKeyVector(stateDataBorders);
    auto mainSolveMarginalMatrix =
      std::make_shared<Matrix>(constructMatrixFromJointMarginal(
      allBoundaryMarginal, boundaryKeyVector, dim));
    intervalJointMarginals[stateDataBorders] = mainSolveMarginalMatrix;
  }
  // ----------------------------------

  std::cout << "Computed joint marginals for " << intervalJointMarginals.size() << " intervals." << std::endl;
  return intervalJointMarginals;
}

template <typename PoseType>
Values Interpolator<PoseType>::interpolatePosesAndVelocities(
    const NonlinearFactorGraph& mainSolveGraph, const Values& mainSolveSolution,
    const StateDataSet& mainSolveStates, const StateDataSet& interpolatedStates,
    std::shared_ptr<CovarianceMap> covarianceMapOut) const {

    // Map from intervals [t1, t2) to query times inside that interval (bucket)
    std::map<StateDataInterval, std::vector<StateData>> queryBuckets;

    // Find all intervals [t_k, t_kp1) that contain at least one query time
    for (const auto& stateDataInterp : interpolatedStates) {
      auto it2 = mainSolveStates.upper_bound(stateDataInterp);
      StateDataInterval interval;
      interval.first = it2 == mainSolveStates.begin() ? std::nullopt : std::optional<StateData>(*std::prev(it2));
      interval.second = it2 == mainSolveStates.end() ? std::nullopt : std::optional<StateData>(*it2);
      queryBuckets[interval].push_back(stateDataInterp);
    }

    Values interpolatedSolution;

    std::unique_ptr<Marginals> fullMarginal;  // Only construct a Marginals if requested
    // JointMarginal matrices for each interval (every pair of boundary states involved)
    auto intervalJointMarginals = std::map<StateDataInterval, std::shared_ptr<Matrix>>();
    if (covarianceMapOut) {
      // Compute all required joint marginals
      fullMarginal = std::make_unique<Marginals>(mainSolveGraph, mainSolveSolution);
      intervalJointMarginals = computeJointMarginals(queryBuckets, fullMarginal);
    }

    // Perform interpolation for each bucket
    for (const auto& [stateDataBorder, stateDataInterpVec] : queryBuckets) {

      auto makeTimestampedPV = [&](const std::optional<StateData>& s)
          -> std::optional<TimestampedPoseVel> {
        if (!s) return std::nullopt;
        return TimestampedPoseVel(
            mainSolveSolution.at<PoseType>(s->pose),
            mainSolveSolution.at<VelocityType>(s->vel),
            s->time
        );
      };
      
      // Get the poses and velocities at t_k and t_kp1
      std::optional<TimestampedPoseVel> pvk  = makeTimestampedPV(stateDataBorder.first);
      std::optional<TimestampedPoseVel> pvkp1 = makeTimestampedPV(stateDataBorder.second);
      Matrix covarianceOut;                             // (2*dim, 2*dim)

      // Interpolate for all query times within this query interval (bucket)
      for (const auto& stateDataInterp : stateDataInterpVec) {
        std::shared_ptr<Matrix> mainSolveMarginalMatrix;  // (4*dim, 4*dim)
        if (covarianceMapOut) {
          mainSolveMarginalMatrix =
              intervalJointMarginals[stateDataBorder];
        }
        auto pvtau =
            interpolatePoseAndVelocity(pvk, pvkp1, stateDataInterp.time, nullptr,
                                       mainSolveMarginalMatrix, &covarianceOut);
        auto [T_tau, varpi_tau] = pvtau;
        interpolatedSolution.insert(stateDataInterp.pose, T_tau);
        interpolatedSolution.insert(stateDataInterp.vel, varpi_tau);
        if (covarianceMapOut) {
          // upper left covariance block corresponds to pose, lower right block
          // corresponds to velocity
          (*covarianceMapOut)[stateDataInterp.pose] =
              covarianceOut.topLeftCorner(dim, dim);
          (*covarianceMapOut)[stateDataInterp.vel] =
              covarianceOut.bottomRightCorner(dim, dim);
        }
      }
    }
  }
  return interpolatedSolution;
}

template <typename PoseType>
std::pair<Matrix, Matrix> Interpolator<PoseType>::getLambdaPsi(
    double t_k, double t_kp1, double t_tau) const {
  // TODO (SL): This is currently hardcoded for WNOA.
  // we might want to move this code into the WNOA-specific factor and call it?
  double dt = t_kp1 - t_k;
  double alpha = (t_tau - t_k) / dt;
  double alpha2 = alpha * alpha;
  double alpha3 = alpha2 * alpha;

  Matrix2N Lambda, Psi;

  double Psi_00 = 3 * alpha2 - 2 * alpha3;
  double Psi_01 = dt * (alpha3 - alpha2);
  double Psi_10 = (6 * alpha / dt) - (6 * alpha2 / dt);
  double Psi_11 = 3 * alpha2 - 2 * alpha;

  double Lambda_00 = 1 - Psi_00;
  double Lambda_01 = dt * (alpha - 2 * alpha2 + alpha3);
  double Lambda_10 = -Psi_10;
  double Lambda_11 = 1 - 4 * alpha + 3 * alpha2;

  Lambda << MatrixN::Identity() * Lambda_00, MatrixN::Identity() * Lambda_01,
      MatrixN::Identity() * Lambda_10, MatrixN::Identity() * Lambda_11;

  Psi << MatrixN::Identity() * Psi_00, MatrixN::Identity() * Psi_01,
      MatrixN::Identity() * Psi_10, MatrixN::Identity() * Psi_11;

  return std::make_pair(Lambda, Psi);
}

template <typename PoseType>
std::pair<Matrix, Matrix> Interpolator<PoseType>::getLambdaPsiGeneral(
    double t_k, double t_kp1, double t_tau) const {
  // Build transition matrices for all combinations
  double dt = t_kp1 - t_k;
  auto Phi_12 = transitionFunction_(dt);
  auto Phi_1tau = transitionFunction_(t_tau - t_k);
  auto Phi_tau2 = transitionFunction_(t_kp1 - t_tau);

  // Construct Q
  auto Q_12_inv = inverseCovarianceFunction_(dt, Q_psd_);
  auto Q_1tau = covarianceFunction_(t_tau - t_k, Q_psd_);

  // Eq. (11.41) in the book
  auto Lambda = Phi_1tau - Q_1tau * Phi_tau2.transpose() * Q_12_inv * Phi_12;
  auto Psi = Q_1tau * Phi_tau2.transpose() * Q_12_inv;

  return std::make_pair(Lambda, Psi);
}

template <typename PoseType>
typename Interpolator<PoseType>::Matrix2N
Interpolator<PoseType>::computeConditionalCov(
    const TimestampedPoseVel& tPoseVel_k,
    const TimestampedPoseVel& tPoseVel_kp1,
    const TimestampedPoseVel& tPoseVel_tau, OptionalMatrixType Lambda,
    OptionalMatrixType Psi) const {
    
    auto [poseVel_k, t_k] = tPoseVel_k;
    auto [poseVel_kp1, t_kp1] = tPoseVel_kp1;
    auto [poseVel_tau, t_tau] = tPoseVel_tau;

    // Todo (Daniel): handle the case when t_tau == t_k or t_tau == t_kp1
    assert(t_tau > t_k && t_tau < t_kp1 &&
           "t_tau must be in the interval (t_k, t_kp1)");

    // see Figure 5.4 in the paper
    Matrix2N Q_tau_prev_inv = inverseCovarianceFunction_(t_tau - t_k, Q_psd_);
    Matrix2N Q_tau_next_inv = inverseCovarianceFunction_(t_kp1 - t_tau, Q_psd_);
    Matrix2N E_tau = computeJacobianNext_(poseVel_k.asPair(), poseVel_tau.asPair(), t_tau - t_k);
    Matrix2N F_k_tau = computeJacobianPrev_(poseVel_tau.asPair(), poseVel_kp1.asPair(), t_kp1 - t_tau);
    Matrix2N Sigma_inv = E_tau.transpose() * Q_tau_prev_inv * E_tau +
                         F_k_tau.transpose() * Q_tau_next_inv * F_k_tau;
    Matrix2N Sigma = Sigma_inv.inverse();
    
    // (5.23) in the paper
    if (Lambda) {
        Matrix2N F_tau_km1 = computeJacobianPrev_(poseVel_k.asPair(), poseVel_tau.asPair(), t_tau - t_k);
        *Lambda = Sigma * E_tau.transpose() * Q_tau_prev_inv * F_tau_km1;
    }
    if (Psi) {
        Matrix2N E_k = computeJacobianNext_(poseVel_k.asPair(), poseVel_kp1.asPair(), t_kp1 - t_k);
        *Psi = Sigma * F_k_tau.transpose() * Q_tau_next_inv * E_k;
    }
    return Sigma;
}

template <typename PoseType>
Matrix Interpolator<PoseType>::constructMatrixFromJointMarginal(
    const JointMarginal& blockMatrix, const KeyVector& keyVector,
    size_t blockSize) {
  size_t n = keyVector.size();
  Matrix M(n * blockSize, n * blockSize);

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j <= i;
         ++j) {  // symmetric block matrix: j <= i to avoid repeats
      auto block = blockMatrix(keyVector[i], keyVector[j]);  // query block
      // fill in both (i, j) and (j, i) blocks
      M.block(i * blockSize, j * blockSize, blockSize, blockSize) = block;
      if (i != j) {
        M.block(j * blockSize, i * blockSize, blockSize, blockSize) =
            block.transpose();
      }
    }
  }
  return M;
}

template <typename PoseType>
Matrix Interpolator<PoseType>::reorderSymmetricMatrix(
    const Matrix& mat, size_t block_size,
    const std::vector<size_t>& block_order) {
  // This was previously used to reorder a matrix from
  // JointMarginal.fullMatrix() e.g. if KeyVector is {p1, v1, p2, v2},
  // JointMarginal.fullMatrix() would be the marginal corresponding to {v1,
  // v2, p1, p2}. Then, we could call
  // reorderSymmetricMatrix(mainSolveMarginalPermuted, dim, {2, 0, 3, 1});
  assert(mat.rows() == mat.cols() && "Matrix must be square");
  assert(block_order.size() * block_size == static_cast<size_t>(mat.rows()) &&
         "Block order size must match matrix dimensions");
  Matrix reordered(mat.rows(), mat.cols());
  for (size_t i = 0; i < block_order.size(); ++i)
    for (size_t j = 0; j < block_order.size(); ++j)
      reordered.block(i * block_size, j * block_size, block_size, block_size) =
          mat.block(block_order[i] * block_size, block_order[j] * block_size,
                    block_size, block_size);

  return reordered;
}

// ---- Explicit Instantiations ----
template class Interpolator<Point1>;
template class Interpolator<Point2>;
template class Interpolator<Point3>;
template class Interpolator<Pose2>;
template class Interpolator<Pose3>;

}  // namespace gtsam