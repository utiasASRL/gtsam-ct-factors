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
    std::function<Matrix(double dt, const VectorN& Q_psd)> inverseCovarianceFunction,
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
Interpolator<PoseType>::interpolatePoseAndVelocity(
    const TimestampedPoseVel& tPoseVel_k,
    const TimestampedPoseVel& tPoseVel_kp1,
    double t_tau,
    OptionalMatrixVecType H,
    const std::shared_ptr<Matrix>& mainSolveMarginalMatrix,
    Matrix* covarianceOut) const {

    // unpack inputs
    auto [t_k, poseVel_k] = tPoseVel_k;
    auto [T_k, varpi_k] = poseVel_k;
    auto [t_kp1, poseVel_kp1] = tPoseVel_kp1;
    auto [T_kp1, varpi_kp1] = poseVel_kp1;

    // if t_tau is equal to t_k or t_kp1, return the corresponding pose and
    // velocity
    if (equal(t_tau, t_k)) {
      if(H)
      {
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
      if(covarianceOut && mainSolveMarginalMatrix) {
        // if t_tau == t_k, then the covariance is the same as that of Tvarpi_k
        *covarianceOut = mainSolveMarginalMatrix->topLeftCorner(dim*2, dim*2);
      }
      return poseVel_k;
      

    } else if (equal(t_tau, t_kp1)) {
      if(H)
      {
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
      if(covarianceOut && mainSolveMarginalMatrix) {
        // if t_tau == t_kp1, then the covariance is the same as that of Tvarpi_kp1
        *covarianceOut = mainSolveMarginalMatrix->bottomRightCorner(dim*2, dim*2);
      }
      return poseVel_kp1;

    } else if (t_tau < t_k || t_tau > t_kp1 || std::isinf(t_k) ||
               std::isinf(t_kp1)) {
      auto poseVel_ex =
          (t_tau < t_k || std::isinf(t_kp1)) ? poseVel_k : poseVel_kp1;
      double t_diff;
      if (t_tau < t_k || std::isinf(t_kp1)) {
        t_diff = t_tau - t_k;
      } else if (t_tau > t_kp1 || std::isinf(t_k)) {
        t_diff = t_tau - t_kp1;
      } else {  // shouldn't happen unless this code is bugged
        throw std::runtime_error(
            "Unexpected case in interpolatePoseAndVelocity");
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

    } else {

      Matrix2N Lambda,
          Psi;  // ensure Lambda and Psi are 2*dim x 2*dim matrices,
                // rather than using auto in the line below
      std::tie(Lambda, Psi) = getLambdaPsi(t_k, t_kp1, t_tau);

      MatrixNx2N Lambda_1 = Lambda.topRows(dim);
      MatrixNx2N Lambda_2 = Lambda.bottomRows(dim);
      MatrixNx2N Psi_1 = Psi.topRows(dim);
      MatrixNx2N Psi_2 = Psi.bottomRows(dim);

      // form quantities for Eq. (11.45) in the book, (5.13) in the paper
      Vector2N gamma_k;
      gamma_k.topRows(dim).setZero();
      gamma_k.bottomRows(dim) = varpi_k;

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

      Vector2N gamma_kp1;
      gamma_kp1 << xi, right_jac_inv * varpi_kp1;

      auto xi_tau = Lambda_1 * gamma_k + Psi_1 * gamma_kp1;
      auto xidot_tau = Lambda_2 * gamma_k + Psi_2 * gamma_kp1;
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
        if constexpr (std::is_same_v<
                          typename traits<PoseType>::structure_category,
                          vector_space_tag>) {
          dxidot_dxi.setZero();
          dvarpitau_dxitau.setZero();
        } else {
          // For Lie groups
          dxidot_dxi = -PoseType::adjointMap(varpi_kp1) / 2.0;
          dvarpitau_dxitau = PoseType::adjointMap(xidot_tau) / 2.0;
        }
        // dgammakp1
        Eigen::Matrix<double, 2 * dim, dim> dgammakp1_dTk;
        dgammakp1_dTk << dxi_dTk, dxidot_dxi * dxi_dTk;
        Eigen::Matrix<double, 2 * dim, dim> dgammakp1_dTkp1;
        dgammakp1_dTkp1 << dxi_dTkp1, dxidot_dxi * dxi_dTkp1;
        Eigen::Matrix<double, 2 * dim, dim> dgammakp1_dvarpikp1;
        dgammakp1_dvarpikp1 << Eigen::Matrix<double, dim, dim>::Zero(),
            right_jac_inv;
        // dxitau
        MatrixN dxitau_dTk = Psi_1 * dgammakp1_dTk;
        MatrixN dxitau_dvarpik = Lambda_1.template block<dim, dim>(0, dim);
        MatrixN dxitau_dTkp1 = Psi_1 * dgammakp1_dTkp1;
        MatrixN dxitau_dvarpikp1 = Psi_1 * dgammakp1_dvarpikp1;
        // dxidottau
        MatrixN dxidottau_dTk = Psi_2 * dgammakp1_dTk;
        MatrixN dxidottau_dvarpik = Lambda_2.template block<dim, dim>(0, dim);
        MatrixN dxidottau_dTkp1 = Psi_2 * dgammakp1_dTkp1;
        MatrixN dxidottau_dvarpikp1 = Psi_2 * dgammakp1_dvarpikp1;

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
        (*H)[5] = right_jac_tau * dxidottau_dvarpik +
                  dvarpitau_dxitau * dxitau_dvarpik;
        // dvarpitau_dTkp1
        (*H)[6] =
            right_jac_tau * dxidottau_dTkp1 + dvarpitau_dxitau * dxitau_dTkp1;
        // dvarpitau_dvarpikp1
        (*H)[7] = right_jac_tau * dxidottau_dvarpikp1 +
                  dvarpitau_dxitau * dxitau_dvarpikp1;
      }
      
      // Output pair
      auto poseVel_tau = PoseVel{T_tau, varpi_tau};

      // compute covariance of the interpolated pose and velocity, if required
      if (mainSolveMarginalMatrix) {
        Eigen::Matrix<double, 2*dim, 4*dim> LambdaPsi;

        // uncomment this block to recompute Lambda and Psi using (5.23) in paper
        // Matrix Lambda_paper(2*dim, 2*dim), Psi_paper(2*dim, 2*dim);
        // Matrix2N Sigma = computeConditionalCov(Tvarpi_k, Tvarpi_kp1, Tvarpi_tau, t_k, t_kp1, t_tau,
        //                                       &Lambda_paper, &Psi_paper);
        // LambdaPsi << Lambda_paper, Psi_paper;

        // use existing Lambda and Psi computed from (11.41) in the book
        Matrix2N Sigma = computeConditionalCov(tPoseVel_k, tPoseVel_kp1, TimestampedPoseVel{t_tau, poseVel_tau});
        LambdaPsi << Lambda, Psi;

        *covarianceOut = Sigma + LambdaPsi * *mainSolveMarginalMatrix * LambdaPsi.transpose();
      }
      return poseVel_tau;
    }
}

template <typename PoseType>
Values Interpolator<PoseType>::interpolatePosesAndVelocities(
    const NonlinearFactorGraph& mainSolveGraph,
    const Values& mainSolveSolution, const StateDataSet& mainSolveStates,
    const StateDataSet& interpolatedStates,
    std::shared_ptr<CovarianceMap> covarianceMapOut) const {

    // Map from intervals [t1, t2) to query times inside that interval (bucket)
    std::map<std::pair<StateData, StateData>, std::vector<StateData>> queryBuckets;

    for (const auto& stateDataInterp : interpolatedStates) {
      auto it2 = mainSolveStates.upper_bound(stateDataInterp);
      if(it2 == mainSolveStates.end()) {
        auto it1 = std::prev(it2);
        queryBuckets[std::make_pair(*it1, StateData::PosInf)].push_back(stateDataInterp);
      }
      else if(it2 == mainSolveStates.begin()) {
        queryBuckets[std::make_pair(StateData::NegInf, *it2)].push_back(stateDataInterp);
      }
      else {
        auto it1 = std::prev(it2);
        queryBuckets[std::make_pair(*it1, *it2)].push_back(stateDataInterp);
      }
    }

    Values interpolatedSolution;

    std::unique_ptr<Marginals>
        marginals;  // Only construct a Marginals if requested
    if (covarianceMapOut) {
      marginals =
          std::make_unique<Marginals>(mainSolveGraph, mainSolveSolution);
    }
    for (const auto& [stateDataBorders, stateDataInterpVec] : queryBuckets) {
      StateData state1 = stateDataBorders.first;
      StateData state2 = stateDataBorders.second;

      // Get the poses and velocities at t_k and t_kp1
      auto pvk = std::isinf(state1.time)
                     ? TimestampedPoseVel(state1.time, PoseVel())
                     : TimestampedPoseVel(state1.time,
                                          mainSolveSolution.at<PoseType>(
                                              state1.pose),
                                          mainSolveSolution.at<VelocityType>(
                                              state1.vel));
      auto pvkp1 = std::isinf(state2.time)
                       ? TimestampedPoseVel(state2.time, PoseVel())
                       : TimestampedPoseVel(state2.time,
                                            mainSolveSolution.at<PoseType>(
                                                state2.pose),
                                            mainSolveSolution.at<VelocityType>(
                                                state2.vel));

      // Compute covariances of the interpolated poses and velocities
      std::shared_ptr<Matrix> mainSolveMarginalMatrix;  // (4*dim, 4*dim)
      Matrix covarianceOut;                             // (2*dim, 2*dim)
      if (covarianceMapOut) {
        // following (5.22) in paper
        KeyVector variables;
        if (std::isinf(state1.time)) {
          variables = {state2.pose, state2.vel};  // {p2, v2}
        } else if (std::isinf(state2.time)) {
          variables = {state1.pose, state1.vel};  // {p1, v1}
        } else {
          variables = {
              // {p1, v1, p2, v2}
              state1.pose, state1.vel, state2.pose, state2.vel};
        }
        JointMarginal mainSolveMarginal =
            marginals->jointMarginalCovariance(variables);
        // avoid using JointMarginal.fullMatrix() as it returns covariance
        // in alphabetical order of the keys...
        mainSolveMarginalMatrix =
            std::make_shared<Matrix>(constructMatrixFromJointMarginal(
                mainSolveMarginal, variables, dim));
      }

      // Interpolate for all query times within this query interval (bucket)
      for (auto stateDataInterp : stateDataInterpVec) {
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
    return interpolatedSolution;
}

template <typename PoseType>
std::pair<Matrix, Matrix>
Interpolator<PoseType>::getLambdaPsi(double t_k, double t_kp1,
                                    double t_tau) const {

    // Build transition matrices for all combinations
    double dt = t_kp1 - t_k;
    auto Phi_12 = transitionFunction_(dt);
    auto Phi_1tau = transitionFunction_(t_tau - t_k);
    auto Phi_tau2 = transitionFunction_(t_kp1 - t_tau);

    // Construct Q
    auto Q_12_inv = inverseCovarianceFunction_(dt, Q_psd_);
    auto Q_1tau = covarianceFunction_(t_tau - t_k, Q_psd_);

    // Eq. (11.41) in the book
    auto Lambda =
        Phi_1tau - Q_1tau * Phi_tau2.transpose() * Q_12_inv * Phi_12;
    auto Psi = Q_1tau * Phi_tau2.transpose() * Q_12_inv;

    return std::make_pair(Lambda, Psi);
}

template <typename PoseType>
typename Interpolator<PoseType>::Matrix2N
Interpolator<PoseType>::computeConditionalCov(
    const TimestampedPoseVel& tPoseVel_k,
    const TimestampedPoseVel& tPoseVel_kp1,
    const TimestampedPoseVel& tPoseVel_tau,
    OptionalMatrixType Lambda,
    OptionalMatrixType Psi) const {
    
    // unpacking then repacking... maybe this can be written better
    auto [t_k, poseVel_k] = tPoseVel_k;
    auto [t_kp1, poseVel_kp1] = tPoseVel_kp1;
    auto [t_tau, poseVel_tau] = tPoseVel_tau;

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
        reordered.block(i * block_size, j * block_size, block_size,
                        block_size) =
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

} // namespace gtsam