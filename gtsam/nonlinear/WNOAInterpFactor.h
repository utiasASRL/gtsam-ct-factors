#pragma once

#include <gtsam/base/Lie.h>
#include <gtsam/base/Testable.h>
#include <gtsam/base/VectorSpace.h>
#include <gtsam/geometry/Point1.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/Interpolator.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/StateData.h>
#include <gtsam/nonlinear/Values.h>

#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace std;

namespace gtsam {
/* Wrapper class that allows a variable of a factor to be replaced by a WNOA
 * interpolation.
 * It is assumed that estimated state entries are sorted by time.
 * If using for WNOAInterpFactor, note that interpolation is performed for
 * every state in interp_states */
template <class PoseType>
class WNOAInterpFactor : public NoiseModelFactor {
 private:
  using Base = NoiseModelFactor;
  using This = WNOAInterpFactor<PoseType>;
  using VelocityType = typename gtsam::traits<PoseType>::TangentVector;
  static constexpr int dim = traits<PoseType>::dimension;

  // Convenient matrices
  using Matrix2N = Eigen::Matrix<double, 2 * dim, 2 * dim>;
  using MatrixN = Eigen::Matrix<double, dim, dim>;

  // Inner factor that is called on interpolated values
  const NoiseModelFactor::shared_ptr inner_factor_;
  // Interpolator class
  const Interpolator<PoseType> interpolator_;
  // disable noise model updates
  const bool fixed_noise_model_;
  // map keys to interpolated state
  unordered_map<Key, StateData> key_to_interp;
  // map interpolated state to border states.
  unordered_map<StateData, pair<StateData, StateData>> interp_to_borders;

  // map outer key to outer key index (for Jacobians)
  unordered_map<Key, int> outer_key_to_index;

 public:
  /* @brief Constructor of WNOA Interpolation Factor. This factor wraps around a
   * factor that has interpolated states and maps measurements to the bordering
   * estimated states
   * estimated_states and interp_states are sets that are ordered on their
   * defined times. Q_psd is the diagonal of the power spectral density
   * corresponding to the WNOA model.*/
  WNOAInterpFactor(const NoiseModelFactor::shared_ptr inner_factor,
                   const set<StateData> estimated_states,
                   const set<StateData> interp_states,
                   const Eigen::Vector<double, dim> Q_psd,
                   const bool fixed_noise_model = false)
      : Base(inner_factor->noiseModel()),
        inner_factor_(inner_factor),
        interpolator_(Q_psd),
        fixed_noise_model_(fixed_noise_model) {
    // PROCESS INTERPOLATED STATES
    // est state iterator
    auto iter_est_state = estimated_states.begin();
    // loop through interpolated states
    for (const StateData& state : interp_states) {
      // search for estimated state that upper bound current interpolated state
      iter_est_state =
          std::lower_bound(iter_est_state, estimated_states.end(), state);
      if (iter_est_state == estimated_states.begin()) {
        throw runtime_error(
            "Interpolated state time is before all estimated state times");
      } else if (iter_est_state == estimated_states.end()) {
        throw runtime_error(
            "Interpolated state time is after all estimated state times");
      } else {
        // decrement iterator (point to left border state)
        iter_est_state--;
        // map interp to left border index
        interp_to_borders[state] = pair(*iter_est_state, *next(iter_est_state));
        // map keys to interp state
        key_to_interp[state.pose] = state;
        key_to_interp[state.vel] = state;
      }
    }
    // DEFINE KEYS
    // Go through inner keys and add to outer keys accordingly
    unordered_set<Key> outer_key_set;
    for (Key key : inner_factor->keys()) {
      if (key_to_interp.find(key) == key_to_interp.end()) {
        // inner key is not interpolated, add to outer keys
        outer_key_set.insert(key);
      } else {
        // inner key is interpolated, add associated border state keys
        StateData& interp = key_to_interp.at(key);
        auto [left, right] = interp_to_borders.at(interp);
        outer_key_set.insert(left.pose);
        outer_key_set.insert(left.vel);
        outer_key_set.insert(right.pose);
        outer_key_set.insert(right.vel);
      }
    }
    // Convert to key vector
    keys_ = KeyVector(outer_key_set.begin(), outer_key_set.end());

    // map outer keys to their associated index (used when mapping jacobians
    // later)
    for (size_t i = 0; i < this->keys_.size(); i++) {
      outer_key_to_index[this->keys_[i]] = i;
    }
  };

  ~WNOAInterpFactor() override {};

  /** implement functions needed for Testable */
  /** print */
  void print(
      const std::string& s = "",
      const KeyFormatter& keyFormatter = DefaultKeyFormatter) const override {
    cout << s << "WNOAInterpFactor on ";
    for (const auto& k : this->keys()) {
      cout << keyFormatter(k) << " ";  // raw numeric key
    }
    cout << endl;
    this->inner_factor_->print("Inner Factor: ");
  }
  /** equals */
  bool equals(const NonlinearFactor& expected,
              double tol = 1e-9) const override {
    const This* e = dynamic_cast<const This*>(&expected);
    return e != nullptr && Base::equals(*e, tol);
  }

  // Provide access to the Matrix& and the non-Jacobian version of
  // unwhitenedError
  using Base::unwhitenedError;

  // Override of unwhitened error function required by derivatives of
  // NoiseModelFactor
  Vector unwhitenedError(const Values& values,
                         OptionalMatrixVecType H = nullptr) const override {
    return computeInterpolatedError(values, H);
  }

  /* @brief Custom version of linearize function that allows us to update the
   * noise model on-the-fly. This is required to accurately represent the
   * uncertainty of measurements on interpolated states.*/
  std::shared_ptr<GaussianFactor> linearize(const Values& x) const override {
    // if noise model fixed, just use the NonlinearFactor linearize approach.
    if (fixed_noise_model_) {
      return Base::linearize(x);
    }

    // Only linearize if the factor is active
    if (!active(x)) return std::shared_ptr<JacobianFactor>();

    // Call evaluate error to get Jacobians and RHS vector b
    std::vector<Matrix> A(size());
    std::vector<Matrix> JacInner(inner_factor_->size());
    std::unordered_map<StateData, Matrix2N> InterpCondCovs;
    Vector b = -computeInterpolatedError(x, &A, &JacInner, &InterpCondCovs);
    // get interpolated noise model
    auto noise_model = getInterpolatedNoiseModel(JacInner, InterpCondCovs);
    // Whiten the corresponding system now
    noise_model->WhitenSystem(A, b);

    // Fill in terms, needed to create JacobianFactor below
    std::vector<std::pair<Key, Matrix>> terms(size());
    for (size_t j = 0; j < size(); ++j) {
      terms[j].first = keys()[j];
      terms[j].second.swap(A[j]);
    }

    // TODO pass unwhitened + noise model to Gaussian factor
    using noiseModel::Constrained;
    if (noiseModel_ && noiseModel_->isConstrained())
      return GaussianFactor::shared_ptr(new JacobianFactor(
          terms, b,
          std::static_pointer_cast<Constrained>(noiseModel_)->unit()));
    else {
      return GaussianFactor::shared_ptr(new JacobianFactor(terms, b));
    }
  }

  /* @brief Returns the noise model including the increase in covariance due to
   * interpolation. */
  SharedGaussian noiseModel(Values& x) const {
    // if fixed noise then just return the standard gaussian model
    if (fixed_noise_model_) {
      return dynamic_pointer_cast<noiseModel::Gaussian>(Base::noiseModel());
    }
    // Call evaluate error to get inner Jacobians and convariances
    std::vector<Matrix> JacInner(inner_factor_->size());
    std::unordered_map<StateData, Matrix2N> InterpCondCovs;
    Vector b =
        -computeInterpolatedError(x, nullptr, &JacInner, &InterpCondCovs);
    // get interpolated noise model
    return getInterpolatedNoiseModel(JacInner, InterpCondCovs);
  }

 private:
  /* Computes the unwhitened error, and, optionally, the inner factor jacobians
   * and interpolated conditional covariances.*/
  Vector computeInterpolatedError(
      const Values& values, OptionalMatrixVecType H = nullptr,
      OptionalMatrixVecType H_inner = nullptr,
      unordered_map<StateData, Matrix2N>* InterpCondCovs = nullptr) const {
    // Define mapping from interpolated keys to interpolation jacobians
    // Nested Key: map[interp key][estimated key] = Jac
    unordered_map<Key, unordered_map<Key, Matrix>> InterpJacobians;
    // process interpolated states, get Jacobians and covariances
    Values values_interp;
    if (H) {
      // Compute interpolation Jacobians
      values_interp =
          getInterpolatedValues(values, &InterpJacobians, InterpCondCovs);
    } else {
      values_interp = getInterpolatedValues(values, nullptr, InterpCondCovs);
    }

    // construct values for inner factor
    Values values_inner;
    for (const auto& key : inner_factor_->keys()) {
      if (values_interp.exists(key)) {
        // found key in interpolated values
        values_inner.insert(key, values_interp.at(key));
      } else if (values.exists(key)) {
        // found key in values passed to outer factor
        values_inner.insert(key, values.at(key));
      } else {
        // if key is not found then there is a problem
        throw runtime_error("Key " + DefaultKeyFormatter(key) +
                            " not in interpolated states or outer keys");
      }
    }

    // Call inner factor error function with interpolated values.
    Vector error;
    vector<Matrix> H_inner_(inner_factor_->size());
    if (!H_inner) {
      // if H_inner not passed in, use local variable.
      H_inner = &H_inner_;
    }
    if (H || !fixed_noise_model_) {
      error = inner_factor_->unwhitenedError(values_inner, H_inner);
    } else {
      error = inner_factor_->unwhitenedError(values_inner);
    }

    // compute Jacobians for outer keys
    if (H) {
      // loop through inner keys and update outer keys via backpropagation
      // NOTE: it is possible for two inner keys to affect the same outer key
      for (size_t i = 0; i < inner_factor_->keys().size(); i++) {
        Key inner_key = inner_factor_->keys()[i];
        if (values_interp.exists(inner_key)) {
          // Get interpolated state and border states
          const StateData& interp = key_to_interp.at(inner_key);
          const auto& [left, right] = interp_to_borders.at(interp);
          // Create vector of outer keys
          KeyVector outer_keys = {left.pose, left.vel, right.pose, right.vel};
          // Loop over keys and update Jacobian
          for (Key& outer_key : outer_keys) {
            // get position of outer key
            int k = outer_key_to_index.at(outer_key);
            // add to the outer jacobian
            safeMatrixAdd(
                (*H)[k], (*H_inner)[i] * InterpJacobians[inner_key][outer_key]);
          }
        } else {
          // not an interpolated key, directly map to outer jacobian
          int k = outer_key_to_index.at(inner_key);
          safeMatrixAdd((*H)[k], (*H_inner)[i]);
        }
      }
    }

    return error;
  }

  /* @brief Interpolate all interpolated states based on estimated states.
   * Put their values in a Values structure and compute their Jacobians.*/
  Values getInterpolatedValues(
      const Values& values,
      unordered_map<Key, unordered_map<Key, Matrix>>* InterpJacobians = nullptr,
      unordered_map<StateData, Matrix2N>* InterpCondCovs = nullptr) const {
    Values values_interp;  // interpolated values
    // loop through interpolated state map and compute values
    for (const auto& [interp_state, border_states] : interp_to_borders) {
      // unpack border states
      auto& [left, right] = border_states;
      // retrieve estimated state values
      const auto state_left = TimestampedPoseVelocity<PoseType>(
          values.at<PoseType>(left.pose),
          values.at<VelocityType>(left.vel),
          left.time);

      const auto state_right = TimestampedPoseVelocity<PoseType>(
          values.at<PoseType>(right.pose),
          values.at<VelocityType>(right.vel),
          right.time);

      // Get interpolated state velocity pair
      PoseVelocity<PoseType> result;
      vector<Matrix> H(8);
      if (InterpJacobians) {
        result = interpolator_.interpolatePoseAndVelocity(
            state_left, state_right, interp_state.time,
            &H);
      } else {
        result = interpolator_.interpolatePoseAndVelocity(
            state_left, state_right, interp_state.time);
      }

      // insert into values structure
      values_interp.insert(interp_state.pose, result.pose);
      values_interp.insert(interp_state.vel, result.vel);

      // arrange jacobians in unordered map (for easy access later)
      if (InterpJacobians) {
        (*InterpJacobians)[interp_state.pose][left.pose] = H[0];
        (*InterpJacobians)[interp_state.pose][left.vel] = H[1];
        (*InterpJacobians)[interp_state.pose][right.pose] = H[2];
        (*InterpJacobians)[interp_state.pose][right.vel] = H[3];
        (*InterpJacobians)[interp_state.vel][left.pose] = H[4];
        (*InterpJacobians)[interp_state.vel][left.vel] = H[5];
        (*InterpJacobians)[interp_state.vel][right.pose] = H[6];
        (*InterpJacobians)[interp_state.vel][right.vel] = H[7];
      }

      // Conditional covariance of interpolated states for noise model update
      if (InterpCondCovs) {
        auto state_tau = TimestampedPoseVelocity<PoseType>(
            result, interp_state.time);
        Matrix2N Sigma_tau = interpolator_.computeConditionalCov(
            state_left, state_right, state_tau);
        (*InterpCondCovs)[interp_state] =
            Sigma_tau;  // assumed preallocated vector
      }
    }

    return values_interp;
  }

  /* Gets a new noise model for the wrapper factor. This depends on the
   * linearization point and the current estimate of the covariance of the
   * interpolated state */
  SharedGaussian getInterpolatedNoiseModel(
      const vector<Matrix>& Jacobians,
      const unordered_map<StateData, Matrix2N>& InterpCondCovs) const {
    // Get noise model of inner factor
    noiseModel::Gaussian::shared_ptr noise_model_ptr =
        dynamic_pointer_cast<noiseModel::Gaussian>(inner_factor_->noiseModel());
    // Check that the measurement noise is set up as a gaussian
    assert(noise_model_ptr &&
           "Noise model of inner factor must be noiseModel::Gaussian or "
           "derivative");

    // Initialize new covariance with the existing measurement covariance
    int err_dim = noise_model_ptr->dim();
    Matrix noise_cov = noise_model_ptr->covariance();

    // Get mappings from inner keys to index, required to index into jacobian
    // vector
    KeyVector inner_keys = inner_factor_->keys();
    unordered_map<Key, int> key_index;
    for (size_t i = 0; i < inner_keys.size(); i++) {
      key_index[inner_keys[i]] = i;
    }
    // Compute the covariance update based on interpolated states
    // Note: Here, we leverage the block-diagonal approximation of the
    // interpolated covariances (i.e., independence approximation)
    for (auto& [state, borders] : interp_to_borders) {
      // Retrieve Jacobians from inner factor
      Matrix G_pose(err_dim, dim);
      Matrix G_vel(err_dim, dim);
      if (key_index.count(state.pose) > 0) {
        G_pose = Jacobians[key_index[state.pose]];
      } else {
        G_pose.setZero();
      }
      if (key_index.count(state.vel) > 0) {
        G_vel = Jacobians[key_index[state.vel]];
      } else {
        G_vel.setZero();
      }
      Matrix G_tau(err_dim, 2 * dim);
      G_tau << G_pose, G_vel;
      // add covariance
      const Matrix2N& Sigma_tau = InterpCondCovs.at(state);
      noise_cov += G_tau * Sigma_tau * G_tau.transpose();
    }

    // Return interpolated noise model
    return noiseModel::Gaussian::Covariance(noise_cov);
  }

  /* @brief A helper function for safely adding to a Matrix with possibly
   * unknown dimension. If the matrix is undefined, it adopts the dimensions of
   * the matrix that is added .*/
  template <typename DerivedA, typename DerivedB>
  static inline void safeMatrixAdd(Eigen::MatrixBase<DerivedA>& dst,
                                   const Eigen::MatrixBase<DerivedB>& src) {
    if (dst.rows() == 0 && dst.cols() == 0) {
      // First assignment, adopt dimensions
      dst.derived().resize(src.rows(), src.cols());
      dst = src;
    } else {
      assert(dst.rows() == src.rows() && dst.cols() == src.cols() &&
             "Dimension mismatch");
      dst += src;
    }
  }
};

/* Helper function that converts a given graph to another graph with the
 * interpolated states removed. Factors on interpolated states will be
 replaced
 * with factors on the bordering estimated states. WNOAMotionFactors are
 added
 * to all estimated states. Any factors that do not include any interpolated
 * states are added to the new graph, unaltered.  */
template <class PoseType>
NonlinearFactorGraph interpolateFactorGraph(
    const NonlinearFactorGraph& graph, const set<StateData>& estimated_states,
    const set<StateData>& interp_states, Vector Q_psd) {
  // assert that the pose is the right kind of variable
  static_assert(
      std::is_same_v<typename traits<PoseType>::structure_category,
                     lie_group_tag> ||
          std::is_same_v<typename traits<PoseType>::structure_category,
                         vector_space_tag>,
      "Pose type must be either a Lie group or vector space");
  // check dimension on the power spectral density matrix
  assert(traits<PoseType>::dimension == Q_psd.size());
  // Create new factor graph
  NonlinearFactorGraph new_graph;
  // Add WNOA prior between all estimated states
  auto iter_state = estimated_states.begin();
  while (next(iter_state) != estimated_states.end()) {
    StateData state_k = *iter_state;
    StateData state_kp1 = *next(iter_state);
    // get time diff
    double del_t = state_kp1.time - state_k.time;
    // add factor
    auto motion_factor = std::make_shared<WNOAMotionFactor<PoseType>>(
        state_k.pose, state_k.vel, state_kp1.pose, state_kp1.vel, del_t, Q_psd);
    new_graph.add(motion_factor);
    iter_state++;
  }
  // Get map from keys to interpolated state, and interpolated state to
  // estimated state.
  unordered_map<Key, StateData> key_to_interp;
  unordered_map<StateData, pair<StateData, StateData>> interp_to_borders;
  auto iter_est_state = estimated_states.begin();
  for (const StateData& state : interp_states) {
    // search for estimated state that upper bound current interpolated state
    iter_est_state =
        std::lower_bound(iter_est_state, estimated_states.end(), state);
    if (iter_est_state == estimated_states.begin()) {
      throw runtime_error(
          "Interpolated state time is before all estimated state times");
    } else if (iter_est_state == estimated_states.end()) {
      throw runtime_error(
          "Interpolated state time is after all estimated state times");
    } else {
      // decrement iterator (point to left border)
      iter_est_state--;
      // map interp to left border index
      interp_to_borders[state] = pair(*iter_est_state, *next(iter_est_state));
      // map keys to interp state
      key_to_interp[state.pose] = state;
      key_to_interp[state.vel] = state;
    }
  }
  // loop through factors and wrap factors on interpolated states
  for (auto& factor : graph) {
    // handle null factor
    if (!factor) continue;
    // if the factor is a WNOA motion factor, do not add it
    if (dynamic_pointer_cast<WNOAMotionFactor<PoseType>>(factor)) continue;
    // get ordered sets of interpolated and estimated states
    set<StateData> factor_interp_states;
    set<StateData> factor_estimated_states;
    for (Key& key : factor->keys()) {
      // check if key is an interpolated value
      if (key_to_interp.count(key) > 0) {
        // add indices
        StateData interp_state = key_to_interp[key];
        factor_interp_states.insert(interp_state);
        auto [left, right] = interp_to_borders.at(interp_state);
        factor_estimated_states.insert(left);
        factor_estimated_states.insert(right);
      }
    }
    // add factor to new graph
    if (factor_interp_states.size() == 0) {
      // factor does not require interpolation, just add factor as is
      new_graph.add(factor);
    } else {
      // Downcast the NonlinearFactor to a NoiseModelFactor
      auto nmfactor = dynamic_pointer_cast<NoiseModelFactor>(factor);
      assert(nmfactor &&
             "Defined factors must be NoiseModelFactor or derivative class");

      // Define and add factor to new graph
      const auto wrapped_factor = std::make_shared<WNOAInterpFactor<PoseType>>(
          nmfactor, factor_estimated_states, factor_interp_states, Q_psd);
      new_graph.add(wrapped_factor);
    }
  }

  return new_graph;
}

template <class PoseType>
Values updateInterpValues(const NonlinearFactorGraph& interp_graph,
                          const Values& values,
                          const set<StateData>& estim_states,
                          const set<StateData>& interp_states,
                          const Vector Q_psd) {
  // assert that the pose is the right kind of variable
  static_assert(
      std::is_same_v<typename traits<PoseType>::structure_category,
                     lie_group_tag> ||
          std::is_same_v<typename traits<PoseType>::structure_category,
                         vector_space_tag>,
      "Pose type must be either a Lie group or vector space");
  // check dimension on the power spectral density matrix
  assert(traits<PoseType>::dimension == Q_psd.size());
  // Define interpolator
  Interpolator<PoseType> interpolator(Q_psd);
  // get interpolated values
  Values interp_vals = interpolator.interpolatePosesAndVelocities(
      interp_graph, values, estim_states, interp_states);
  // update values
  Values values_updated(values);
  values_updated.insert(interp_vals);
  return values_updated;
}

/// traits
template <class POSE>
struct traits<WNOAInterpFactor<POSE>>
    : public Testable<WNOAInterpFactor<POSE>> {};

}  // namespace gtsam
