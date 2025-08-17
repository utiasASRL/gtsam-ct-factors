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
#include <gtsam/nonlinear/Values.h>

#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace std;

namespace gtsam {

/* @brief State data structure for keeping track of pose and velocity keys as
 * well as associated timestamp. Used in GP interpolation.*/
struct StateData {
  const Key pose;
  const Key vel;
  const double time;
  // Default constructor for easy init
  StateData() = default;
  // Constructor
  StateData(Key pose_in, Key vel_in, double time_in)
      : pose(pose_in), vel(vel_in), time(time_in) {};
};

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
  const NoiseModelFactor& inner_factor_;
  // List of interpolated states
  const vector<StateData>& interp_states_;
  // List of estimated states
  const vector<StateData>& estimated_states_;
  // Interpolator class
  const Interpolator<PoseType> interpolator_;
  // disable noise model updates
  const bool fixed_noise_model_;
  // map interpolated key to index of left estimated state (required for
  // jacobians)
  unordered_map<Key, int> interp_key_to_left;
  // map interpolated state index to estimated state index
  vector<int> interp_to_estimated;
  // map outer key to outer key index (for Jacobians)
  unordered_map<Key, int> outer_key_to_index;

 public:
  WNOAInterpFactor(const NoiseModelFactor& inner_factor,
                   const vector<StateData>& estimated_states,
                   const vector<StateData>& interp_states,
                   const Eigen::Vector<double, dim>& Q_psd,
                   const bool fixed_noise_model = false)
      : Base(inner_factor.noiseModel()),
        inner_factor_(inner_factor),
        interp_states_(interp_states),
        estimated_states_(estimated_states),
        interpolator_(Q_psd),
        fixed_noise_model_(fixed_noise_model) {
    
    
    // PROCESS INTERPOLATED STATES
    vector<double> estimated_times; // vector of estimated times
    for (auto state : estimated_states) {
      estimated_times.push_back(state.time);
    }
    for (uint i = 0; i < interp_states.size(); i++) {
      StateData state = interp_states[i];
      // Search for associated bordering states
      auto it = std::lower_bound(estimated_times.begin(), estimated_times.end(),
                                 state.time);
      if (it == estimated_times.begin()) {
        throw runtime_error(
            "Interpolated state time is before all estimated state times");
      } else if (it == estimated_times.end()) {
        throw runtime_error(
            "Interpolated state time is after all estimated state times");
      } else {
        // update maps with left index
        int ind = std::distance(estimated_times.begin(), it);
        interp_to_estimated.push_back(ind - 1);
        interp_key_to_left[state.pose] = ind - 1;
        interp_key_to_left[state.vel] = ind - 1;
      }
    }

    // DEFINE KEYS
    // Go through inner keys and add to outer keys accordingly
    unordered_set<Key> outer_key_set;
    for (Key key : inner_factor.keys()){
      if (interp_key_to_left.find(key) == interp_key_to_left.end()) {
        // inner key is not interpolated, add to outer keys
        outer_key_set.insert(key);
      } else {
        // inner key is interpolated, add associated border state keys
        int left_ind = interp_key_to_left[key]; // left border index
        outer_key_set.insert(estimated_states[left_ind].pose);
        outer_key_set.insert(estimated_states[left_ind].vel);
        outer_key_set.insert(estimated_states[left_ind+1].pose);
        outer_key_set.insert(estimated_states[left_ind+1].vel);
      }
    }
    keys_ = KeyVector(outer_key_set.begin(), outer_key_set.end());

    // map outer keys to their associated index (used when mapping jacobians
    // later)
    for (uint i = 0; i < this->keys_.size(); i++) {
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
    this->inner_factor_.print("Inner Factor: ");
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
    std::vector<Matrix> JacInner(inner_factor_.size());
    std::vector<Matrix2N> InterpCondCovs(interp_states_.size());
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
    std::vector<Matrix> JacInner(inner_factor_.size());
    std::vector<Matrix2N> InterpCondCovs(interp_states_.size());
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
      vector<Matrix2N>* InterpCondCovs = nullptr) const {
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
    for (const auto& key : inner_factor_.keys()) {
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
    vector<Matrix> H_inner_(inner_factor_.size());
    if (!H_inner) {
      // if H_inner not passed in, use local variable.
      H_inner = &H_inner_;
    }
    if (H || !fixed_noise_model_) {
      error = inner_factor_.unwhitenedError(values_inner, H_inner);
    } else {
      error = inner_factor_.unwhitenedError(values_inner);
    }

    // compute Jacobians for outer keys
    if (H) {
      // loop through inner keys and update outer keys (backpropagate)
      for (uint i = 0; i < inner_factor_.keys().size(); i++) {
        Key inner_key = inner_factor_.keys()[i];
        if (values_interp.exists(inner_key)) {
          // get index left bordering state states
          int left_index = interp_key_to_left.at(inner_key);
          // get associated outer keys
          KeyVector outer_keys = {estimated_states_[left_index].pose,
                                  estimated_states_[left_index].vel,
                                  estimated_states_[left_index + 1].pose,
                                  estimated_states_[left_index + 1].vel};
          // Loop over keys and update Jacobian
          for (Key& outer_key : outer_keys) {
            int k = outer_key_to_index.at(outer_key);
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
      vector<Matrix2N>* InterpCondCovs = nullptr) const {
    Values values_interp;  // interpolated values
    // loop through interpolated states and compute interpolation
    for (uint i = 0; i < interp_states_.size(); i++) {
      const StateData& interp_state = interp_states_[i];  // interpolated state
      int index = interp_to_estimated[i];  // index of bordering estimated state
      const StateData& left = estimated_states_[index];  // left border state
      const StateData& right =
          estimated_states_[index + 1];  // right border state
      // retrieve estimated states
      const auto state_left = make_pair(values.at<PoseType>(left.pose),
                                        values.at<VelocityType>(left.vel));
      const auto state_right = make_pair(values.at<PoseType>(right.pose),
                                         values.at<VelocityType>(right.vel));

      // Get interpolated state velocity pair
      PoseType pose;
      VelocityType velocity;
      vector<Matrix> H(8);
      if (InterpJacobians) {
        tie(pose, velocity) = interpolator_.interpolatePoseAndVelocity(
            state_left, left.time, state_right, right.time, interp_state.time,
            &H);
      } else {
        tie(pose, velocity) = interpolator_.interpolatePoseAndVelocity(
            state_left, left.time, state_right, right.time, interp_state.time);
      }

      // insert into values structure
      values_interp.insert(interp_state.pose, pose);
      values_interp.insert(interp_state.vel, velocity);

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
        Matrix2N Sigma_tau = interpolator_.computeConditionalCov(
            state_left, state_right, pair(pose, velocity), left.time,
            right.time, interp_state.time);
        (*InterpCondCovs)[i] = Sigma_tau;  // assumed preallocated vector
      }
    }

    return values_interp;
  }

  /* Gets a new noise model for the wrapper factor. This depends on the
   * linearization point and the current estimate of the covariance of the
   * interpolated state */
  SharedGaussian getInterpolatedNoiseModel(
      const vector<Matrix>& Jacobians,
      const vector<Matrix2N>& InterpCondCovs) const {
    // Get noise model of inner factor
    noiseModel::Gaussian::shared_ptr noise_model_ptr =
        dynamic_pointer_cast<noiseModel::Gaussian>(inner_factor_.noiseModel());
    // Check that the measurement noise is set up as a gaussian
    assert(noise_model_ptr &&
           "Noise model of inner factor must be noiseModel::Gaussian or "
           "derivative");

    // Initialize new covariance with the existing measurement covariance
    int err_dim = noise_model_ptr->dim();
    Matrix noise_cov = noise_model_ptr->covariance();

    // Compute the covariance update based on interpolated states
    KeyVector inner_keys = inner_factor_.keys();
    unordered_map<Key, int> key_index;
    for (uint i = 0; i < inner_keys.size(); i++) {
      key_index[inner_keys[i]] = i;
    }
    // Note: Here, we leverage the block-diagonal approximation of the
    // interpolated covariances (i.e., independence approximation)
    for (uint i = 0; i < interp_states_.size(); i++) {
      const StateData& state = interp_states_[i];
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
      noise_cov += G_tau * InterpCondCovs[i] * G_tau.transpose();
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

/// traits
template <class POSE>
struct traits<WNOAInterpFactor<POSE>>
    : public Testable<WNOAInterpFactor<POSE>> {};

}  // namespace gtsam