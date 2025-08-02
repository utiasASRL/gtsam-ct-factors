#include <cmath>

#include "Eigen/Core"
#include "lgmath.hpp"
#include "steam.hpp"

using namespace std;

namespace steam {

/** \brief Evaluator for 2D bearing-range measurements. Assumes that the vehicle
 * and the landmark are close to the z=0 plane */
class BRError2DEvaluator : public Evaluable<Eigen::Matrix<double, 2, 1>> {
 public:
  using Ptr = std::shared_ptr<BRError2DEvaluator>;
  using ConstPtr = std::shared_ptr<const BRError2DEvaluator>;

  using InType = Eigen::Vector4d;
  using OutType = Eigen::Matrix<double, 2, 1>;

  static Ptr MakeShared(const Evaluable<InType>::ConstPtr& point,
                        const Eigen::Vector2d& br_meas) {
    return std::make_shared<BRError2DEvaluator>(point, br_meas);
  }

  BRError2DEvaluator(const Evaluable<InType>::ConstPtr& point,
                     const Eigen::Vector2d& br_meas)
      : point_(point), br_meas_(br_meas) {}

  bool active() const override { return point_->active(); }

  void getRelatedVarKeys(KeySet& keys) const override {
    return point_->getRelatedVarKeys(keys);
  }

  auto value() const -> OutType override {
    double x = point_->value()[0];
    double y = point_->value()[1];
    const auto br_val =
        Eigen::Vector2d(atan2(y, x), sqrt(pow(x, 2) + pow(y, 2)));
    return br_val - br_meas_;
  }
  auto forward() const -> Node<OutType>::Ptr {
    // Call forward of child point node
    const auto child = point_->forward();
    // get value
    double x = child->value()[0];
    double y = child->value()[1];
    OutType value =
        Eigen::Vector2d(atan2(y, x), sqrt(pow(x, 2) + pow(y, 2))) - br_meas_;
    // create node
    const auto node = Node<OutType>::MakeShared(value);
    node->addChild(child);
    return node;
  }

  void backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node,
                Jacobians& jacs) const override {
    // get child node pointer
    const auto child = static_pointer_cast<Node<InType>>(node->at(0));
    // build jacobian
    if (point_->active()) {
      // Retrieve range
      double x = child->value()[0];
      double y = child->value()[1];
      double r = sqrt(pow(x, 2) + pow(y, 2));
      // compute jacobian
      auto Jac = Eigen::Matrix<double, 2, 4>::Zero().eval();
      if (r > 1e-8) {
        Jac(0, 0) = -y / pow(r, 2);
        Jac(0, 1) = x / pow(r, 2);
        Jac(1, 0) = x / r;
        Jac(1, 1) = y / r;
      }
      point_->backward(lhs * Jac, child, jacs);
    }
  }

 private:
  /** \brief point evaluable */
  const Evaluable<InType>::ConstPtr point_;
  const Eigen::Vector2d br_meas_;
  OutType br_val_ = Eigen::Vector2d(0.0, 0.0);
};

}  // namespace steam