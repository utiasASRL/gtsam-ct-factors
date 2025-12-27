/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file MultifrontalSolver.h
 * @brief
 * @author Frank Dellaert
 * @date   December 2025
 */

#pragma once

#include <gtsam/base/SymmetricBlockMatrix.h>
#include <gtsam/base/VerticalBlockMatrix.h>
#include <gtsam/inference/Ordering.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/VectorValues.h>
#include <gtsam/symbolic/SymbolicJunctionTree.h>

#include <map>
#include <vector>

namespace gtsam {

/**
 * Imperative, imperative-style multifrontal solver for Gaussian factor graphs.
 *
 * This class pre-allocates all necessary memory for the elimination tree and
 * provides efficient methods for loading new factors, eliminating the graph,
 * and solving for the update vector.
 */
class GTSAM_EXPORT MultifrontalSolver {
 public:
  struct Clique {
    using shared_ptr = std::shared_ptr<Clique>;

    Key key;                       ///< The clique's key in the junction tree
    std::weak_ptr<Clique> parent;  ///< Parent clique
    std::vector<shared_ptr> children;  ///< Child cliques

    // Local factors (A, b) stored in a VerticalBlockMatrix
    VerticalBlockMatrix Ab;

    // Elimination result (R, S, d) stored in a SymmetricBlockMatrix
    // SBM structure: [R S d; 0 L 0; 0 0 0] where L is the separator Hessian
    SymmetricBlockMatrix sbm;

    // Split-off conditional matrix [R S d]
    VerticalBlockMatrix R_Sd;

    // Indices of factors in the original graph that belong to this clique
    std::vector<size_t> factorIndices;

    KeyVector frontalKeys;    ///< Frontal keys for this clique
    KeyVector separatorKeys;  ///< Separator keys for this clique

    // Mapping from this clique's blocks (after elimination) to parent's blocks
    // Index 0 is the first separator block, last index is the RHS block.
    std::vector<size_t> parentIndices;

    Clique(Key k) : key(k) {}
  };

  using CliquePtr = Clique::shared_ptr;

 private:
  std::vector<CliquePtr> roots_;
  std::vector<CliquePtr> cliques_;           // All cliques
  std::vector<CliquePtr> postOrderCliques_;  // For elimination
  std::map<Key, size_t> dims_;               // Variable dimensions

 public:
  /**
   * Construct the solver from a factor graph and an ordering.
   * This builds the symbolic junction tree and pre-allocates all matrices.
   */
  MultifrontalSolver(const GaussianFactorGraph& graph,
                     const Ordering& ordering);

  /**
   * Load new numerical values from the factor graph.
   * This overwrites the values in the pre-allocated matrices.
   *
   * @param graph The factor graph with updated values (structure must match).
   */
  void load(const GaussianFactorGraph& graph);

  /**
   * Eliminate the graph using Cholesky factorization.
   * This operates in-place on the pre-allocated matrices.
   */
  void eliminate();

  /**
   * Solve for the update vector.
   *
   * @return The solution vector delta.
   */
  VectorValues solve() const;

  // Accessors for testing
  const std::vector<CliquePtr>& roots() const { return roots_; }
};

}  // namespace gtsam
