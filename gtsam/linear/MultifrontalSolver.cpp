/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file MultifrontalSolver.cpp
 * @brief
 * @author Frank Dellaert
 * @date   December 2025
 */

#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/GaussianFactor.h>
#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/linear/MultifrontalSolver.h>
#include <gtsam/symbolic/SymbolicEliminationTree.h>
#include <gtsam/symbolic/SymbolicFactorGraph.h>

#include <algorithm>
#include <functional>
#include <set>
#include <stdexcept>

namespace gtsam {

namespace {

// Helper class to track original factor indices
class IndexedSymbolicFactor : public SymbolicFactor {
 public:
  size_t index_;
  IndexedSymbolicFactor(const GaussianFactor& factor, size_t index)
      : SymbolicFactor(factor), index_(index) {}
};

}  // namespace

/* ************************************************************************* */
MultifrontalSolver::MultifrontalSolver(const GaussianFactorGraph& graph,
                                       const Ordering& ordering) {
  // 0. Pre-compute variable dimensions
  for (const auto& factor : graph) {
    if (!factor) continue;
    if (auto jf = std::dynamic_pointer_cast<JacobianFactor>(factor)) {
      for (auto it = jf->begin(); it != jf->end(); ++it) {
        dims_[*it] = jf->getDim(it);
      }
    } else if (auto hf = std::dynamic_pointer_cast<HessianFactor>(factor)) {
      for (auto it = hf->begin(); it != hf->end(); ++it) {
        dims_[*it] = hf->getDim(it);
      }
    }
  }

  // 1. Convert to SymbolicFactorGraph to build the elimination tree
  SymbolicFactorGraph symbolicGraph;
  symbolicGraph.reserve(graph.size());
  for (size_t i = 0; i < graph.size(); ++i) {
    if (graph[i]) {
      symbolicGraph.push_back(
          std::make_shared<IndexedSymbolicFactor>(*graph[i], i));
    } else {
      symbolicGraph.push_back(SymbolicFactor::shared_ptr());
    }
  }

  // 2. Build SymbolicEliminationTree and then SymbolicJunctionTree
  SymbolicEliminationTree etree(symbolicGraph, ordering);
  SymbolicJunctionTree jtree(etree);

  // 3. Recursive function to build Clique hierarchy
  std::function<CliquePtr(const SymbolicJunctionTree::sharedNode&,
                          std::weak_ptr<Clique>)>
      buildRecursive = [&](const SymbolicJunctionTree::sharedNode& cluster,
                           std::weak_ptr<Clique> parent) -> CliquePtr {
    if (!cluster) return nullptr;

    // Create Clique
    Key representativeKey = cluster->orderedFrontalKeys.empty()
                                ? 0
                                : cluster->orderedFrontalKeys.front();
    auto clique = std::make_shared<Clique>(representativeKey);
    clique->parent = parent;
    clique->frontalKeys = cluster->orderedFrontalKeys;
    cliques_.push_back(clique);

    // Identify factor indices
    for (const auto& factor : cluster->factors) {
      if (auto indexed =
              std::dynamic_pointer_cast<IndexedSymbolicFactor>(factor)) {
        clique->factorIndices.push_back(indexed->index_);
      }
    }

    // Process children
    for (const auto& childCluster : cluster->children) {
      auto childClique = buildRecursive(childCluster, clique);
      clique->children.push_back(childClique);
    }

    // Determine separator keys for THIS clique from local factors only.
    std::set<Key> allKeys;
    for (const auto& factor : cluster->factors) {
      for (Key k : factor->keys()) {
        allKeys.insert(k);
      }
    }

    // Remove frontal keys
    for (Key k : clique->frontalKeys) {
      allKeys.erase(k);
    }

    // Store separator keys
    clique->separatorKeys.assign(allKeys.begin(), allKeys.end());

    // Calculate block dimensions
    std::vector<size_t> blockDims;
    for (Key k : clique->frontalKeys) blockDims.push_back(dims_.at(k));
    for (Key k : clique->separatorKeys) blockDims.push_back(dims_.at(k));
    blockDims.push_back(1);  // RHS

    // Initialize matrices
    clique->sbm = SymmetricBlockMatrix(blockDims);
    clique->sbm.setZero();

    size_t vbmRows = 0;
    for (size_t idx : clique->factorIndices) {
      if (auto jf = std::dynamic_pointer_cast<JacobianFactor>(graph[idx])) {
        vbmRows += jf->rows();
      }
    }
    clique->Ab = VerticalBlockMatrix(blockDims, vbmRows);
    clique->Ab.matrix().setZero();

    // Initial load
    size_t rowOffset = 0;
    for (size_t idx : clique->factorIndices) {
      auto jf = std::dynamic_pointer_cast<JacobianFactor>(graph[idx]);
      if (!jf) continue;

      for (auto it = jf->begin(); it != jf->end(); ++it) {
        Key k = *it;
        auto fIt = std::find(clique->frontalKeys.begin(),
                             clique->frontalKeys.end(), k);
        size_t blockIdx = 0;
        if (fIt != clique->frontalKeys.end()) {
          blockIdx = std::distance(clique->frontalKeys.begin(), fIt);
        } else {
          auto sIt = std::find(clique->separatorKeys.begin(),
                               clique->separatorKeys.end(), k);
          if (sIt != clique->separatorKeys.end()) {
            blockIdx = clique->frontalKeys.size() +
                       std::distance(clique->separatorKeys.begin(), sIt);
          } else
            continue;
        }
        clique->Ab(blockIdx).middleRows(rowOffset, jf->rows()) = jf->getA(it);
      }
      size_t rhsBlockIdx = blockDims.size() - 1;
      clique->Ab(rhsBlockIdx).middleRows(rowOffset, jf->rows()) = jf->getb();
      rowOffset += jf->rows();
    }

    // Pre-compute parent mapping
    if (auto p = parent.lock()) {
      for (Key k : clique->separatorKeys) {
        auto fIt = std::find(p->frontalKeys.begin(), p->frontalKeys.end(), k);
        if (fIt != p->frontalKeys.end()) {
          clique->parentIndices.push_back(
              std::distance(p->frontalKeys.begin(), fIt));
          continue;
        }
        auto sIt =
            std::find(p->separatorKeys.begin(), p->separatorKeys.end(), k);
        if (sIt != p->separatorKeys.end()) {
          clique->parentIndices.push_back(
              p->frontalKeys.size() +
              std::distance(p->separatorKeys.begin(), sIt));
          continue;
        }
        throw std::runtime_error(
            "MultifrontalSolver: separator key not found in parent clique");
      }
      // Last block is RHS
      clique->parentIndices.push_back(p->frontalKeys.size() +
                                      p->separatorKeys.size());
    }

    postOrderCliques_.push_back(clique);
    return clique;
  };

  // 4. Start traversal from roots
  for (const auto& rootCluster : jtree.roots()) {
    if (rootCluster) {
      roots_.push_back(buildRecursive(rootCluster, std::weak_ptr<Clique>()));
    }
  }
}

/* ************************************************************************* */
void MultifrontalSolver::load(const GaussianFactorGraph& graph) {
  for (auto& clique : cliques_) {
    // 1. Reset SymmetricBlockMatrix
    clique->sbm.blockStart() = 0;
    clique->sbm.setZero();

    // 2. Reset VerticalBlockMatrix
    clique->Ab.matrix().setZero();

    // 3. Fill VerticalBlockMatrix from graph
    size_t rowOffset = 0;
    for (size_t idx : clique->factorIndices) {
      auto jf = std::dynamic_pointer_cast<JacobianFactor>(graph[idx]);
      if (!jf) continue;

      for (auto it = jf->begin(); it != jf->end(); ++it) {
        Key k = *it;
        auto fIt = std::find(clique->frontalKeys.begin(),
                             clique->frontalKeys.end(), k);
        size_t blockIdx = 0;
        if (fIt != clique->frontalKeys.end()) {
          blockIdx = std::distance(clique->frontalKeys.begin(), fIt);
        } else {
          auto sIt = std::find(clique->separatorKeys.begin(),
                               clique->separatorKeys.end(), k);
          if (sIt != clique->separatorKeys.end()) {
            blockIdx = clique->frontalKeys.size() +
                       std::distance(clique->separatorKeys.begin(), sIt);
          } else
            continue;
        }
        clique->Ab(blockIdx).middleRows(rowOffset, jf->rows()) = jf->getA(it);
      }
      size_t rhsBlockIdx = clique->Ab.nBlocks() - 1;
      clique->Ab(rhsBlockIdx).middleRows(rowOffset, jf->rows()) = jf->getb();
      rowOffset += jf->rows();
    }
  }
}

/* ************************************************************************* */
void MultifrontalSolver::eliminate() {
  for (auto& clique : postOrderCliques_) {
    // 1. Add local Hessian: SBM += Ab' * Ab
    clique->sbm.selfadjointView().rankUpdate(clique->Ab.matrix().transpose());

    // 2. Eliminate frontal variables
    clique->sbm.choleskyPartial(clique->frontalKeys.size());

    // 3. Always split to store R_Sd for back-substitution
    clique->R_Sd = clique->sbm.split(clique->frontalKeys.size());

    // 4. Propagate separator to parent
    if (auto parent = clique->parent.lock()) {
      size_t nSeparatorBlocks = clique->separatorKeys.size();
      for (size_t i = 0; i <= nSeparatorBlocks; ++i) {
        size_t p_i = clique->parentIndices[i];
        parent->sbm.updateDiagonalBlock(p_i, clique->sbm.diagonalBlock(i));
        for (size_t j = i + 1; j <= nSeparatorBlocks; ++j) {
          size_t p_j = clique->parentIndices[j];
          parent->sbm.updateOffDiagonalBlock(
              p_i, p_j, clique->sbm.aboveDiagonalBlock(i, j));
        }
      }
    }
  }
}

/* ************************************************************************* */
VectorValues MultifrontalSolver::solve() const {
  VectorValues x;
  for (const auto& clique : cliques_) {
    size_t nFrontals = clique->frontalKeys.size();
    size_t nSeparators = clique->separatorKeys.size();

    size_t nFrontalDim = 0;
    for (Key k : clique->frontalKeys) nFrontalDim += dims_.at(k);

    const Matrix& RSd = clique->R_Sd.full();
    Vector rhs = RSd.col(RSd.cols() - 1);

    for (size_t i = 0; i < nSeparators; ++i) {
      Key k = clique->separatorKeys[i];
      rhs.noalias() -= clique->R_Sd(nFrontals + i) * x.at(k);
    }

    Vector xF = RSd.leftCols(nFrontalDim)
                    .template triangularView<Eigen::Upper>()
                    .solve(rhs);

    size_t offset = 0;
    for (Key k : clique->frontalKeys) {
      size_t d = dims_.at(k);
      x.insert(k, xF.segment(offset, d));
      offset += d;
    }
  }
  return x;
}

}  // namespace gtsam
