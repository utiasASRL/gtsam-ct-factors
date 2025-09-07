/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    SymmetricBlockMatrix.cpp
 * @brief   Access to matrices via blocks of pre-defined sizes.  Used in GaussianFactor and GaussianConditional.
 * @author  Richard Roberts
 * @date    Sep 18, 2010
 */

#include <gtsam/base/SymmetricBlockMatrix.h>
#include <gtsam/base/VerticalBlockMatrix.h>
#include <gtsam/base/cholesky.h>
#include <gtsam/base/timing.h>
#include <gtsam/base/ThreadsafeException.h>

namespace gtsam {

/* ************************************************************************* */
SymmetricBlockMatrix::SymmetricBlockMatrix() : blockStart_(0) {
  variableColOffsets_.push_back(0);
  assertInvariants();
}

/* ************************************************************************* */
SymmetricBlockMatrix SymmetricBlockMatrix::LikeActiveViewOf(
    const SymmetricBlockMatrix& other) {
  SymmetricBlockMatrix result;
  result.variableColOffsets_.resize(other.nBlocks() + 1);
  for (size_t i = 0; i < result.variableColOffsets_.size(); ++i)
    result.variableColOffsets_[i] = other.variableColOffsets_[other.blockStart_
        + i] - other.variableColOffsets_[other.blockStart_];
  result.matrix_.resize(other.cols(), other.cols());
  result.assertInvariants();
  return result;
}

/* ************************************************************************* */
SymmetricBlockMatrix SymmetricBlockMatrix::LikeActiveViewOf(
    const VerticalBlockMatrix& other) {
  SymmetricBlockMatrix result;
  result.variableColOffsets_.resize(other.nBlocks() + 1);
  for (size_t i = 0; i < result.variableColOffsets_.size(); ++i)
    result.variableColOffsets_[i] = other.variableColOffsets_[other.blockStart_
        + i] - other.variableColOffsets_[other.blockStart_];
  result.matrix_.resize(other.cols(), other.cols());
  result.assertInvariants();
  return result;
}

/* ************************************************************************* */
Matrix SymmetricBlockMatrix::block(DenseIndex I, DenseIndex J) const {
  if (I == J) {
    return diagonalBlock(I);
  } else if (I < J) {
    return aboveDiagonalBlock(I, J);
  } else {
    return aboveDiagonalBlock(J, I).transpose();
  }
}

/* ************************************************************************* */
void SymmetricBlockMatrix::negate() {
  full().triangularView<Eigen::Upper>() *= -1.0;
}

/* ************************************************************************* */
void SymmetricBlockMatrix::invertInPlace() {
  std::cout << "DEBUG: Starting invertInPlace()" << std::endl;
  std::cout << "DEBUG: Original matrix:\n" << this->full() << std::endl;
  const auto identity = Matrix::Identity(rows(), rows());
  full().triangularView<Eigen::Upper>() =
      selfadjointView().llt().solve(identity).triangularView<Eigen::Upper>();
  std::cout << "DEBUG: Final inverted matrix:\n" << this->full() << std::endl;
  std::cout << "DEBUG: invertInPlace() completed successfully" << std::endl;
}


// Solve Rkk * X = M for X, where Rkk is upper-triangular (block).
static inline Matrix leftDivideByUpper(const Matrix& Rkk_upper, const Matrix& M) {
  // Solve Rkk^T * Y^T = M^T, then X = Y^T (avoids forming explicit inverse)
  Matrix Mt  = M.transpose();
  Matrix Yt  = Rkk_upper.transpose().triangularView<Eigen::Lower>().solve(Mt);
  return Yt.transpose();
}

void SymmetricBlockMatrix::invertInPlaceSparse() {
  // --- 0) Materialize numeric Cholesky factor R (upper) into full() ---
  {
    // Treat current matrix as SPD and factor its upper triangle
    Eigen::LLT<Matrix> llt(this->full().selfadjointView<Eigen::Upper>());
    Matrix R = llt.matrixU();
    this->full().setZero();
    this->full().triangularView<Eigen::Upper>() = R.triangularView<Eigen::Upper>();
  }

  const size_t B = this->nBlocks();
  if (B == 0) return;

  // --- 1) Block sizes and scalar offsets ---
  std::vector<int> bsize(B);
  for (size_t i = 0; i < B; ++i) {
    const Matrix di = this->block(i, i);
    if (di.size() == 0) throw std::runtime_error("Takahashi: empty diagonal block");
    bsize[i] = static_cast<int>(di.rows());
  }
  std::vector<int> offset(B + 1, 0);
  for (size_t i = 0; i < B; ++i) offset[i + 1] = offset[i] + bsize[i];

  // Access a block of R (upper). Returns empty if i>k (below diagonal).
  auto Rblock = [&](size_t i, size_t k) -> Matrix {
    if (i > k) return Matrix();
    const int r0 = offset[i], c0 = offset[k];
    return this->full().block(r0, c0, bsize[i], bsize[k]);
  };

  // --- 2) Storage for selected inverse: Zcol[j][i] == Z_{i,j}, i >= j ---
  std::vector<std::unordered_map<size_t, Matrix>> Zcol(B);

  // Workspace: z[i] holds current column j’s block (b_i x b_j)
  std::vector<Matrix> z(B);

  // --- 3) Main Takahashi sweep: columns j = B-1 .. 0 ---
  for (int jj = static_cast<int>(B) - 1; jj >= 0; --jj) {
    const size_t j = static_cast<size_t>(jj);
    const int bj   = bsize[j];

    // Zero workspace and SCATTER any already-computed Z(:,j) (from later columns)
    for (size_t i = 0; i < B; ++i) z[i] = Matrix::Zero(bsize[i], bj);
    if (!Zcol[j].empty()) {
      for (const auto& pr : Zcol[j]) {
        const size_t i = pr.first;     // i >= j
        z[i] = pr.second;              // copy into workspace
      }
    }

    // Diagonal seed: D_j^{-1} = R_{jj}^{-T} * R_{jj}^{-1}
    const Matrix Rjj = Rblock(j, j);               // upper, (bj x bj)
    if (Rjj.size() == 0) throw std::runtime_error("Takahashi: missing Rjj");
    Matrix Ijj  = Matrix::Identity(bj, bj);
    Matrix W    = Rjj.triangularView<Eigen::Upper>().solve(Ijj);   // Rjj * W = I
    Matrix Dinv = W.transpose() * W;                                // Rjj^{-T} Rjj^{-1}
    z[j] += Dinv;

    // Strict-upper sweep: for k = j-1 .. 0
    for (int kk = jj - 1; kk >= 0; --kk) {
      const size_t k = static_cast<size_t>(kk);
      const int bk   = bsize[k];
      const Matrix Rkk = Rblock(k, k);             // (bk x bk), upper
      Matrix acc = Matrix::Zero(bk, bj);

      // sum over ell > k where R_{k,ell} exists
      for (size_t ell = k + 1; ell < B; ++ell) {
        Matrix Rkell = Rblock(k, ell);             // (bk x b_ell)
        if (!Rkell.size()) continue;
        // Uhat_{k,ell} = R_{kk}^{-1} * R_{k,ell}  (LEFT division by row-diagonal)
        Matrix Uhat_kell = leftDivideByUpper(Rkk, Rkell);
        if (!z[ell].size()) continue;             // (b_ell x b_j)
        acc.noalias() += Uhat_kell * z[ell];      // (bk x bj)
      }
      z[k] = -acc;
    }

    // Left-looking scatter into earlier columns k < j
    for (size_t k = 0; k < j; ++k) {
      Matrix Rkj = Rblock(k, j);                   // (b_k x b_j)
      if (!Rkj.size()) continue;
      const Matrix Rkk = Rblock(k, k);             // (b_k x b_k), upper
      // Uhat_{k,j} = R_{kk}^{-1} * R_{k,j},  Lhat_{j,k} = Uhat_{k,j}^T
      Matrix Uhat_kj = leftDivideByUpper(Rkk, Rkj);
      Matrix Lhat_jk = Uhat_kj.transpose();        // (b_j x b_k)

      // Update all rows i >= k (lower triangle)
      for (size_t i = k; i < B; ++i) {
        if (!z[i].size()) continue;                // (b_i x b_j)
        Matrix delta = z[i] * Lhat_jk;             // (b_i x b_k)
        auto it = Zcol[k].find(i);
        if (it == Zcol[k].end()) Zcol[k].emplace(i, -delta);
        else                      it->second.noalias() -= delta;
      }
    }

    // Gather only this column j (rows i >= j). Do NOT touch i<j or i>j in other columns.
    for (size_t i = j; i < B; ++i) {
      if (!z[i].size()) continue;
      Zcol[j][i] = z[i];                           // overwrite / set Z_{i,j}
    }
  }

  // --- 4) Write Z back into block-sparse storage (lower triangle) ---
  this->setZero();

  for (size_t j = 0; j < B; ++j) {
    for (const auto& pr : Zcol[j]) {
      const size_t i = pr.first;                   // i >= j
      const Matrix& Zij = pr.second;               // (b_i x b_j)
      if (i == j) this->setDiagonalBlock(i, Zij);
      else         this->setOffDiagonalBlock(i, j, Zij);
    }
  }
}

/* ************************************************************************* */
void SymmetricBlockMatrix::choleskyPartial(DenseIndex nFrontals) {
  gttic(VerticalBlockMatrix_choleskyPartial);
  DenseIndex topleft = variableColOffsets_[blockStart_];
  if (!gtsam::choleskyPartial(matrix_, offset(nFrontals) - topleft, topleft)) {
    throw CholeskyFailed();
  }
}

/* ************************************************************************* */
VerticalBlockMatrix SymmetricBlockMatrix::split(DenseIndex nFrontals) {
  gttic(VerticalBlockMatrix_split);

  // Construct a VerticalBlockMatrix that contains [R Sd]
  const size_t n1 = offset(nFrontals);
  VerticalBlockMatrix RSd = VerticalBlockMatrix::LikeActiveViewOf(*this, n1);

  // Copy into it.
  RSd.full() = matrix_.topRows(n1);
  RSd.full().triangularView<Eigen::StrictlyLower>().setZero();

  // Take lower-right block of Ab_ to get the remaining factor
  blockStart() = nFrontals;

  return RSd;
}

/* ************************************************************************* */

} //\ namespace gtsam

