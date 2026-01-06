/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    timeMultifrontalSolver.cpp
 * @brief   Compare MultifrontalSolver against standard elimination
 * @author  Frank Dellaert
 * @date    December 2025
 */

#include <gtsam/linear/GaussianBayesTree.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/MultifrontalSolver.h>
#include <tests/smallExample.h>

#include <chrono>
#include <iostream>

#include "timeSFMBAL.h"

using namespace std;
using namespace gtsam;
using namespace example;

namespace {
// Threshold chosen empirically for these timing experiments: merging small
// frontal cliques tends to improve performance without materially affecting
// numerical behavior for the tested problems.
constexpr size_t kMergeDimCap = 16;
}  // namespace

/// Run standard GTSAM multifrontal elimination and optimization.
static void runStandardSolver(const GaussianFactorGraph& smoother,
                              const Ordering& ordering, size_t iterations) {
  for (size_t i = 0; i < iterations; ++i) {
    GaussianBayesTree bayesTree = *smoother.eliminateMultifrontal(ordering);
    VectorValues solution = bayesTree.optimize();
    (void)solution;
  }
}

/// Run new MultifrontalSolver elimination and optimization.
static void runMultifrontalSolver(MultifrontalSolver& solver,
                                  const GaussianFactorGraph& graph,
                                  size_t iterations) {
  for (size_t i = 0; i < iterations; ++i) {
    solver.eliminateInPlace(graph);
    const VectorValues& solution = solver.updateSolution();
    (void)solution;
  }
}

namespace {
const std::string bal135 = findExampleDataFile("dubrovnik-135-90642-pre.txt");
const string bal16 = findExampleDataFile("dubrovnik-16-22106-pre");
const string bal88 = findExampleDataFile("dubrovnik-88-64298-pre");
}  // namespace

void runBAL135Benchmark() {
  const size_t iterations = 1;
  cout << "\nSingle MFS test: " << bal135 << " (iterations=" << iterations
       << ")" << std::endl;

  const SfmData db = SfmData::FromBalFile(bal135);
  const NonlinearFactorGraph graph = buildGeneralSfmGraph(db, 0.1);
  const Values initial = buildGeneralSfmInitial(db);
  const GaussianFactorGraph linear = *graph.linearize(initial);
  const Ordering ordering = createSchurOrdering(db, false);

  MultifrontalSolver::Parameters params;
  params.mergeDimCap = kMergeDimCap;
  MultifrontalSolver solver(linear, ordering, params);
  auto start = std::chrono::high_resolution_clock::now();
  runMultifrontalSolver(solver, linear, iterations);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> t_imperative = end - start;
  cout << "  MultifrontalSolver: " << t_imperative.count() << " s" << std::endl;
  tictoc_print();
}

void runBALBenchmark() {
  const size_t bal_iterations = 2;
  for (const auto& filename : {bal16, bal88, bal135}) {
    cout << "\nProcessing BAL file: " << filename << std::endl;
    const SfmData db = SfmData::FromBalFile(filename);
    const NonlinearFactorGraph graph = buildGeneralSfmGraph(db, 0.1);
    const Values initial = buildGeneralSfmInitial(db);
    const GaussianFactorGraph linear = *graph.linearize(initial);

    const Ordering ordering = createSchurOrdering(db, false);
    cout << "\nBAL Benchmark (" << filename << ", iterations=" << bal_iterations
         << "):" << std::endl;

    MultifrontalSolver::Parameters params;
    params.mergeDimCap = kMergeDimCap;
    params.reportStream = &std::cout;
    MultifrontalSolver solver(linear, ordering, params);
    solver.eliminateInPlace(linear);  // Warm up cache.
    auto start = std::chrono::high_resolution_clock::now();
    runMultifrontalSolver(solver, linear, bal_iterations);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> t_imperative = end - start;
    cout << "  MultifrontalSolver: " << t_imperative.count() << " s"
         << std::endl;

    start = std::chrono::high_resolution_clock::now();
    runStandardSolver(linear, ordering, bal_iterations);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> t_standard = end - start;
    cout << "  Standard GTSAM:     " << t_standard.count() << " s" << std::endl;

    cout << "  Speedup:            "
         << t_standard.count() / t_imperative.count() << "x" << std::endl;
  }
}

void runChainBenchmark() {
  const std::vector<size_t> T_values = {10, 50, 100, 500, 1000, 5000};
  const size_t iterations = 500;

  for (size_t T : T_values) {
    cout << "\nBenchmark (T=" << T << ", iterations=" << iterations
         << "):" << std::endl;
    GaussianFactorGraph smoother = createSmoother(T);
    const Ordering ordering = Ordering::Metis(smoother);

    auto start = std::chrono::high_resolution_clock::now();
    MultifrontalSolver::Parameters params;
    params.mergeDimCap = kMergeDimCap;
    params.reportStream = &std::cout;
    MultifrontalSolver solver(smoother, ordering, params);
    runMultifrontalSolver(solver, smoother, iterations);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> t_imperative = end - start;
    cout << "\nTiming results:\n";
    cout << "  MultifrontalSolver: " << t_imperative.count() << " s"
         << std::endl;

    start = std::chrono::high_resolution_clock::now();
    runStandardSolver(smoother, ordering, iterations);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> t_standard = end - start;
    cout << "  Standard GTSAM:     " << t_standard.count() << " s" << std::endl;

    cout << "  Speedup:            "
         << t_standard.count() / t_imperative.count() << "x" << std::endl;
  }
}

void tuneMergingBAL() {
  const size_t iterations = 2;
  const std::vector<std::string> balFiles = {bal16, bal88, bal135};
  cout << "\nTune leaf merging (BAL, iterations=" << iterations << ")"
       << std::endl;

  const std::vector<size_t> caps = {0, 64, 128, 256, 512, 1024, 2048};
  std::vector<std::vector<double>> results(
      caps.size(), std::vector<double>(balFiles.size(), 0.0));

  for (size_t fileIndex = 0; fileIndex < balFiles.size(); ++fileIndex) {
    const std::string& filename = balFiles[fileIndex];
    cout << "\n  BAL file: " << filename << std::endl;
    const SfmData db = SfmData::FromBalFile(filename);
    const NonlinearFactorGraph graph = buildGeneralSfmGraph(db, 0.1);
    const Values initial = buildGeneralSfmInitial(db);
    const GaussianFactorGraph linear = *graph.linearize(initial);
    const Ordering ordering = createSchurOrdering(db, false);

    MultifrontalSolver::Parameters params;
    params.mergeDimCap = 0;
    params.reportStream = &std::cout;
    for (size_t capIndex = 0; capIndex < caps.size(); ++capIndex) {
      const size_t cap = caps[capIndex];
      params.leafMergeDimCap = cap;

      MultifrontalSolver solver(linear, ordering, params);
      solver.eliminateInPlace(linear);  // Warm up cache.
      auto start = std::chrono::high_resolution_clock::now();
      runMultifrontalSolver(solver, linear, iterations);
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> t_imperative = end - start;
      results[capIndex][fileIndex] = t_imperative.count();
      cout << "  leafMergeDimCap=" << cap << " -> " << t_imperative.count()
           << " s\n"
           << std::endl;
    }
  }

  cout << "\n| LeafMergeDimCap | BAL16 | BAL88 | BAL135 |\n";
  cout << "| --- | --- | --- | --- |\n";
  for (size_t capIndex = 0; capIndex < caps.size(); ++capIndex) {
    cout << "| " << caps[capIndex];
    for (size_t fileIndex = 0; fileIndex < balFiles.size(); ++fileIndex) {
      cout << " | " << results[capIndex][fileIndex];
    }
    cout << " |\n";
  }
}

void tuneMergeChain() {
  const size_t iterations = 100;
  const size_t T = 5000;
  cout << "\nTune mergeDimCap (chain T=" << T << ", iterations=" << iterations
       << ")" << std::endl;

  GaussianFactorGraph smoother = createSmoother(T);
  const Ordering ordering = Ordering::Metis(smoother);

  const std::vector<size_t> caps = {0, 16, 32, 64, 128, 256, 512};
  std::vector<std::pair<size_t, double>> results;
  MultifrontalSolver::Parameters params;
  params.leafMergeDimCap = 0;
  params.reportStream = &std::cout;
  for (size_t cap : caps) {
    params.mergeDimCap = cap;
    MultifrontalSolver solver(smoother, ordering, params);

    auto start = std::chrono::high_resolution_clock::now();
    runMultifrontalSolver(solver, smoother, iterations);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> t_imperative = end - start;
    results.emplace_back(cap, t_imperative.count());
    cout << "  mergeDimCap=" << cap << " -> " << t_imperative.count() << " s\n"
         << std::endl;
  }

  cout << "\n| Phase | Cap | Seconds |\n";
  cout << "| --- | --- | --- |\n";
  for (const auto& result : results) {
    cout << "| mergeDimCap | " << result.first << " | " << result.second
         << " |\n";
  }
}

int main() {
  cout << "Merging dim cap " << kMergeDimCap << std::endl;

  runBAL135Benchmark();
  runBALBenchmark();
  runChainBenchmark();
  // tuneMergingBAL();
  // tuneMergeChain();
  return 0;
}
