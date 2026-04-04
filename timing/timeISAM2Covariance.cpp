/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file    timeISAM2Covariance.cpp
 * @brief   Incremental timing study for ISAM2 joint covariance queries.
 * @date    April 2026
 * @author  Codex 5.4, prompted by Frank Dellaert
 */

#include <gtsam/geometry/Pose2.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/dataset.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace gtsam;
using namespace std;

namespace {

/// Structural statistics for one localized query support.
struct SupportStats {
  size_t supportCliques = 0;
  size_t compressedCliques = 0;
  size_t maxFrontalDim = 0;
  size_t maxSeparatorDim = 0;
};

/// One incremental timing sample.
struct StepResult {
  size_t stepIndex = 0;
  Key currentKey = 0;
  size_t poseCount = 0;
  size_t factorCount = 0;
  double updateMs = 0.0;
  double queryMeanMs = 0.0;
  double queryMedianMs = 0.0;
  double queryStddevMs = 0.0;
  size_t supportCliques = 0;
  size_t compressedCliques = 0;
  size_t reducedStateDim = 0;
  size_t maxFrontalDim = 0;
  size_t maxSeparatorDim = 0;
  size_t numCachedSeparatorMarginals = 0;
};

/// Return all Pose2 keys in sorted order.
KeyVector poseKeys(const Values& values) {
  KeyVector keys;
  for (const auto& keyValue : values.extract<Pose2>()) {
    keys.push_back(keyValue.first);
  }
  sort(keys.begin(), keys.end());
  return keys;
}

/// Sum the dimensions of an ordered key range.
template <class RANGE>
size_t totalDimension(const RANGE& keys, const Values& values) {
  size_t dim = 0;
  for (Key key : keys) {
    dim += values.at(key).dim();
  }
  return dim;
}

/// Compute the lowest common ancestor of two Bayes-tree cliques.
template <class CLIQUE>
shared_ptr<CLIQUE> findLowestCommonAncestor(const shared_ptr<CLIQUE>& lhs,
                                            const shared_ptr<CLIQUE>& rhs) {
  unordered_set<shared_ptr<CLIQUE>> ancestors;
  for (auto current = lhs; current; current = current->parent()) {
    ancestors.insert(current);
  }
  for (auto current = rhs; current; current = current->parent()) {
    if (ancestors.count(current)) {
      return current;
    }
  }
  return nullptr;
}

/// Estimate support size and clique-width statistics for a query.
template <class BAYESTREE>
SupportStats analyzeSupport(const BAYESTREE& bayesTree,
                            const KeyVector& queryKeys,
                            const Values& values) {
  using Clique = typename BAYESTREE::Clique;
  vector<shared_ptr<Clique>> queryCliques;
  unordered_set<shared_ptr<Clique>> seen;
  for (Key key : queryKeys) {
    auto clique = bayesTree.clique(key);
    if (seen.insert(clique).second) {
      queryCliques.push_back(clique);
    }
  }
  if (queryCliques.empty()) {
    return {};
  }

  auto root = queryCliques.front();
  for (size_t i = 1; i < queryCliques.size(); ++i) {
    root = findLowestCommonAncestor(root, queryCliques[i]);
  }

  unordered_set<shared_ptr<Clique>> support;
  support.insert(root);
  for (const auto& clique : queryCliques) {
    for (auto current = clique; current && current != root;
         current = current->parent()) {
      support.insert(current);
    }
  }

  unordered_map<shared_ptr<Clique>, size_t> supportChildren;
  for (const auto& clique : support) {
    supportChildren[clique] = 0;
  }
  for (const auto& clique : support) {
    if (clique == root) {
      continue;
    }
    auto parent = clique->parent();
    if (parent && support.count(parent)) {
      ++supportChildren[parent];
    }
  }

  unordered_set<shared_ptr<Clique>> querySet(queryCliques.begin(),
                                             queryCliques.end());
  size_t compressed = 1;
  for (const auto& clique : support) {
    if (clique == root) {
      continue;
    }
    if (querySet.count(clique) || supportChildren[clique] > 1) {
      ++compressed;
    }
  }

  size_t maxFrontalDim = 0;
  size_t maxSeparatorDim = 0;
  for (const auto& clique : support) {
    const auto conditional = clique->conditional();
    maxFrontalDim =
        max(maxFrontalDim, totalDimension(conditional->frontals(), values));
    maxSeparatorDim =
        max(maxSeparatorDim, totalDimension(conditional->parents(), values));
  }

  return {support.size(), compressed, maxFrontalDim, maxSeparatorDim};
}

/// Compute the median of a small sample vector.
double median(vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  sort(values.begin(), values.end());
  const size_t mid = values.size() / 2;
  if (values.size() % 2 == 0) {
    return 0.5 * (values[mid - 1] + values[mid]);
  }
  return values[mid];
}

/// Compute the arithmetic mean of a sample vector.
double mean(const vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  const double sum = accumulate(values.begin(), values.end(), 0.0);
  return sum / static_cast<double>(values.size());
}

/// Compute the population standard deviation of a sample vector.
double stddev(const vector<double>& values) {
  if (values.size() < 2) {
    return 0.0;
  }
  const double sampleMean = mean(values);
  double sumSquares = 0.0;
  for (double value : values) {
    const double delta = value - sampleMean;
    sumSquares += delta * delta;
  }
  return sqrt(sumSquares / static_cast<double>(values.size()));
}

/// Read a string argument from argv or return a default value.
string argumentOrDefault(char** begin, char** end, const string& flag,
                         const string& defaultValue) {
  for (auto it = begin; it != end; ++it) {
    if (string(*it) == flag && it + 1 != end) {
      return *(it + 1);
    }
  }
  return defaultValue;
}

/// Read an unsigned integer argument from argv or return a default value.
size_t sizeTArgumentOrDefault(char** begin, char** end, const string& flag,
                              size_t defaultValue) {
  return static_cast<size_t>(
      stoull(argumentOrDefault(begin, end, flag, to_string(defaultValue))));
}

/// Write incremental timing results to CSV.
void writeCsv(const filesystem::path& path, const vector<StepResult>& results) {
  ofstream os(path);
  os << "step_index,current_key,pose_count,factor_count,update_ms,query_mean_ms,"
        "query_median_ms,query_stddev_ms,support_cliques,compressed_cliques,"
        "reduced_state_dim,max_frontal_dim,max_separator_dim,"
        "num_cached_separator_marginals\n";
  os << fixed << setprecision(6);
  for (const auto& result : results) {
    os << result.stepIndex << ',' << result.currentKey << ','
       << result.poseCount << ',' << result.factorCount << ','
       << result.updateMs << ',' << result.queryMeanMs << ','
       << result.queryMedianMs << ',' << result.queryStddevMs << ','
       << result.supportCliques << ',' << result.compressedCliques << ','
       << result.reducedStateDim << ',' << result.maxFrontalDim << ','
       << result.maxSeparatorDim << ','
       << result.numCachedSeparatorMarginals << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  const string datasetName =
      argumentOrDefault(argv, argv + argc, "--dataset", "w20000.txt");
  const filesystem::path outputPath = argumentOrDefault(
      argv, argv + argc, "--output",
      (filesystem::path("timing") / "results" / "bayes_tree_covariance" /
       "isam2_w20000.csv")
          .string());
  const size_t queryRepeats =
      sizeTArgumentOrDefault(argv, argv + argc, "--query-repeats", 5);
  const size_t maxSteps =
      sizeTArgumentOrDefault(argv, argv + argc, "--max-steps", 0);

  const auto [graphPtr, initialPtr] =
      load2D(findExampleDataFile(datasetName));
  const KeyVector allPoseKeys = poseKeys(*initialPtr);
  if (allPoseKeys.empty()) {
    throw runtime_error("Dataset contains no Pose2 variables.");
  }

  unordered_map<Key, size_t> stepByKey;
  for (size_t index = 0; index < allPoseKeys.size(); ++index) {
    stepByKey[allPoseKeys[index]] = index;
  }

  ISAM2Params parameters;
  parameters.optimizationParams = ISAM2GaussNewtonParams();
  parameters.relinearizeSkip = 1;
  parameters.relinearizeThreshold = 0.01;
  ISAM2 isam2(parameters);

  const Key firstKey = allPoseKeys.front();
  const auto priorNoise = noiseModel::Diagonal::Sigmas(
      (Vector(3) << 1e-6, 1e-6, 1e-6).finished());

  vector<StepResult> results;
  results.reserve(allPoseKeys.size());
  size_t factorCount = 0;
  size_t nextMeasurement = 0;

  const size_t stepLimit =
      maxSteps == 0 ? allPoseKeys.size() : min(maxSteps, allPoseKeys.size());
  for (size_t stepIndex = 0; stepIndex < stepLimit; ++stepIndex) {
    const Key currentKey = allPoseKeys[stepIndex];
    NonlinearFactorGraph newFactors;
    Values newTheta;

    if (stepIndex == 0) {
      newTheta.insert(currentKey, initialPtr->at<Pose2>(currentKey));
      newFactors.addPrior(currentKey, initialPtr->at<Pose2>(currentKey),
                          priorNoise);
    }

    optional<Pose2> predictedPose;
    while (nextMeasurement < graphPtr->size()) {
      auto between =
          dynamic_pointer_cast<BetweenFactor<Pose2>>((*graphPtr)[nextMeasurement]);
      if (!between) {
        ++nextMeasurement;
        continue;
      }

      size_t newestStep = 0;
      for (Key key : between->keys()) {
        newestStep = max(newestStep, stepByKey.at(key));
      }
      if (newestStep > stepIndex) {
        break;
      }

      newFactors.push_back(between);
      if (stepIndex > 0) {
        const Key previousKey = allPoseKeys[stepIndex - 1];
        if (between->key1() == previousKey && between->key2() == currentKey) {
          predictedPose =
              isam2.calculateEstimate<Pose2>(previousKey) * between->measured();
        } else if (between->key1() == currentKey &&
                   between->key2() == previousKey) {
          predictedPose = isam2.calculateEstimate<Pose2>(previousKey) *
                          between->measured().inverse();
        }
      }
      ++nextMeasurement;
    }

    if (stepIndex > 0 && !isam2.valueExists(currentKey)) {
      if (predictedPose) {
        newTheta.insert(currentKey, *predictedPose);
      } else {
        newTheta.insert(currentKey, initialPtr->at<Pose2>(currentKey));
      }
    }

    const auto updateStart = chrono::steady_clock::now();
    try {
      isam2.update(newFactors, newTheta);
    } catch (const std::exception& error) {
      cerr << "ISAM2 update failed at step " << stepIndex << " key "
           << currentKey << ": " << error.what() << endl;
      throw;
    }
    const auto updateEnd = chrono::steady_clock::now();
    factorCount += newFactors.size();

    const KeyVector queryKeys{firstKey, currentKey};
    try {
      (void)isam2.jointMarginalCovariance(queryKeys);
    } catch (const std::exception& error) {
      cerr << "Warmup covariance query failed at step " << stepIndex << " key "
           << currentKey << ": " << error.what() << endl;
      throw;
    }

    vector<double> queryTimes;
    queryTimes.reserve(queryRepeats);
    for (size_t repeatIndex = 0; repeatIndex < queryRepeats; ++repeatIndex) {
      const auto queryStart = chrono::steady_clock::now();
      JointMarginal covariance;
      try {
        covariance = isam2.jointMarginalCovariance(queryKeys);
      } catch (const std::exception& error) {
        cerr << "Covariance query failed at step " << stepIndex << " key "
             << currentKey << " repeat " << repeatIndex << ": "
             << error.what() << endl;
        throw;
      }
      const auto queryEnd = chrono::steady_clock::now();
      volatile double checksum = covariance.fullMatrix().sum();
      (void)checksum;
      queryTimes.push_back(
          chrono::duration<double, milli>(queryEnd - queryStart).count());
    }

    const Values& linearizationPoint = isam2.getLinearizationPoint();
    const SupportStats supportStats =
        analyzeSupport(isam2, queryKeys, linearizationPoint);
    const GaussianFactorGraph reducedGraph =
        *isam2.joint(queryKeys, parameters.getEliminationFunction());

    StepResult result;
    result.stepIndex = stepIndex;
    result.currentKey = currentKey;
    result.poseCount = stepIndex + 1;
    result.factorCount = factorCount;
    result.updateMs =
        chrono::duration<double, milli>(updateEnd - updateStart).count();
    result.queryMeanMs = mean(queryTimes);
    result.queryMedianMs = median(queryTimes);
    result.queryStddevMs = stddev(queryTimes);
    result.supportCliques = supportStats.supportCliques;
    result.compressedCliques = supportStats.compressedCliques;
    result.reducedStateDim = totalDimension(reducedGraph.keys(), linearizationPoint);
    result.maxFrontalDim = supportStats.maxFrontalDim;
    result.maxSeparatorDim = supportStats.maxSeparatorDim;
    result.numCachedSeparatorMarginals = isam2.numCachedSeparatorMarginals();
    results.push_back(result);
  }

  filesystem::create_directories(outputPath.parent_path());
  writeCsv(outputPath, results);
  cout << "Wrote incremental timing results to " << outputPath << endl;
  return 0;
}
