/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file testPriorityScheduler.cpp
 * @brief Unit tests for PriorityScheduler with simplified recursive API.
 * @author Frank Dellaert
 * @date May, 2025
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/PriorityScheduler.h>

#include <atomic>  // For void test counter
#include <functional>
#include <map>  // For A* closed set
#include <memory>
#include <optional>  // For A* result
#include <vector>

using namespace gtsam;

// Common TreeNode definition (using shared_ptr)
struct TreeNode {
  int value;
  std::vector<std::shared_ptr<TreeNode>> children;
  TreeNode(int v) : value(v) {}
  void addChild(std::shared_ptr<TreeNode> child) { children.push_back(child); }
};

namespace {
std::shared_ptr<TreeNode> createSimpleTree() {
  auto n5 = std::make_shared<TreeNode>(5);
  auto n3 = std::make_shared<TreeNode>(3);
  auto n8 = std::make_shared<TreeNode>(8);
  auto n2 = std::make_shared<TreeNode>(2);
  auto n1 = std::make_shared<TreeNode>(1);
  auto n4 = std::make_shared<TreeNode>(4);
  auto n7 = std::make_shared<TreeNode>(7);
  n5->children = {n3, n8, n2};
  n3->children = {n1, n4};
  n2->children = {n7};
  return n5;
}
}  // namespace

/* ************************************************************************* */
// Test 1: Recursive Node Counting (returns size_t)
/* ************************************************************************* */
namespace {
size_t countNodesRecursive(
    std::shared_ptr<TreeNode> node,
    RecursiveTaskContext<size_t, std::shared_ptr<TreeNode>> context) {
  if (!node) return 0;
  size_t count = 1;
  double childPriorityBase = static_cast<double>(node->value);

  for (const auto& child : node->children) {
    count += context.processChild(child, childPriorityBase - 1.0,
                                  countNodesRecursive);
  }
  return count;
}
}  // namespace

TEST(PriorityScheduler, RecursiveNodeCounting) {
  std::shared_ptr<TreeNode> root = createSimpleTree();
  PriorityScheduler<size_t> scheduler(std::thread::hardware_concurrency());

  std::function<size_t(std::shared_ptr<TreeNode>,
                       RecursiveTaskContext<size_t, std::shared_ptr<TreeNode>>)>
      countFunc = countNodesRecursive;
  size_t totalCount = scheduler.runRecursive(root, 0.0, countFunc);

  const size_t expectedTotalNodes = 7;
  EXPECT_LONGS_EQUAL(expectedTotalNodes, totalCount);
}

/* ************************************************************************* */
// Test 2: Recursive Void Tasks
/* ************************************************************************* */
namespace {
std::atomic<size_t> globalVoidTaskCounter{0};

void processNodeVoid(
    std::shared_ptr<TreeNode> node,
    RecursiveTaskContext<void, std::shared_ptr<TreeNode>> context) {
  if (!node) {
    return;
  }
  globalVoidTaskCounter.fetch_add(1, std::memory_order_relaxed);
  double childPriorityBase = static_cast<double>(node->value);

  for (const auto& child : node->children) {
    context.processChild(child, childPriorityBase - 1.0, processNodeVoid);
  }
}
}  // namespace

TEST(PriorityScheduler, testSimplifiedRecursiveVoidTasks) {
  std::shared_ptr<TreeNode> root = createSimpleTree();
  PriorityScheduler<void> scheduler(std::thread::hardware_concurrency());
  globalVoidTaskCounter.store(0);

  // FIX: Wrap the function pointer in std::function
  std::function<void(std::shared_ptr<TreeNode>,
                     RecursiveTaskContext<void, std::shared_ptr<TreeNode>>)>
      processFunc = processNodeVoid;
  scheduler.runRecursive(root, 0.0, processFunc);

  const size_t expectedTotalNodes = 7;
  EXPECT_LONGS_EQUAL(expectedTotalNodes, globalVoidTaskCounter.load());
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */