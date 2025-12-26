/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file PriorityScheduler.h
 * @brief Priority-based task scheduler with a lightweight recursive API.
 *
 * @details
 * This header defines a small, thread-based scheduler that executes tasks in
 * priority order. A lower numeric priority is executed before a higher numeric
 * priority (min-heap behavior). Tasks return a value of type `Y`, and callers
 * can wait on results via `std::future<Y>`.
 *
 * The scheduler also supports a recursive execution model via
 * `RecursiveTaskContext`. A recursive function can enqueue child tasks with
 * specific priorities and block until each child returns, while still allowing
 * other worker threads to make progress on queued tasks.
 *
 * @par Example
 * @code
 * // Compute a sum over a tree where children are scheduled by priority.
 * struct Node { std::vector<Node> children; int value; };
 *
 * gtsam::PriorityScheduler<int> scheduler(4);
 *
 * auto sum = scheduler.runRecursive<Node>(
 *     root, 0.0, // initialPriority
 *     [](Node node, gtsam::RecursiveTaskContext<int, Node> ctx) {
 *       int total = node.value;
 *       double childPriority = 1.0;
 *       for (const auto& child : node.children) {
 *         total += ctx.processChild(child, childPriority,
 *             [](Node c, gtsam::RecursiveTaskContext<int, Node> childCtx) {
 *               return childCtx.runSubTask(2.0, [c]() { return c.value; });
 *             });
 *         childPriority += 1.0;
 *       }
 *       return total;
 *     });
 * @endcode
 *
 * @note `runRecursive` blocks until the root task finishes. Each recursive
 * call can enqueue more work and synchronously wait on its results.
 *
 * @author Frank Dellaert
 * @date May, 2025
 */

#pragma once

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>  // For std::runtime_error
#include <thread>
#include <type_traits>  // For std::is_void_v
#include <vector>

namespace gtsam {

/// Forward declaration for the recursive context helper.
template <typename Y>
class PriorityScheduler;

/**
 * @brief Helper passed into recursive tasks to schedule child work.
 *
 * @details This class is intentionally lightweight: it holds a raw pointer to
 * the scheduler and offers convenience methods that enqueue a task and
 * immediately wait for its result.
 *
 * @tparam Y Result type produced by scheduled tasks.
 * @tparam Node node type used by the recursive user function.
 */
template <typename Y, typename Node>
class RecursiveTaskContext {
  PriorityScheduler<Y>* scheduler_;

 public:
  /// Construct a context tied to a scheduler.
  explicit RecursiveTaskContext(PriorityScheduler<Y>* scheduler)
      : scheduler_(scheduler) {}

  /**
   * @brief Schedule a recursive child task and wait for its result.
   *
   * @param childNode Node to pass to the recursive function.
   * @param priority Lower values execute before higher values.
   * @param recursiveFunc The user-provided recursive function.
   * @return Result produced by the child task.
   */
  Y processChild(Node childNode, double priority,
                 const std::function<Y(Node, RecursiveTaskContext<Y, Node>)>&
                     recursiveFunc) {
    assert(scheduler_);

    // The lambda captures recursiveFunc by reference (as it's a const ref
    // parameter) and scheduler by pointer. childNode is copied.
    auto childJob = [childNode, scheduler = scheduler_, &recursiveFunc]() -> Y {
      RecursiveTaskContext<Y, Node> childContext(scheduler);
      return recursiveFunc(childNode, childContext);
    };

    std::future<Y> future = scheduler_->schedule(priority, std::move(childJob));
    return future.get();
  }

  /**
   * @brief Schedule an arbitrary subtask and wait for its result.
   *
   * @param priority Lower values execute before higher values.
   * @param subTaskJob Callable representing the subtask.
   * @return Result produced by the subtask.
   */
  Y runSubTask(double priority, std::function<Y()> subTaskJob) {
    assert(scheduler_);
    std::future<Y> future =
        scheduler_->schedule(priority, std::move(subTaskJob));
    return future.get();
  }
};

/**
 * @brief Thread pool scheduler that prioritizes tasks by numeric priority.
 *
 * @details
 * - Lower numeric values are executed first.
 * - Tasks are executed by worker threads created at construction.
 * - `schedule` returns a `std::future<Y>` for the task result.
 * - `runRecursive` executes a root task and allows it to spawn child tasks
 *   through `RecursiveTaskContext`.
 *
 * @tparam Y Result type returned by tasks. Use `void` for no return value.
 */
template <typename Y>
class PriorityScheduler {
  struct Task {
    double priority;
    std::function<Y()> job;
    std::promise<Y> promise;

    Task(double p, std::function<Y()> j, std::promise<Y> _promise)
        : priority(p), job(std::move(j)), promise(std::move(_promise)) {}
  };

  using TaskPtr = std::shared_ptr<Task>;

  struct Compare {
    bool operator()(const TaskPtr& a, const TaskPtr& b) const {
      return a->priority > b->priority;
    }
  };

  std::priority_queue<TaskPtr, std::vector<TaskPtr>, Compare> taskQueue_;
  std::vector<std::thread> workers_;
  mutable std::mutex queueMutex_;
  std::condition_variable condition_;
  std::atomic<bool> stop_{false};
  std::atomic<int> activeTasks_{0};

  /**
   * @brief Worker thread function for processing tasks from the priority queue.
   *
   * This function runs in a loop, waiting for tasks to become available in the
   * task queue. It uses a condition variable (`condition_`) to efficiently wait
   * until either:
   *   - The scheduler is stopped (`stop_` is true), or
   *   - There is at least one task in the queue (`!taskQueue_.empty()`).
   *
   * The condition variable is notified whenever a new task is added or when the
   * scheduler is stopped, allowing the worker thread to wake up and check the
   * queue or exit if needed. The use of the condition variable ensures that
   * worker threads do not spin-wait, but instead sleep efficiently until work
   * is available or shutdown is requested.
   *
   * When a task is available, the worker thread pops the highest-priority task
   * from the queue, executes it, and sets the result or exception in the
   * associated promise. After processing a task, it decrements the count of
   * active tasks and notifies all waiting threads via the condition variable,
   * in case other threads are waiting for tasks to complete.
   *
   */
  void worker_thread() {
    while (true) {
      TaskPtr task;
      {
        std::unique_lock<std::mutex> lock(queueMutex_);
        condition_.wait(lock, [this] {
          return stop_.load(std::memory_order_relaxed) || !taskQueue_.empty();
        });

        if (taskQueue_.empty()) {
          if (stop_.load(std::memory_order_relaxed)) return;
          continue;
        }

        // Pop the highest-priority task:
        task = taskQueue_.top();
        taskQueue_.pop();
      }

      try {
        // Execute the task and set the promise value:
        if constexpr (std::is_void_v<Y>) {
          task->job();
          task->promise.set_value();
        } else {
          task->promise.set_value(task->job());
        }
      } catch (...) {
        // On exception, set the exception on the promise:
        try {
          task->promise.set_exception(std::current_exception());
        } catch (...) { /* ignore */
        }
      }

      // Decrement active task count and notify waiting threads:
      std::lock_guard<std::mutex> lock(queueMutex_);
      activeTasks_.fetch_sub(1, std::memory_order_relaxed);
      condition_.notify_all();
    }
  }

 public:
  /**
   * @brief Construct a scheduler with a fixed number of worker threads.
   *
   * @param numThreads Number of worker threads to create. If zero, a single
   * thread is created.
   */
  PriorityScheduler(size_t numThreads = std::thread::hardware_concurrency()) {
    if (numThreads == 0) numThreads = 1;
    for (size_t i = 0; i < numThreads; ++i) {
      workers_.emplace_back(&PriorityScheduler::worker_thread, this);
    }
  }

  /**
   * @brief Wait for all tasks to finish, then stop worker threads.
   *
   * @note The destructor calls `waitForAllTasks` before stopping workers.
   */
  ~PriorityScheduler() {
    waitForAllTasks();
    stop_.store(true, std::memory_order_relaxed);
    condition_.notify_all();
    for (std::thread& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  /**
   * @brief Enqueue a task for execution.
   *
   * @param priority Lower values execute before higher values.
   * @param job Callable returning a `Y`.
   * @return `std::future<Y>` associated with the task.
   */
  std::future<Y> schedule(double priority, std::function<Y()> job) {
    if (stop_.load(std::memory_order_relaxed)) {
      std::promise<Y> err_promise;
      err_promise.set_exception(std::make_exception_ptr(
          std::runtime_error("Scheduler is stopping or stopped.")));
      return err_promise.get_future();
    }

    std::promise<Y> promise;
    std::future<Y> future = promise.get_future();
    auto task =
        std::make_shared<Task>(priority, std::move(job), std::move(promise));

    {
      std::lock_guard<std::mutex> lock(queueMutex_);
      taskQueue_.push(task);
      activeTasks_.fetch_add(1, std::memory_order_relaxed);
    }
    condition_.notify_one();
    return future;
  }

  /**
   * @brief Execute a recursive task starting at a root node.
   *
   * @details The user function can schedule child tasks via the provided
   * `RecursiveTaskContext`. This call blocks until the root task completes.
   *
   * @param rootNode The starting node for recursion.
   * @param initialPriority Priority for the root task.
   * @param userRecursiveFunc The recursive function implementation.
   * @return Result of the root task.
   */
  template <typename Node>
  Y runRecursive(
      Node rootNode, double initialPriority,
      // This std::function is copied when topLevelJob is created.
      std::function<Y(Node, RecursiveTaskContext<Y, Node>)> userRecursiveFunc) {
    auto topLevelJob = [this, rootNode,
                        userRecursiveFunc_copy =
                            std::move(userRecursiveFunc)]() -> Y {
      RecursiveTaskContext<Y, Node> rootContext(this);
      return userRecursiveFunc_copy(rootNode, rootContext);
    };

    std::future<Y> topLevelFuture =
        schedule(initialPriority, std::move(topLevelJob));
    return topLevelFuture.get();
  }

  /**
   * @brief Block until all queued and active tasks complete.
   *
   * @note If the scheduler is stopping, this returns early.
   */
  void waitForAllTasks() {
    std::unique_lock<std::mutex> lock(queueMutex_);
    condition_.wait(lock, [this] {
      return stop_.load(std::memory_order_relaxed) ||
             (activeTasks_.load(std::memory_order_relaxed) == 0 &&
              taskQueue_.empty());
    });
  }

  template <typename Y_friend, typename Node_friend>
  friend class RecursiveTaskContext;
};

}  // namespace gtsam
