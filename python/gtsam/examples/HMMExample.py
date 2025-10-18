"""
GTSAM Copyright 2010-2020, Georgia Tech Research Corporation,
Atlanta, Georgia 30332-0415
All Rights Reserved
Authors: Frank Dellaert, et al. (see THANKS for the full author list)

See LICENSE for the license information

Hidden Markov Model example, discrete.

Author: Frank Dellaert
Date: July 12, 2020
"""

import numpy as np

import gtsam
from gtsam import (DiscreteBayesNet, DiscreteFactorGraph, DiscreteMarginals,
                   DiscreteKeys, Ordering)


def main():
    """
    This example demonstrates a Hidden Markov Model (HMM) using discrete Bayes nets.
    
    A Hidden Markov Model consists of:
    - A sequence of hidden states over time
    - Transition probabilities between states
    - Observation/measurement probabilities for each state
    
    In this example:
    - We have 4 nodes (time steps) with 3 possible states each
    - We define a transition model that favors staying in the same state
    - We add measurements/observations at some (not all) time steps
    """
    
    nr_nodes = 4
    nr_states = 3

    # Define variables as well as ordering
    ordering = Ordering()
    keys = []
    for k in range(nr_nodes):
        key_i = (k, nr_states)
        keys.append(key_i)
        ordering.push_back(k)

    # Create HMM as a DiscreteBayesNet
    hmm = DiscreteBayesNet()

    # Define backbone transition model
    # "8/1/1 1/8/1 1/1/8" means the transition matrix favors staying in the same state
    # For each parent state (0,1,2), it gives probabilities for child states (0,1,2)
    transition = "8/1/1 1/8/1 1/1/8"
    
    # Add transition conditionals for k=1,2,3 conditioned on k-1
    for k in range(1, nr_nodes):
        # Helper function to create DiscreteKeys for parents
        parents = DiscreteKeys()
        parents.push_back(keys[k - 1])
        hmm.add(keys[k], parents, transition)

    # Add some measurements, not needed for all time steps!
    # These are observation likelihoods
    hmm.add(keys[0], "7/2/1")  # Measurement at time 0
    hmm.add(keys[1], "1/9/0")  # Measurement at time 1
    hmm.add(keys[3], "5/4/1")  # Measurement at time 3 (last node)

    # Print the HMM
    hmm.print("HMM\n")

    # Convert to factor graph
    factor_graph = DiscreteFactorGraph(hmm)

    # Do max-product to find the most probable explanation (MPE)
    mpe = factor_graph.optimize()
    print("mpe:", end="")
    for k in range(nr_nodes):
        print(f" ({k}, {mpe[k]})", end="")
    print()

    # Create solver and eliminate
    # This will create a DAG ordered with arrow of time reversed
    chordal = factor_graph.eliminateSequential(ordering)
    chordal.print("Eliminated\n")

    # We can also sample from it
    print("\n10 samples:")
    for k in range(10):
        sample = chordal.sample()
        print("sample:", end="")
        for i in range(nr_nodes):
            print(f" ({i}, {sample[i]})", end="")
        print()

    # Or compute the marginals. This re-eliminates the FG into a Bayes tree
    print("\nComputing Node Marginals ..")
    marginals = DiscreteMarginals(factor_graph)
    for k in range(nr_nodes):
        marg_probs = marginals.marginalProbabilities((k, nr_states))
        print(f"marginal {k}{marg_probs}")


if __name__ == "__main__":
    main()

