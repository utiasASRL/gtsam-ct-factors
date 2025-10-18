"""
GTSAM Copyright 2010, Georgia Tech Research Corporation,
Atlanta, Georgia 30332-0415
All Rights Reserved
Authors: Frank Dellaert, et al. (see THANKS for the full author list)

See LICENSE for the license information

Discrete Bayes Net example with famous Asia Bayes Network

Author: Frank Dellaert
Date: July 10, 2020
"""

import gtsam
from gtsam import (DiscreteBayesNet, DiscreteFactorGraph, DiscreteKeys, 
                   Ordering)


def create_discrete_keys(*args):
    """Create a DiscreteKeys instance from a variable number of DiscreteKey pairs."""
    dks = DiscreteKeys()
    for key in args:
        dks.push_back(key)
    return dks


def main():
    """
    This example demonstrates the famous Asia Bayes Network using discrete Bayes nets.
    
    The Asia network is a classic example in probabilistic reasoning that models
    relationships between:
    - Visiting Asia (travel history)
    - Smoking habits
    - Diseases: Tuberculosis, Lung Cancer, Bronchitis
    - Symptoms: Dyspnea (shortness of breath)
    - Tests: X-Ray results
    
    The network shows how these variables are conditionally dependent on each other.
    """
    
    # Create the Asia Bayes Network
    asia = DiscreteBayesNet()
    
    # Define discrete keys for each variable (key_id, num_states)
    # Using the same key IDs as the C++ version for consistency
    Asia = (0, 2)        # Been to Asia: No=0, Yes=1
    Smoking = (4, 2)     # Smoking: No=0, Yes=1  
    Tuberculosis = (3, 2) # Tuberculosis: No=0, Yes=1
    LungCancer = (6, 2)   # Lung Cancer: No=0, Yes=1
    Bronchitis = (7, 2)   # Bronchitis: No=0, Yes=1
    Either = (5, 2)       # Either TB or LC: No=0, Yes=1
    XRay = (2, 2)         # X-Ray positive: No=0, Yes=1
    Dyspnea = (1, 2)      # Dyspnea: No=0, Yes=1
    
    # Add prior probabilities
    asia.add(Asia, "99/1")        # P(Asia) = [0.99, 0.01]
    asia.add(Smoking, "50/50")    # P(Smoking) = [0.5, 0.5]
    
    # Add conditional probabilities
    # P(Tuberculosis | Asia)
    asia.add(Tuberculosis, create_discrete_keys(Asia), "99/1 95/5")
    
    # P(LungCancer | Smoking)  
    asia.add(LungCancer, create_discrete_keys(Smoking), "99/1 90/10")
    
    # P(Bronchitis | Smoking)
    asia.add(Bronchitis, create_discrete_keys(Smoking), "70/30 40/60")
    
    # P(Either | Tuberculosis, LungCancer) - OR gate: Either = TB OR LC
    # "F T T T" means: P(Either=1|TB,LC) = [False, True, True, True]
    # for combinations (TB=0,LC=0), (TB=0,LC=1), (TB=1,LC=0), (TB=1,LC=1)
    asia.add(Either, create_discrete_keys(Tuberculosis, LungCancer), "F T T T")
    
    # P(XRay | Either)
    asia.add(XRay, create_discrete_keys(Either), "95/5 2/98")
    
    # P(Dyspnea | Either, Bronchitis)
    asia.add(Dyspnea, create_discrete_keys(Either, Bronchitis), "9/1 2/8 3/7 1/9")
    
    # Print the network with pretty variable names
    pretty_names = ["Asia", "Dyspnea", "XRay", "Tuberculosis", 
                   "Smoking", "Either", "LungCancer", "Bronchitis"]
    
    def formatter(key):
        return pretty_names[key]
    
    asia.print("Asia", formatter)
    
    # Convert to factor graph
    fg = DiscreteFactorGraph(asia)
    
    # Create elimination ordering
    ordering = Ordering()
    for i in [0, 1, 2, 3, 4, 5, 6, 7]:
        ordering.push_back(i)
    
    # Solve for most probable explanation (MPE)
    mpe = fg.optimize()
    print("mpe:", end="")
    for i in range(8):
        print(f" ({i}, {mpe[i]})", end="")
    print()
    
    # Build a Bayes tree (directed junction tree)
    bayes_tree = fg.eliminateMultifrontal(ordering)
    bayes_tree.print("bayesTree", formatter)
    
    # Add evidence: we were in Asia and we have dyspnea
    fg.add(Asia, "0 1")      # Evidence: Asia = 1 (Yes, been to Asia)
    fg.add(Dyspnea, "0 1")   # Evidence: Dyspnea = 1 (Yes, have dyspnea)
    
    # Solve again with evidence
    mpe2 = fg.optimize()
    print("mpe2:", end="")
    for i in range(8):
        print(f" ({i}, {mpe2[i]})", end="")
    print()
    
    # Sample from the posterior distribution
    chordal = fg.eliminateSequential(ordering)
    print("\n10 samples:")
    for i in range(10):
        sample = chordal.sample()
        print("sample:", end="")
        for j in range(8):
            print(f" ({j}, {sample[j]})", end="")
        print()


if __name__ == "__main__":
    main()
