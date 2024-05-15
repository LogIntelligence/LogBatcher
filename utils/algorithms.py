from collections import Counter
import math
import numpy as np

def dpp_sample(S, k):
    # S: similarity matrix
    # k: number of items to sample
    n = S.shape[0]

    # Initialize empty set Y
    Y = set()

    for _ in range(k):
        best_i = -1
        best_p = -1

        for i in range(n):
            if i not in Y:
                # Compute determinant of submatrix
                det_Yi = np.linalg.det(S[np.ix_(list(Y) + [i], list(Y) + [i])])

                # Compute probability of adding i to Y
                p_add = det_Yi / (1 + det_Yi)

                if p_add > best_p:
                    best_p = p_add
                    best_i = i

        # Add best item to Y
        Y.add(best_i)

    return list(Y)


def entropy_calculate(inputs, k):
    entropies = []
    for input in inputs:
        # input is a log-template pair, if you want to calculate the entropy by varaible, use extract_variables function instead
        calculate_list = list(input[0]) + list(input[1])
        counter = Counter(calculate_list)
        probs = [count / len(calculate_list) for count in counter.values()]
        entropy = -sum(p * math.log2(p) for p in probs)
        entropies.append((input, entropy))
    
    # sort by entropy
    sorted_pairs = sorted(entropies, key=lambda x: x[1], reverse=True)

    # select top-k pairs
    selected_pairs = sorted_pairs[:k]
    return [pair for pair in selected_pairs]
