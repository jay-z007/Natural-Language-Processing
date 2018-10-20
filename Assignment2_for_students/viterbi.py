import numpy as np

def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is the size N array/seq of integers representing the best sequence.
    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]

    emission_scores = emission_scores.T # [L, N]
    best_previous_scores = (start_scores + emission_scores[:, 0]).reshape(L, 1) # [L, 1]
    back_pointers = np.zeros((L, N), dtype=int)

    y = []

    for i in xrange(1, N):
        # add best_previous_score to the transition matrix, elementwise
        # This will result in a matrix of size [L, L].
        # Then add the corresponding emission prob and find the max in each column
        # This will become the new best_previous_scores for the next loop/column

        new_scores = emission_scores[:, i] + best_previous_scores + trans_scores
        back_pointers[:, i] = np.argmax(new_scores, axis=0)
        best_previous_scores = np.amax(new_scores, axis=0).reshape(L, 1)

    best_previous_scores = np.squeeze(best_previous_scores) 

    best_previous_scores += end_scores
    max_ind = np.argmax(best_previous_scores)

	# Backtracking the best path
    for i in range(1, N)[::-1]:
        y.append(max_ind)
        max_ind = back_pointers[max_ind, i]

    y.append(max_ind)

	# return the best score and the reversed sequence. 
    return (np.amax(best_previous_scores), y[::-1])
