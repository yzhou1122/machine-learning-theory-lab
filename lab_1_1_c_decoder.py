
# coding: utf-8

import numpy as np
import time


def load_decode_input(fn, m=100, K=26, dim=128):

    arr = []
    with open(fn, "r") as f:
        for val in f:
            arr.append(float(val.rstrip()))

    arr = np.array(arr, dtype='float64')

    X = arr[:m*dim].reshape(-1, dim)
    W = arr[m*dim:m*dim+K*dim].reshape(dim, -1)
    T = arr[m*dim+K*dim:].reshape(K, -1)

    print(arr.shape, X.shape, W.shape, T.shape)

    return X, W, T


def decoder(X, W, T):
    """
    X: (sequence(time) length,vector dim matrix, e.g. (100,128)), with each row a vector representation of the letter xi
    W: (vector dim, number of labels matrix, e.g. (128,26)), with each column a vector weights for the label yi
    T: (# labels, # labels matrix): transition table Tij.
    """
    time_length = X.shape[0]
    num_labels = T.shape[0]

    max_table = np.zeros((time_length, num_labels))
    argmax_table = np.zeros((time_length, num_labels), dtype='int64')

    t = 0
    for label_id in range(num_labels):
        max_table[t, label_id] = 0
    for t in range(1, time_length):
        for label_id in range(num_labels):

            max_value_arr = []
            for prev_label_id in range(num_labels):

                v = np.dot(X[t-1], W[:, prev_label_id]) + T[prev_label_id,
                                                            label_id] + max_table[t-1, prev_label_id]
                max_value_arr.append(v)

            max_value = max(max_value_arr)
            max_label_id = np.argmax(max_value_arr)

            max_table[t, label_id] = max_value
            argmax_table[t, label_id] = max_label_id

    sequence = []

    v_w_k_x_k = []

    for final_label_id in range(num_labels):

        v_w_k_x_k.append(np.dot(X[time_length-1], W[:, final_label_id]))

    v_w_k_x_k = np.array(v_w_k_x_k)

    # this is correct. need wym * xm term
    next_label = (max_table[time_length-1]+v_w_k_x_k).argmax()
    print(max(max_table[time_length-1]+v_w_k_x_k))  # for report only, this is the max objective

    sequence.append(next_label)
    for t in range(time_length-1, -1, -1):
        next_label = argmax_table[t, next_label]
        sequence.append(next_label)
    return [idx+1 for idx in sequence[::-1][1:]]


def decoder_brute(X, W, T):
    """
    X: (sequence(time) length,vector dim matrix, e.g. (100,128)), with each row a vector representation of the letter xi
    W: (vector dim, number of labels matrix, e.g. (128,26)), with each column a vector weights for the label yi
    T: (# labels, # labels matrix): transition table Tij.
    """
    def _get_permuted_lst(num_labels, time_length):

        import itertools

        return [list(pair) for pair in itertools.product(range(num_labels), repeat=time_length)]

    def _get_score(seq, X, W, T):
        # seq = [1,9,3,4,0]
        leftsum = 0
        rightsum = 0

        for seq_idx, label_id in enumerate(seq):

            leftsum += np.dot(X[seq_idx], W[:, label_id])

        for seq_idx in range(len(seq)-1):

            rightsum += T[seq[seq_idx], seq[seq_idx+1]]

        return leftsum + rightsum

    time_length = X.shape[0]
    num_labels = T.shape[0]

    all_possible_seqs = _get_permuted_lst(num_labels, time_length)

    dic = {}  # store seq->score

    for seq in all_possible_seqs:

        v = _get_score(seq, X, W, T)

        k = str(seq)

        dic[k] = v

    best = max(dic, key=dic.get)
    print(dic[best])  # this is max objective

    return [i+1 for i in eval(best)]


if __name__ == '__main__':

    # test on dummy inputs
    Xdummy = np.random.rand(5, 64)
    Wdummy = np.random.rand(64, 10)
    Tdummy = np.random.rand(10, 10)

    start_time = time.time()
    l = decoder(Xdummy, Wdummy, Tdummy)  # 0 - 9
    print(l)
    print([chr(ord('`')+idx) for idx in l])
    print((time.time() - start_time))

    start_time = time.time()
    l2 = decoder_brute(Xdummy, Wdummy, Tdummy)  # 0 - 9
    print(l2)
    print([chr(ord('`')+idx) for idx in l2])
    print((time.time() - start_time))

    # read decode input for lab 1
    fn = 'decode_input.txt'
    X, W, T = load_decode_input(fn)

    start_time = time.time()
    l3 = decoder(X, W, T)  # 0 - 9
    print(l3)  # 205.6819266768395 max objective
    print([chr(ord('`')+idx) for idx in l3])
    print((time.time() - start_time))
    print(len(l3))

    with open('decode_output.txt', 'w') as f:
        for item in l3:
            f.write("%s\n" % item)
