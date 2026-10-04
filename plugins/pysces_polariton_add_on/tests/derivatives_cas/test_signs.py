import numpy as np
import itertools

length = 5
combos = itertools.combinations_with_replacement(range(length), length)
combos = list(combos)
for c in combos:
    signs = np.ones(length, dtype=int)
    s_mat = np.zeros((length, length))
    for i in c:
        signs[i] = -1
    for i in range(length):
        for j in range(length):
            s_mat[i, j] = signs[i]*signs[j]
        # j = 0
        # s_mat[i, j] = s_mat[j, i] = signs[i]*signs[j]


    # print(s_mat/s_mat[:, 0])
    ratio = s_mat/s_mat[:, 0]
    ratio = ratio[:, 0].astype(int)
    print(ratio == signs, signs, np.sum(signs))

