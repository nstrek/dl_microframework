# X = np.random.random((6, 10))
# A = np.random.random((2, 10))
#
# print(A[None, :, :].shape)
#
# print(np.matmul(A[None, :, :], X[:, :, None]).squeeze(-1))