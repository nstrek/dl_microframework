# X = np.random.random((6, 10))
# A = np.random.random((2, 10))
#
# print(A[None, :, :].shape)
#
# print(np.matmul(A[None, :, :], X[:, :, None]).squeeze(-1))

###################################################################

# X = np.random.random((100, 10))
# A = np.random.random((2, 10))
# b = np.random.random(2)
#
# print(A[None, :, :].shape)
#
# print((np.matmul(A[None, :, :], X[:, :, None]) + b[:, None]).squeeze(-1).shape)

###################################################################

# dense = Dense(10, 2)
#
# X = Tensor(np.random.random((100, 10)))
#
# print(dense.forward(X))

###################################################################

