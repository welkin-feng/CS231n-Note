一、 Python Numpy Tutorial

Numpy

1. 创建矩阵：（create matrix）
a1 = np.array([1,2,3,4])
a2 = np.array([[1,2],[3,4]])

a = np.zeros((2,2))
b = np.ones((1,2))
c = np.full((2,2), 7)
d = np.eye(2)
e = np.random.random((2,2))


2. 访问矩阵
bool_idx = (a > 2) # 根据a条件得到一个bool矩阵
