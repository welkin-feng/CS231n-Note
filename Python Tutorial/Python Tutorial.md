<h1>一、 Python Numpy Tutorial</h1>

<h2>Numpy</h2>
<br>
<h3>1. 创建矩阵：（create matrix）</h3> <br>
a1 = np.array([1,2,3,4]) <br>
a2 = np.array([[1,2],[3,4]]) <br>
<br>
a = np.zeros((2,2)) <br>
b = np.ones((1,2)) <br>
c = np.full((2,2), 7) <br>
d = np.eye(2) <br>
e = np.random.random((2,2)) <br>
<br><br>

<h3>2. 访问矩阵 </h3>
bool_idx = (a > 2) # 根据a条件得到一个bool矩阵 <br>
