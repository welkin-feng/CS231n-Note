<h1>Python Numpy Tutorial</h1>

<h2>Numpy</h2>
<br>
<h3>1. 创建数组 (Create array)</h3> <br>
a1 = np.array([1,2,3,4]) <br>
a2 = np.array([[1,2],[3,4]]) <br>
<br>
a = np.zeros((2,2)) <br>
b = np.ones((1,2)) <br>
c = np.full((2,2), 7) <br>
d = np.eye(2) <br>
e = np.random.random((2,2)) <br>
<br><br>

<h3>2. 访问数组 (Array indexing)</h3>
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]]) <br>
b = np.array([0, 2, 0, 1]) <br>
print(a[np.arange(4), b])  # Prints "[ 1  6  7 11]"  <br>
a[np.arange(4), b] += 10  <br>
print(a)  # prints "array([[11,  2,  3],  <br>
          #                [ 4,  5, 16],  <br>
          #                [17,  8,  9],  <br>
          #                [10, 21, 12]]) <br>
bool_idx = (a > 2) # 根据a条件得到一个bool矩阵 <br>
