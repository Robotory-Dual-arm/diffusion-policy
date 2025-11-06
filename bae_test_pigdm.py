import numpy as np


a = np.array([[[1,2,3],[4,5,6],[7,8,9],[10,11,12]],
              [[13,14,15],[16,17,18],[19,20,21],[22,23,24]]])

print(a[..., :2])

w = np.array([10, 10, 10, 10])

print(w[:, None])
print(a * w[:, None])

b = np.array([[[1,2,3],[4,5,6],[7,8,9],[10,11,12]]])
c = np.zeros_like(b)

b = b[:, 2:]

c[:, :4 - len(b[0])] = b


print(b.shape)
print(c)