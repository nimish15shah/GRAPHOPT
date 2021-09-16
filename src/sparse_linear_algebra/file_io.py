
import scipy.io
import scipy.sparse

def read_mat(path):
  print(path)
  mat = scipy.io.loadmat(path)
  print(mat['Problem'])
  for i in range(len(mat['Problem'][0][0])):
    obj= mat['Problem'][0][0][i]
    if scipy.sparse.issparse(obj):
      return obj

