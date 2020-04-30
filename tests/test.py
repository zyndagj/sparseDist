from scipy.stats import entropy
from scipy.sparse import csr_matrix
import numpy as np

A = [0,2,0,1]
print csr_matrix(A)/sum(A)

A = np.array([0,2,0,1],dtype=np.float32)
B = np.array([1,0,3,0],dtype=np.float32)
C = np.array([0,2,3,1],dtype=np.float32)
D = np.array([1,4,3,5],dtype=np.float32)

sA = csr_matrix(A)
sD = csr_matrix(D)

Af = A/sum(A)
print Af
Bf = B/sum(B)
Df = D/sum(D)

print entropy(A,B)
print entropy(A,D)
print entropy(sA,sD)
agz = A>0
print Af[agz]*np.log(Af[agz]/Bf[agz])
print np.sum(Af[agz]*np.log(Af[agz]/Df[agz]))
