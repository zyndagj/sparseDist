#!/usr/bin/env python

#The MIT License (MIT)
#
#Copyright (c) 2015 Greg Zynda
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import numpy as np
import scipy.sparse as ss

def _validate_svs(a,b):
	if not ss.issparse(a):
		raise TypeError("First vector isn't sparse")
	if not ss.issparse(b):
		raise TypeError("Second vector isn't sparse")
	if a.shape != b.shape:
		raise ValueError("Vector dimensions don't match")
	if len(a.shape) != 2 or 1 not in a.shape:
		raise ValueError("First vector isn't a 1-D matrix")
	if len(b.shape) != 2 or 1 not in b.shape:
		raise ValueError("Second vector isn't a 1-D matrix")

def _validate_sv(a):
	if not ss.issparse(a):
		raise TypeError("Vector isn't sparse")
	if len(a.shape) != 2 or 1 not in a.shape:
		raise ValueError("Vector isn't a 1-D matrix")

def svNorm(v,p=2):
	'''
	Returns the p-Norm of a sparse vector.
	TODO: add tests
	
	Check to see if 2-Norm works as expected.
	>>> svNorm(ss.csr_matrix(([1,2,3],([0,0,0],[20,30,10])),shape=(1,50)))
	3.7416573867739413

	Check to see if an error is thrown for matrices that aren't 1-D
	>>> svNorm(ss.csr_matrix(([1,2,3],([0,0,0],[20,30,10])),shape=(2,50)))
	Traceback (most recent call last):
	...
	ValueError: Vector isn't a 1-D matrix

	Check to see if it works with transposed vectors
	>>> svNorm(ss.csr_matrix(([1,2,3],([20,30,10],[0,0,0])),shape=(50,1)))
	3.7416573867739413
	
	Check to see if it works with zero-values
	>>> svNorm(ss.csr_matrix((1,50)))
	0.0
	
	>>> X = ss.csr_matrix([[1,2,3]])
	>>> svNorm(X,p=0)
	3
	>>> svNorm(X,p=1)
	6
	>>> svNorm(X)
	3.7416573867739413
	>>> svNorm(X,p=3)
	3.3019272488946263
	>>> svNorm(X,p=np.Inf)
	3
	'''
	_validate_sv(v)
	if p==0:
		return v.nnz
	if p==1:
		return np.sum(np.abs(v.data))
	if p==2:
		return np.sqrt(np.sum(v.data**2))
	if p==np.Inf:
		return np.max(np.abs(v.data))
	return np.add.reduce(np.abs(v.data)**p)**(1.0/p)

def cosine(a,b):
	'''
	Returns the cosine distance between two vectors
	
	>>> cosine(ss.csr_matrix([[0,2,0,1]]), ss.csr_matrix([[1,1,3,0]]))
	0.7303200550147031
	>>> cosine(ss.csr_matrix([[0,2,0,1]]), ss.csr_matrix([[1,0,3,0]]))
	1.0
	>>> cosine(ss.csr_matrix([[1,0,3,0]]), ss.csr_matrix([[1,0,3,0]])).round(3)
	0.0
	'''
	_validate_svs(a,b)
	N = svNorm(a)*svNorm(b)
	dProd = a.dot(b.transpose())
	if dProd.nnz:
		return 1.0-dProd.data[0]/N
	else:
		return 1.0

def minkowski(a,b,p=2):
	'''
	>>> A = [0,2,0,1]
	>>> B = [1,0,3,0]
	>>> C = [0,2,3,1]
	>>> minkowski(ss.csr_matrix([A]), ss.csr_matrix([A]))
	0.0
	>>> minkowski(ss.csr_matrix([A]), ss.csr_matrix([B]))
	3.872983346207417
	>>> minkowski(ss.csr_matrix([B]), ss.csr_matrix([A]))
	3.872983346207417
	>>> minkowski(ss.csr_matrix([C]), ss.csr_matrix([B]))
	2.4494897427831779
	>>> minkowski(ss.csr_matrix([B]), ss.csr_matrix([C]))
	2.4494897427831779
	'''
	_validate_svs(a,b)
	return svNorm(a-b,p)

def euclidean(a,b):
	'''
	>>> euclidean(ss.csr_matrix([[0,2,0,1]]), ss.csr_matrix([[1,0,3,0]]))
	3.872983346207417
	'''
	_validate_svs(a,b)
	return svNorm(a-b)

def correlation(a, b):
	'''
	Replicates the scipy correlation distance.

	>>> A = [0,2,0,1]
	>>> B = [1,0,3,0]
	>>> C = [0,2,3,1]
	>>> correlation(ss.csr_matrix([A]), ss.csr_matrix([B]))
	1.7385489458759964
	>>> correlation(ss.csr_matrix([C]), ss.csr_matrix([B]))
	0.45227744249483393
	'''
	_validate_svs(a,b)
	amu = a.mean()
	bmu = b.mean()
	am = a.toarray()[0,:]-amu
	bm = b.toarray()[0,:]-bmu
	N = np.linalg.norm(am)*np.linalg.norm(bm)
	dProd = np.dot(am,bm)
	return 1.0-dProd/N

def hamming(a,b):
	'''
	Replicates the scipy hamming distance that returns a fraction.

	>>> A = [0,2,0,1]
	>>> B = [1,0,3,0]
	>>> C = [0,2,3,1]
	>>> hamming(ss.csr_matrix([A]), ss.csr_matrix([B]))
	1.0
	>>> hamming(ss.csr_matrix([C]), ss.csr_matrix([B]))
	0.75
	'''
	_validate_svs(a,b)
	return a._binopt(b,'_ne_').nnz/float(max(a.shape))

def cityblock(a,b):
	'''
	>>> A = [0,2,0,1]
	>>> B = [1,0,3,0]
	>>> C = [0,2,3,1]
	>>> cityblock(ss.csr_matrix([A]), ss.csr_matrix([A]))
	0
	>>> cityblock(ss.csr_matrix([A]), ss.csr_matrix([B]))
	7
	>>> cityblock(ss.csr_matrix([B]), ss.csr_matrix([A]))
	7
	>>> cityblock(ss.csr_matrix([C]), ss.csr_matrix([B]))
	4
	>>> cityblock(ss.csr_matrix([B]), ss.csr_matrix([C]))
	4
	'''
	_validate_svs(a,b)
	return np.sum(np.abs(a-b).data)

def kullbackleibler(a,b):
	'''
	Implementation of the kullback-leibler divergance.
	>>> A = [0,2,0,1]
	>>> B = [1,0,3,0]
	>>> C = [0,2,3,1]
	>>> kullbackleibler(ss.csr_matrix([A]), ss.csr_matrix([A]))
	0
	>>> kullbackleibler(ss.csr_matrix([A]), ss.csr_matrix([B]))
	7
	>>> kullbackleibler(ss.csr_matrix([B]), ss.csr_matrix([A]))
	7
	>>> kullbackleibler(ss.csr_matrix([C]), ss.csr_matrix([B]))
	4
	>>> kullbackleibler(ss.csr_matrix([B]), ss.csr_matrix([C]))
	4
	'''
	_validate_svs(a,b)
	
	M = (a+b)/2.0
	0.5*D(a,M)+0.5*D(b,M)
	return np.sum(np.abs(a-b).data)

def jensenshannon(a,b):
	'''
	Implementation of the Jensen-Shannon divergence.

	https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
	
	Returns a distance in the range [0, Inf)

	>>> A = [0,2,0,1]
	>>> B = [1,0,3,0]
	>>> C = [0,2,3,1]
	>>> jensenshannon(ss.csr_matrix([A]), ss.csr_matrix([A]))
	0
	>>> jensenshannon(ss.csr_matrix([A]), ss.csr_matrix([B]))
	7
	>>> jensenshannon(ss.csr_matrix([B]), ss.csr_matrix([A]))
	7
	>>> jensenshannon(ss.csr_matrix([C]), ss.csr_matrix([B]))
	4
	>>> jensenshannon(ss.csr_matrix([B]), ss.csr_matrix([C]))
	4
	'''
	_validate_svs(a,b)
	## Check for frequencies
	if np.abs(np.add.reduce(a.data) - 1.0) > 0.0001:
		fA = a/np.add.reduce(a.data)
	else: fA = a
	if np.abs(np.add.reduce(b.data) - 1.0) > 0.0001:
		fB = b/np.add.reduce(b.data)
	else: fB = b
	M = (a+b)/2.0
	0.5*D(a,M)+0.5*D(b,M)
	return np.sum(np.abs(a-b).data)

def chebyshev(a,b):
	'''
	>>> A = [0,2,0,1]
	>>> B = [1,0,3,0]
	>>> C = [0,2,3,1]
	>>> chebyshev(ss.csr_matrix([A]), ss.csr_matrix([A]))
	0
	>>> chebyshev(ss.csr_matrix([A]), ss.csr_matrix([B]))
	3
	>>> chebyshev(ss.csr_matrix([C]), ss.csr_matrix([B]))
	2
	'''
	_validate_svs(a,b)
	return np.abs(a-b).max()

def braycurtis(a,b):
	'''
	Computes the Bray-Curtis distance between two 1-D sparse matrices. Implementation follows http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.braycurtis.html#scipy.spatial.distance.braycurtis

	```
	sum(|a-b|)/sum(|a+b|)
	```
	
	The Bray-Curtis distance lies in the range [0,1].

	>>> a = [0,2,0,1]
	>>> b = [1,0,3,0]
	>>> c = [0,2,3,1]
	>>> braycurtis(ss.csr_matrix([a]), ss.csr_matrix([b]))
	1.0
	>>> braycurtis(ss.csr_matrix([c]), ss.csr_matrix([b]))
	0.40000000000000002
	'''
	_validate_svs(a,b)
	return np.add.reduce(np.abs((a-b).data)) / float(np.add.reduce(np.abs((a+b).data)))

def canberra(a,b):
	'''
	Computes the Canberra distance, a weighted taxicab distance,
	between two 1-D sparse matrices and returns a distance
	value [0, Inf].

	sum(|a-b|/(|a|+|b|))

	>>> a = [0,2,0,1]
	>>> b = [1,0,3,0]
	>>> c = [0,2,3,1]
	>>> canberra(ss.csr_matrix([a]), ss.csr_matrix([b]))
	4.0
	>>> canberra(ss.csr_matrix([c]), ss.csr_matrix([b]))
	3.0
	'''
	_validate_svs(a,b)
	apb = np.abs(a)+np.abs(b)
	amb = np.abs(a-b).asfptype()
	return np.add.reduce(amb[0,apb.indices].toarray()[0]/apb.data)
