# sparseDist
sparseDist is a reimplementation of the [scipy.spatial.distance](http://docs.scipy.org/doc/scipy/reference/spatial.distance.html) functions to support the input of sparse 1-D matrices.

# Distance Functions

## Bray-Curtis

**sparseDist.braycurtis(a,b)**

Computes the Bray-Curtis distance between two 1-D sparse matrices and returns a distance value in the range [0,1]. Implementation follows

http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.braycurtis.html

#### Equation
```
sum(|a-b|)/sum(|a+b|)
```
#### Usage

| Parameter |   |
|:---------:|:--|
| a | 1-D sparse matrix \(1,N\) or \(N,1\) |
| b | 1-D sparse matrix \(1,N\) or \(N,1\) |
| **Returns** |  |
|float| Bray-Curtis distance [0,1] |

```python
>>> import scipy.sparse as ss
>>> a = [0,2,0,1]
>>> b = [1,0,3,0]
>>> c = [0,2,3,1]
>>> sparseDist.braycurtis(ss.csr_matrix([a]), ss.csr_matrix([b]))
1.0
>>> sparseDist.braycurtis(ss.csr_matrix([c]), ss.csr_matrix([b]))
0.40000000000000002
```
