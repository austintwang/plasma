from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import numpy as np

cimport numpy as np
from libc.math cimport sqrt
cimport cython

# @cython.boundscheck(False)
@cython.wraparound(False)
cdef bcksub(double[:, ::1] U, double[::1] b, np.intp_t N):
	cdef np.intp_t i = N
	cdef np.intp_t j
	cdef double v
	cdef double[::1] x = cython.view.array(shape=(N,), itemsize=sizeof(double), format="i")

	x[i] = b[i] / U[i,i]
	for i in range(N-2, -1, -1):
		v = b[i]
		for j in range(i+1, N):
			v -= x[j] * U[i,j]
		x[i] = v / U[i,i]

	return x

# @cython.boundscheck(False)
@cython.wraparound(False)
cdef fwdsub(double[:, ::1] L, double[::1] b, np.intp_t N):
	cdef np.intp_t i = 0
	cdef np.intp_t j
	cdef double v
	cdef double[::1] x = cython.view.array(shape=(N,), itemsize=sizeof(double), format="i")

	x[i] = b[i] / L[i,i]
	for i in range(1, N):
		v = b[i]
		for j in range(i):
			v -= x[j] * L[i,j]
		x[i] = v / U[i,i]

	return x

# @cython.boundscheck(False)
@cython.wraparound(False)
cdef r1update(double[:, ::1] U, double[::1] x, np.intp_t N):
	cdef np.intp_t i = 0
	cdef double w, v, r, c, s
	cdef double[::1] x_ = x.copy()
	double[:, ::1] U_ = cython.view.array(shape=(N,N), itemsize=sizeof(double), format="i")
	
	for i in range(N):
		w = U[i,i]
		v = x_[i]
		r = sqrt(w**2 + v**2)
		c = r / w
		s = v / w
		U_[i,i] = r
		for j in range(i+1, N):
			U_[i,j] = (U[i,j] + s * x_[j]) / c
			x_[j] = c * x_[j] - s * U_[i,j]

# @cython.boundscheck(False)
@cython.wraparound(False)
cdef r1downdate(double[:, ::1] U, double[::1] x, np.intp_t N):
	cdef np.intp_t i = 0
	cdef double w, v, r, c, s
	cdef double[::1] x_ = x.copy()
	double[:, ::1] U_ = cython.view.array(shape=(N,N), itemsize=sizeof(double), format="i")
	
	for i in range(N):
		w = U[i,i]
		v = x_[i]
		r = sqrt(w**2 - v**2)
		c = r / w
		s = v / w
		U_[i,i] = r
		for j in range(i+1, N):
			U_[i,j] = (U[i,j] - s * x_[j]) / c
			x_[j] = c * x_[j] - s * U_[i,j]

# @cython.boundscheck(False)
@cython.wraparound(False)
cdef add_entry(
	double[:, :] U, 
	double[:] a, 
	double b,
	double[:, ::1] V, 
	np.intp_t N,
	np.intp_t M
):
	cdef double[:, ::1] V_11 = U[:M,:M].copy()

	cdef double[:, ::1] V_11T = V_11.T.copy()
	cdef double[::1] A_12 = a[:M].copy()
	cdef double[::1] V_12 = fwdsub(V_11T, M)

	cdef double[:, ::1] V_13 = U[:M,M:].copy()

	cdef double p = b
	cdef np.intp_t i
	for i in range(M):
		p -= V_12[i] ** 2
	cdef double V_22 = sqrt(p)

	cdef np.intp_t j
	cdef np.intp_t k
	cdef np.intp_t D = N - M
	cdef double[:, ::1] V_13T = V_13.T.copy()
	cdef double[::1] V_23 = a[M:].copy()
	cdef double a
	for j in range(D):
		a = V_23[j]
		for k in range(M):
			a -= V_13T[j,k] * V_12[k]
		V_23[j] = a / V_22

	cdef double[:, ::1] U_33 = U[:M,:M].copy()
	cdef double[:, ::1] V_33 = r1downdate(U_33, V_23, D)



# @cython.boundscheck(False)
@cython.wraparound(False)
cdef del_entry():

# @cython.boundscheck(False)
@cython.wraparound(False)
def eval_bf():

