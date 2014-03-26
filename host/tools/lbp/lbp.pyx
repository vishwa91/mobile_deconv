import numpy as cnp
cimport numpy as cnp
cimport cython

DTYPE = cnp.int
ctypedef cnp.int_t DTYPE_t

@cython.boundscheck(False)
cpdef inline lbp(cnp.ndarray[DTYPE_t, ndim=2] im):
	'''Return the LBP map of an image'''
	cdef DTYPE_t xdim, ydim
	xdim = im.shape[0]; ydim = im.shape[1]
	cdef cnp.ndarray[DTYPE_t, ndim=2] patch = cnp.zeros((3,3), dtype=DTYPE)
	cdef cnp.ndarray[DTYPE_t, ndim=2] imlbp = cnp.zeros((xdim, ydim), dtype=DTYPE)
	for x in range(1, xdim-1):
		for y in range(1, ydim-1):
			patch = im[x-1:x+2, y-1:y+2] - im[x,y]
			u, v = cnp.where(patch > 0)
			patch[:,:] = 0; patch[u,v] = 1
			imlbp[x, y] = (patch[1,0]*1 + patch[0,1]*2 + patch[0,1]*4 +
						   patch[0,2]*8 + patch[1,2]*16 + patch[2,2]*32 +
						   patch[2,1]*64 + patch[2,0]*128)
	return imlbp