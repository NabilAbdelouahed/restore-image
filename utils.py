import numpy as np
import cv2

def read_image_and_get_pixels(image_path):
	"""[pixel_table] = read_image_and_get_pixels(f): loads image in greyscale and returns a table of pixels
	% image_path : path to the image
	"""
	img = cv2.imread(image_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	pixel_table = img

	return pixel_table



def grad(f):
	"""[v,h] = grad(f): computes spatial gradients of image f
	% f : input image
	% v : vertical gradient
	% h : horizontal gradient
	"""
	K,L = np.shape(f)

	n = np.arange(K-1)
	m = np.arange(L-1)

	return f[1:K, :(L-1)] - f[:(K-1), :(L-1)], f[:(K-1), 1:L] - f[:(K-1), :(L-1)]



def div(v,h):
	"""
	f = div(v,h): computes the divergence (adjoint of the spatial
	 gradient)
	 v : input vertical gradient
	 h : inout horizontal gradient
	 f : output image
	"""

	K,L = np.shape(v)

	f = np.zeros( (K+1,L+1) )

	n = np.arange(K)
	m = np.arange(L)

	f[1:(K+1),:L] = v
	f[:K,:L]      = f[:K,:L] - v
	f[:K,1:(L+1)] = f[n,1:(L+1)] + h
	f[:K,:L]      = f[:K,:L] - h

	return f


def projballl2(xr, z, rho, indI):
	"""
	%projection of xr onto the l2 semi-ball || x(indI)-z(indI) || <= rho
	% xr : vector to be projected
	% z : center of the ball
	% rho : radius of the ball
	% indI : indices of the contrained components
	% p : computed projection
	"""

	no = np.linalg.norm( xr[indI] - z[indI] )

	p = np.copy(xr)

	if no > rho:
	    p[indI] = 0 # TO BE COMPLETED

	return p