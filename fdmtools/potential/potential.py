import numpy as np
import scipy as sp
import scipy.optimize as scopt
import scipy.interpolate as scinterp
import scipy.integrate as scinteg
import scipy.special as scsp
import scipy.stats as scstat
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

def test_func(x):
	y = x**2 * np.pi
	return y