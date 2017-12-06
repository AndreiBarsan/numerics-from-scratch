"""From the pysfm project."""

from numpy import *
from numpy.linalg import *
import numpy as np

import numpy_test

################################################################################
def axis(i, ndim):
    x = zeros(ndim)
    x[i] = 1.
    return x

################################################################################
def numeric_derivative(f, x0, h=1e-8):
    assert isscalar(x0), 'numeric_derivative can only deal with scalar inputs'
    return (f(x0 + h) - f(x0 - h)) / (2. * h)

################################################################################
def numeric_jacobian(f, x0, h=1e-8):
    x0 = atleast_1d(x0)
    f0 = atleast_1d(f(x0))

    assert ndim(x0) <= 1
    assert ndim(f0) <= 1
    xlen = x0.size
    flen = f0.size

    J = empty((flen, xlen))
    for i in range(xlen):
        ei = h * axis(i,xlen)
        J[:,i] = (f(x0+ei) - f(x0-ei)) / (2. * h)

    return J

################################################################################
def numeric_hessian(f, x0, h=1e-6):
    x0 = atleast_1d(asarray(x0, float128))
    f0 = atleast_1d(asarray(f(x0), float128))

    assert ndim(x0) <= 1
    assert ndim(f0) <= 1
    xlen = x0.size
    flen = f0.size

    H = empty((flen, xlen, xlen), float128)
    for i in range(xlen):
        ei = h * axis(i,xlen)

        # Compute on-diagonal terms
        H[:,i,i] = ( f(x0 + ei) - 2.*f0 + f(x0 - ei) )
        H[:,i,i] /= h*h

        # Comptue off-diagonal terms
        for j in range(i):
            ej = h * axis(j, xlen)
            A = asarray(f(x0+ei+ej), float128)
            B = asarray(f(x0+ei-ej), float128)
            C = asarray(f(x0-ei+ej), float128)
            D = asarray(f(x0-ei-ej), float128)
            H[:,i,j] = (A + D - B - C) / (4.*h*h)
            H[:,j,i] = H[:,i,j]

    return H

################################################################################
def check_jacobian(f, Jf, x0):
    # if Jf is a function then just evaluate it once
    if callable(Jf):
        Jf = Jf(x0)

    # Upgrade Jf to 2 dimensions because that is what numeric_jacobian returns
    Jf = atleast_2d(Jf)

    # Compute jaobians numerically
    x0 = asarray(x0)
    Jf_numeric = numeric_jacobian(f, x0, 1e-8)

    # Check shapes
    if Jf.shape != Jf_numeric.shape:
        print('Error: shape mismatch')
        print('  analytic jacobian was '+str(Jf.shape))
        print('  numeric jacobian was '+str(Jf_numeric.shape))
        return False

    # Check values
    Jf_abserr = np.abs(Jf - Jf_numeric)
    mask = np.abs(Jf_numeric) > 1.
    Jf_abserr[mask] /= np.abs(Jf_numeric[mask])

    abserr = np.max(Jf_abserr)  # threshold should be independent of matrix size
    if abserr < 1e-5:
        print('JACOBIAN IS CORRECT')
    else:
        if Jf.size < 100:
            print('Numeric jacobian:')
            print(Jf_numeric)
            print('Analytic jacobian:')
            print(Jf)
            print('Residual of jacobian:')
            print(Jf_abserr)

        if Jf.size > 10:
            print('Numeric Jacobian (sparsity pattern):')
            numpy_test.spy(Jf_numeric)
            print('Analytic Jacobian (sparsity pattern):')
            numpy_test.spy(Jf)
            print('Residual of Jacobian (sparsity pattern):')
            numpy_test.spy(Jf_abserr, 1e-6)

        print('Max error in Jacobian:')
        print(abserr)
        i = unravel_index(argmax(Jf_abserr), Jf_abserr.shape)
        print('  (%f vs %f at %d,%d)' % (Jf[i], Jf_numeric[i], i[0], i[1]))
        print('JACOBIAN IS WRONG')

    return abserr, Jf_abserr

################################################################################
# Check the validity of an update vector found during gradient
# descent. The minimum requirements are that it should have a positive
# dot product with the gradient: if this is violated, the descent
# direction is doomed. Unfortunately, a random vector will also pass
# this test with 50% probability, so we do additional heuristic tests.
def check_descent_direction(update, f, x0, h=1e-8):
    f0 = squeeze(f(x0))
    assert ndim(f0) == 0, 'function must return a scalar'

    # check that the update has the right shape
    assert shape(update) == shape(x0), 'update had shape %s' % shape(update)

    # compute direction of steepest descent at x0
    # note that this is the *negative* gradient
    steepest_descent = -squeeze(numeric_jacobian(f, x0, h))
    assert ndim(steepest_descent) == 1

    # update *must* have a positive dot product with g
    dp = dot(steepest_descent,update) / (norm(steepest_descent)*norm(update))
    assert dp > 0, \
        '''update was not a descent direction (update * gradient = %f)
gradient=%s\nupdate=%s''' % (dp, str(steepest_descent), str(update))

    # An angle greater than 45 degrees probably indicates something went wrong
#    if dp < sqrt(.5):
#        theta = rad2deg(arccos(dp))
#        print '  Warning: update was far from the descent direction (dp=%f, theta=%f)' % (dp,theta)

    # Compute the gradient in the direction of steepest descent
    g_steepest = (f(x0 + steepest_descent*h) - f0) / h
    assert g_steepest < 1e-8, \
        'A small step in the direction of steepest descent increases the cost. (delta cost=%f)' \
        % g_steepest

    # Compute the gradient in the direction of update
    g_update = (f(x0 + update*h) - f0) / h
#    if g_update > 1e-8:
#        print '  Warning: epsilon*update increases the cost. (delta cost=%f)' % g_update


