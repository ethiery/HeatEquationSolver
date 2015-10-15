import matgen
import cholesky
import numpy as np
import scipy.linalg as splg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.figure as fg

def heatEquationMatrix(n):
    ''' Returns a tridiagonal n^2*n^2 matrix A to solve the heat equation 
    in a discrete way by solving the linear system Ax = b, where b is a n^2 
    vector such as b[n*x+y] is the heat flux density in the point (x,y)
    and h is the distance on both axis between 2 consecutive points'''
    N = n*n

    A = np.diag([1]*(N-n), -n) + np.diag([1]*(N-n), n)
    sub = -4*np.eye(n) + np.diag([1]*(n-1), 1) + np.diag([1]*(n-1), -1) 
    for i in range(0, n):
        A[i*n:(i+1)*n,i*n:(i+1)*n] = sub[:,:]
    return A


def matToVect(M):
    ''' M shoud be a square matrix of size nxn
    Returns a vector containing the values in M, readfrom the left hand column
    to the right hand column, from top to bottom, and stop from top to bottom'''
    n = M.shape[0]
    return M.transpose().reshape((n*n,))

def vectToMat(v):
    ''' v shoud be a vector of length nxn
    Returns a matrix containing the values in v, read from top to bottom and then 
    stored from the left hand column to the right hand column, from top to bottom'''
    n = np.sqrt(v.shape[0])
    return v.reshape((n,n)) 

def solveHeatEquation(heatFlux, h, conductivity):
    n = heatFlux.shape[0]
    heatFlux = matToVect(heatFlux)*h*h/conductivity
    M = heatEquationMatrix(n)
    T = np.linalg.cholesky(-M)
    y = splg.solve_triangular(T, heatFlux, lower=True, check_finite=False)
    x = splg.solve_triangular(-T.transpose(), y, check_finite=False)
    return vectToMat(x)

def printHeatSolution(sol):
    plt.imshow(sol.transpose(), interpolation='bilinear', cmap=cm.jet_r)
    plt.colorbar()
    plt.show()


# size = 50
# h = 0.01
# airConductivity = 0.025
# heatFlux = np.zeros((size,size))
# # ext[size/2][size/2] = 200
# heatFlux[10][10] = 100
# heatFlux[10][30] = 150
# heatFlux[30][10] = 200
# heatFlux[30][30] = 250

# printHeatSolution(solveHeatEquation(heatFlux, h, airConductivity))
