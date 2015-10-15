# A library for running Cholesky factorization and Cholesky incomplete
# factorization
# author : Etienne THIERY
import trunk.matgen
from numpy import *

def oldCompleteCholesky(M):
    ''' Computes the cholesky factorization of a matrix M, i.e the
    lower triangular matrix T such as M = T . T.transpose
    M should be a symmetric positive definite matrix
    runs in O(n^3/3) where M is a matrix of size nxn'''
    n = M.shape[0]
    T = zeros((n,n))
    col = 0
    while col < n:
        T[col][col] = M[col][col]
        row = 0
        while row < col: 
            T[col][col] -= T[col][row]*T[col][row]
            row+=1
        T[col][col] = sqrt(T[col][col])
        row+=1
        while row < n:
            T[row][col] = M[row][col]
            k = 0
            while k < col:
                T[row,col] -= T[col,k]*T[row,k]
                k+=1
            T[row,col] /= T[col,col]
            row+=1
        col+=1
    
    return T

def completeCholesky(M):
    ''' Computes the cholesky factorization of a matrix M, i.e the
    lower triangular matrix T such as M = T . T.transpose
    M should be a symmetric positive definite matrix
    runs in O(n^3/3) where M is a matrix of size nxn'''
    n = M.shape[0]
    T = zeros((n,n))
    for col in range(n):
        T[col,col] = sqrt(M[col,col] - dot(T[col,:col], T[col,:col]))
        T[col+1:n,col] = (M[col+1:n,col] - dot(T[col+1:n,:col], (T[col,:col]).transpose())) / T[col,col]
        
    return T

def oldIncompleteCholesky(M):
    '''Only performs an incomplete cholesky factorization
    which result in only an approximation of the result, but is faster '''

    n = M.shape[0]
    T = zeros((n,n))
    col = 0
    while col < n:
        T[col][col] = M[col][col]
        row = 0
        if M[col][col] != 0:
            while row < col: 
                T[col][col] -= T[col][row]*T[col][row]
                row+=1
            T[col][col] = sqrt(T[col][col])
            row+=1
        else:
            row = col+1

        while row < n:
            if M[row][col] != 0:
                T[row][col] = M[row][col]
                k = 0
                while k < col:
                    T[row,col] -= T[col,k]*T[row,k]
                    k+=1
                T[row,col] /= T[col,col]
            row+=1
        col+=1    
    return T


def incompleteCholesky(M):
    '''Only performs an incomplete cholesky factorization
    which result in only an approximation of the result, but is faster '''
    n = M.shape[0]
    T = zeros((n,n))
    for col in range(n):
        T[col,col] = sqrt(M[col,col] - dot(T[col,:col], T[col,:col]))
        for row in range(col+1, n):
            if M[row,col] == 0:
                T[row,col] = 0
            else:
                T[row,col] = (M[row,col] - dot(T[row,:col], (T[col,:col]).transpose())) / T[col,col]      
    return T
