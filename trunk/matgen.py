# A little library for the generation and attribute checking of
# matrix of integers
# author : Etienne THIERY

from numpy import *

def symmetricPositiveDefinite(n, maxValue= 1):
    ''' Generates a n x n random symmetric, positive-definite matrix.
    The optionnal maxValue argument can be used to specify a maximum
    absolute value for extradiagonal coefficients.
    Diagonal coefficient will be inferior to 22 times maxValue in
    absolute value.

    Runs in O(n^2)'''
    
    # To generate such a matrix we use the fact that a symmetric 
    # diagonnaly dominant matrix is symmetric positive definite

    # We first generate a random matrix
    # with coefficients between -maxValue and +maxValue 
    A = random.random_integers(-maxValue, maxValue, (n, n))

    # Then by adding to this matrix its transpose, we obtain 
    # a symmetric matrix
    A = A + A.transpose()

    # Finally we make sure it is strictly diagonnaly dominant by
    # adding 2*n*maxValue times the identity matrix
    A += 2*n*maxValue*eye(n)
    return A

def symmetricSparsePositiveDefinite(n, nbZeros, maxValue= 1):
    ''' Generates a n x n random symmetric, positive-definite matrix.
    with around nbZeros null coefficients (more precisely nbZeros+-1)
    nbZeros must be between 0 and n*(n-1)
    The optionnal maxValue argument can be used to specify a maximum
    absolute value for extradiagonal coefficients.
    Diagonal coefficient will be inferior to 11 times maxValue in
    absolute value.

    Runs in O(n^2)'''
    
    # The algorithm is the same as in symmetricPositiveDefinite
    # except that the matrix generated in the beginning is 
    # sparse symmetric

    A = zeros((n,n))
    currentNbZeros = n*(n-1)
    while currentNbZeros > nbZeros:
        i, j = random.randint(n, size=2)
        if i != j and A[i,j] == 0:
            while A[i,j] == 0:
                A[i,j] = A[j,i] = random.randint(-maxValue, maxValue+1)
            currentNbZeros -= 2 

    # Then we make sure it is strictly diagonnaly dominant by
    # adding n*maxValue times the identity matrix
    A += n*maxValue*eye(n)
    return A
    
def isSymmetric(M):
    ''' Returns true if and only if M is symmetric'''
    return array_equal(M, M.transpose())

def isDefinitePositive(M):
    ''' Returns true if and only if M is definite positive'''
    # using the fact that if all its eigenvalues are positive, 
    # M is definite positive
    # be careful, as eigvals use numerical methods, some eigenvalues
    # which are in reality equal to zero can be found negative
    eps = 1e-5
    for ev in linalg.eigvals(M):
        if ev <= 0-eps:
            return False
    return True
