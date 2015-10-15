# author : Etienne THIERY

from matgen import *
import random
import numpy

def test_symmetricPositiveDefinite():
    for i in range(10):
        print(".", end="", flush=True)
        size = random.randint(400, 500)
        maxVal = random.randint(0, 1000)
        M = symmetricPositiveDefinite(size, maxVal)
        if not (isSymmetric(M) and isDefinitePositive(M)):
            return False
    return True

def test_symmetricSparsePositiveDefinite():
    for i in range(10):
        print(".", end="", flush=True)
        size = random.randint(400, 500)
        maxVal = random.randint(0, 1000)
        nbZeros = random.randint(0, size*(size-1))
        M = symmetricSparsePositiveDefinite(size, nbZeros, maxVal)
        if not (isSymmetric(M) and isDefinitePositive(M) and abs(numberOfZeros(M)-nbZeros) <= 1):
            return False
    return True

def numberOfZeros(M):
    count = 0
    for line in M:
            for coeff in line:
                if coeff == 0:
                    count+=1
    return count

def printTest(test_func):
    print("Testing " + test_func.__name__[5:] + " : ", end="", flush=True)
    print(("" if test_func() else "un") + "expected behaviour", flush=True)

printTest(test_symmetricPositiveDefinite)
printTest(test_symmetricSparsePositiveDefinite)
