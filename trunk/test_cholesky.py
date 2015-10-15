# author : Etienne THIERY

import numpy, random
import cholesky, matgen
import matplotlib.pyplot as plt
import timeit

def test_cholesky():
    random.seed()
    for i in range(10):
        print(".", end="", flush= True)
        size = random.randint(100, 200)
        M = matgen.symmetricPositiveDefinite(size)
        T = cholesky.completeCholesky(M)
        if not numpy.allclose(M, numpy.dot(T, T.transpose())):
            return False
    return True

def testIncompleteCholeskyPrecision():
    size = 100
    nbOfPoints = 100 
    x, y = [], []
    for i in numpy.linspace(0, size*(size-1), nbOfPoints):
        x.append(i/(size*size))
        M = matgen.symmetricSparsePositiveDefinite(size, i)
        complete = cholesky.completeCholesky(M)
        incomplete = cholesky.incompleteCholesky(M)
        y.append(numpy.linalg.norm(complete-incomplete)/numpy.linalg.norm(complete))
    plt.plot(x, y, marker='o')
    plt.title("Relative error made by Cholesky incomplete factorization, in function of matrix density\n")
    plt.ylabel("relative error on the norm of the matrix obtained")
    plt.xlabel("density of the initial 100*100 matrix")
    plt.show()

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

def compareCompleteVSIncomplete():
    size = 100
    nbOfPoints = 50
    nbValuesForAverage = 4
    x, yComplete, yIncomplete = [], [], []

    for i in numpy.linspace(0, size*(size-1), nbOfPoints):
        # Computes matrix density
        x.append((size*size-i)/(size*size))
        # Computes average execution times on 3 calls on different matrix
        completeTime, incompleteTime = 0, 0
        for j in range(nbValuesForAverage):    
            M = matgen.symmetricSparsePositiveDefinite(size, i)
            wrapped1 = wrapper(cholesky.completeCholesky, M)
            completeTime += timeit.timeit(wrapped1, number=1)
            wrapped2 = wrapper(cholesky.incompleteCholesky, M)
            incompleteTime += timeit.timeit(wrapped2, number=1)
        yComplete.append(completeTime/nbValuesForAverage)
        yIncomplete.append(incompleteTime/nbValuesForAverage)

    p1 = plt.plot(x, yComplete, 'b', marker='o')
    p2 = plt.plot(x, yIncomplete, 'g', marker='o')
    plt.title("Average execution time for the Cholesky factorization in function of matrix density\n")
    plt.legend(["New custom Complete factorization", "Custom incomplete factorization"], loc=4)
    plt.ylabel("Execution time (s)")
    plt.xlabel("density of the initial 100*100 matrix")
    plt.show()

def compareCholeskyCustomVSNumpy():
    maxSize = 500
    nbOfPoints = 200
    nbValuesForAverage = 2
    x, yCustom, yNumpy = [], [], []

    for size in [round(i) for i in numpy.linspace(5, maxSize, nbOfPoints)]:
        # Computes matrix density
        x.append(size)
        # Computes average execution times on nbValuesForAverage calls 
        # on different matrix
        customTime, numpyTime = 0, 0
        for j in range(nbValuesForAverage):
            M = matgen.symmetricPositiveDefinite(size)    
            wrapped1 = wrapper(cholesky.completeCholesky, M)
            customTime += timeit.timeit(wrapped1, number=1)
            wrapped2 = wrapper(numpy.linalg.cholesky, M)
            numpyTime += timeit.timeit(wrapped2, number=1)
        yCustom.append(customTime/nbValuesForAverage)
        yNumpy.append(numpyTime/nbValuesForAverage)

    p1 = plt.plot(x, yCustom, 'b', marker='o')
    p2 = plt.plot(x, yNumpy, 'g', marker='o')
    plt.title("Average execution time for the Cholesky factorization (custom and Numpy's) in function of size\n")
    plt.legend(["New custom Cholesky factorization", "Numpy Cholesky factorization"], loc=2)
    plt.ylabel("Execution time (s)")
    plt.xlabel("size")
    plt.show()


def printTest(test_func):
    print("Testing " + test_func.__name__[5:] + " : ", end="", flush=True)
    print(("" if test_func() else "un") + "expected behaviour", flush=True)

printTest(test_cholesky)
print("Testing the precision of my custom Incomplete Cholesky Factorization")
testIncompleteCholeskyPrecision()
print("Comparing the execution times of my custom Cholesky Factorizations")
compareCompleteVSIncomplete()
print("Comparing the execution times of my custom Cholkesy Factorization and Numpy's")
compareCholeskyCustomVSNumpy()