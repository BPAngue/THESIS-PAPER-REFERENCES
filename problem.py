import numpy as np

# The problem is defined in this file

def sphere(x): # x is expected to be a numpy array
    z = np.sum(np.power(x,2)) # gets the sum of the squares of all numpy array elements in x
    return z


def problem(x):

    z = sphere(x)

    return z