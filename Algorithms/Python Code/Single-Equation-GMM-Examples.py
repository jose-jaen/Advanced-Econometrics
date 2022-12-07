# Required libraries
import numpy as np
from numpy.random import normal
from numpy.linalg import matrix_rank

# Create a 3x3 matrix
A = np.array([
    [1, 2, 3],
    [1, 6, 17],
    [1, 8, 23]
])

print(matrix_rank(A)) # rank(A) = 3

# Generate random variable
random = normal(0, 1, 3)

# Add the new variable to matrix A
A = np.vstack([A, random])

print(matrix_rank(A)) # rank(A) = 3