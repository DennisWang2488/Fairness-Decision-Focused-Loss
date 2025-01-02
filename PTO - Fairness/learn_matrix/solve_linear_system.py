import numpy as np

def generate_matrix(n):
    """
    Generates a random nxn matrix.
    
    Parameters:
    n (int): The dimension of the matrix to be generated.
    
    Returns:
    np.ndarray: A random nxn matrix.
    """
    return np.random.rand(n, n)

def solve_system_by_inversion(A, B):
    """
    Solves the system Ax = B by matrix inversion.
    
    Parameters:
    A (np.ndarray): The coefficient matrix.
    B (np.ndarray): The right-hand side matrix or vector.
    
    Returns:
    np.ndarray: The solution vector or matrix x.
    """
    # Check if the matrix A is invertible
    if np.linalg.det(A) == 0:
        raise ValueError("Matrix A is not invertible.")
    
    # Calculate the inverse of A
    A_inv = np.linalg.inv(A)
    
    # Calculate the solution x
    x = np.dot(A_inv, B)
    
    return x

# Example usage:
n = 3
A = generate_matrix(n)
B = np.random.rand(n, 1)

print("Matrix A:")
print(A)

print("\nMatrix B:")
print(B)

try:
    x = solve_system_by_inversion(A, B)
    print("\nSolution x:")
    print(x)
except ValueError as e:
    print(e)
