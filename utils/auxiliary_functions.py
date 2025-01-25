import numpy as np
from sympy import symbols,Poly,sqrt,integrate,legendre,ordered

def construct_vector(n_t, n_s, f):
    """
    Construct a vector of size n_t * n_s * 2 with elements f(t_i, x_p, j).
    
    Parameters:
    - n_t: Number of time steps (t_i ranges from 0 to n_t-1).
    - n_s: Number of spatial points (x_p ranges from 0 to n_s-1).
    - f: A function that takes (t, x, j) as arguments and returns a value.

    Returns:
    - A 1D NumPy array of size n_t * n_s * 2.
    """
    # Create indices for t, x, and j
    t_indices = np.arange(n_t)  # Time indices (0, ..., n_t-1)
    x_indices = np.arange(n_s)  # Spatial indices (0, ..., n_s-1)
    j_indices = np.array([0, 1])  # j = 0 or 1

    # Use broadcasting to create a grid of (t, x, j)
    t_grid, x_grid, j_grid = np.meshgrid(t_indices, x_indices, j_indices, indexing='ij')

    # Apply the function f to each combination of (t, x, j)
    vector = f(t_grid, x_grid, j_grid)

    # Flatten the result into a 1D array
    return vector.ravel()

def generate_invertible_matrix(d_j, d_t, d_s, random_state=None):
    """
    Generate a random invertible matrix of size (d_j * d_t * d_s, d_j * d_t * d_s)
    that is not the identity matrix.

    Parameters:
    - d_j: First dimension multiplier
    - d_t: Second dimension multiplier
    - d_s: Third dimension multiplier
    - random_state: Optional random seed for reproducibility

    Returns:
    - A 2D NumPy array representing an invertible matrix.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    size = d_j * d_t * d_s  # Total size of the matrix
    
    while True:
        # Generate a random matrix
        matrix = np.random.rand(size, size)
        
        # Ensure it's not too close to the identity
        if np.allclose(matrix, np.eye(size)):
            continue
        
        # Check if the determinant is non-zero (invertibility condition)
        if np.linalg.det(matrix) != 0:
            return matrix



# def compute_basis_P(N):
#     t, x, j = symbols('t x j')
#     P=np.zeros(N)
    
#     polynomials = set()  # Using a set to avoid duplicates

#     # Generate all combinations of powers for x, t, and j
#     for p in range(N+1):
#         for k in range(N+1):
#             for s in range(2):
#                 # Create the term t^p * x^k * j^s
#                 term = t**p * x**k * j**s
#                 polynomials.add(term)

#     # Convert the set to a list and then to a numpy array
#     polynomials_list = list(polynomials)
#     normalized_polynomials = []

#     for poly in polynomials_list:
#         norm = sqrt(
#             sum(
#                 integrate(integrate(poly.subs(j, j_val)**2, (t, 0, 1)), (x, 0, 1))
#                 for j_val in [0, 1]
#             )
#         )
#         if norm != 0:
#             normalized_poly = poly / norm
#         else:
#             normalized_poly = poly 
#         normalized_polynomials.append(normalized_poly)

#     return np.array(normalized_polynomials)

def compute_basis_P(N):
    """
    Compute the 3D Legendre polynomial basis on [0,1]^2 x {0,1} up to degree N.

    Parameters:
        N (int): Maximum degree of the polynomials in each variable.

    Returns:
        list: List of unique 3D Legendre polynomials on [0,1]^2 x {0,1}.
    """
    # Define symbols for the variables
    x, t, j = symbols('x t j')

    # Function to map [0, 1] to [-1, 1]
    transform = lambda t: 2 * t - 1

    # Generate 1D Legendre polynomials transformed to [0, 1]
    legendre_1d = [legendre(n, transform(x)) for n in range(N + 1)]

    # Generate 2D basis as tensor product of 1D polynomials
    basis_2d = []
    for m in range(N + 1):
        for n in range(N + 1):
            poly_2d = legendre_1d[m] * legendre_1d[n].subs(x, t)
            basis_2d.append(poly_2d)

    # Add the third dimension (j = 0 or 1)
    basis_3d = []
    for poly in basis_2d:
        basis_3d.append(poly)  # Case j = 0
        basis_3d.append(poly * j)  # Case j = 1

    # Remove duplicates (if any)
    basis_3d = list(ordered(set(basis_3d)))

    return np.array(basis_3d)





def check_variables_in_polynomial(poly):
    """
    Checks if the polynomial contains the variables t, x, and j.

    Parameters:
    poly (sympy.Expr): The polynomial to check.

    Returns:
    numpy.ndarray: A boolean array of size 3 indicating the presence of t, x, and j.
    """
    # Define the symbols t, x, j
    t, x, j = symbols('t x j')
    # Check the presence of each symbol in the polynomial
    contains_t = t in poly.free_symbols
    contains_x = x in poly.free_symbols
    contains_j = j in poly.free_symbols
    # Return a numpy array of booleans
    return np.array([contains_t, contains_x, contains_j])

def evaluate_polynomial_conditionally(poly, bool_vector, t1=None, x1=None, j1=None):
    """
    Evaluates the polynomial using the provided values only for the variables present.

    Parameters:
    poly (sympy.Expr): The polynomial to evaluate.
    bool_vector (list or np.ndarray): Boolean vector indicating the presence of t, x, and j.
    t1 (float): Value for t if present.
    x1 (float): Value for x if present.
    j1 (float): Value for j if present.

    Returns:
    float: The evaluated polynomial if possible, or raises an exception if values are missing.
    """
    # Define symbols t, x, j
    t, x, j = symbols('t x j')

    # Create a dictionary for substitutions based on the boolean vector
    substitutions = {}
    if bool_vector[0] and t1 is not None:
        substitutions[t] = t1
    elif bool_vector[0]:
        raise ValueError("Missing value for t")

    if bool_vector[1] and x1 is not None:
        substitutions[x] = x1
    elif bool_vector[1]:
        raise ValueError("Missing value for x")

    if bool_vector[2] and j1 is not None:
        substitutions[j] = j1
    elif bool_vector[2]:
        raise ValueError("Missing value for j")

    # Evaluate the polynomial with the substitutions
    return poly.subs(substitutions)