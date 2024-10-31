import numpy as np

### Part One

# compute the SVD given an NxM matrix
def my_svd(A):
    A_inv = None
    A_AT = np.dot(A, np.transpose(A))      # for U matrix
    AT_A = np.dot(np.transpose(A), A)      # for V matrix

    # compute eigenvalues and eigenvectors for both 
    eigenval_u, eigenvec_u = np.linalg.eig(A_AT)
    eigenval_v, eigenvec_v = np.linalg.eig(AT_A)

    # sort eigenvalues in descending order 
    ncols_u = np.argsort(eigenval_u)[::-1]
    ncols_v = np.argsort(eigenval_v)[::-1]

    U = eigenvec_u[:, ncols_u]
    V = eigenvec_v[:, ncols_v].T

    # compute singular values (sqaure root of eigenvalues)
    singular_vals = np.sqrt(eigenval_u[ncols_u]) 

    print("Eigenvalues:", eigenval_v[ncols_v])
    print("Singular values:", singular_vals)

    # make sigma matrix  
    S = np.zeros(A.shape)
    np.fill_diagonal(S, singular_vals)

    # calculate condiiton number 
    if min(singular_vals) != 0:
        condition_num = max(singular_vals)/min(singular_vals)
    else:
        condition_num = np.inf 

    # compute inverse of A if square and invertible 
    if A.shape[0] == A.shape[1] and np.all(singular_vals):
        A_inv = np.dot(np.dot(V, np.linalg.inv(S)), U.T)
    else:
        print("Inverse does not exist.")

    return U, S, V, condition_num, A_inv, singular_vals

### Part Two

def spring_mass_sys(springs_num, masses_num, spring_const, masses, boundary_condition):
# solve the force balance first (f = A^Tw = A^TCAu = Ku)
    m,n = springs_num, masses_num # m equations, n unknowns 
    A = np.zeros((m,n))

    # fill A based on the boundary condition 
    if boundary_condition == 'fixed-fixed':
        A[0, 0] = 1
        for i in range(1, m-1):
            A[i, i - 1] = -1
            A[i, i] = 1
        A[m - 1, n - 2] = -1  # last row, second-to-last column
        A[m - 1, n - 1] = 1    # last row, last column

    elif boundary_condition == 'fixed-free':
        A[0, 0] = 1
        for i in range(1, m):
            A[i, i - 1] = -1
            A[i, i] = 1

    elif boundary_condition == 'free-free':
        for i in range(m):
            if i > 0:
                A[i, i - 1] = -1
            A[i, i] = 1
    else:
        raise ValueError("Invalid boundary condition")
    
    # calculate stiffness matrix (K = A^TCA)
    C = np.diag(spring_const)
    K = A.T @ C @ A 
    f = np.array(masses)*9.81      # calculate force = (mass)(gravity) 

    # solve for displacements 
    U, S, V, condition_num, K_inv, singular_vals = my_svd(K)
    print("L2-condition number of K:", condition_num)

    u = K_inv @ f if K_inv is not None else None 
    
    # back calculate elongations (e) and internal forces (w)
    if u is not None:
        e = A @ u
        w = C @ e
    else:
        raise ValueError("The system matrix K is singular and cannot be inverted.")

    return u, w, e


def main():
    # ask for user input 
    print('Enter boundary condition type: "fixed-free", "fixed-fixed", or "free-free"')
    boundary_condition = input("Boundary condition: ").strip()
    springs_num = int(input("Enter the number of springs: "))
    masses_num = int(input("Enter the number of masses: "))

    # get spring constants and masses from user
    spring_const = []
    for i in range(springs_num):
        spring_const.append(float(input(f"Enter spring constant k{i+1}: ")))
    
    masses = []
    for i in range(masses_num):
        masses.append(float(input(f"Enter mass m{i+1}: ")))

    # convert lists to numpy arrays
    spring_const = np.array(spring_const)
    masses = np.array(masses)
    
    u, w, e = spring_mass_sys(springs_num, masses_num, spring_const, masses, boundary_condition)
    
    print("\nEquilibrium Displacements (u):\n", u)
    print("Internal Stresses (w):\n", w)
    print("Elongations (e):\n", e)

if __name__ == "__main__":
    main()