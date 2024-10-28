import numpy as np
from scipy.linalg import solve_triangular

def calcularLU(A):
    """
    Calcular la factorización LU de una matriz.

    Args:
        A (numpy.ndarray): Matriz cuadrada que se desea factorizar.

    Returns:
        L (numpy.ndarray): Matriz triangular inferior L.
        U (numpy.ndarray): Matriz triangular superior U.
        P (numpy.ndarray): Matriz de permutación P.
    """
    m = A.shape[0]
    n = A.shape[1]
    Ac = A.copy()
    P = np.eye(n)
    
    if m != n:
        print('Matriz no cuadrada')
        return

    L = np.eye(n) 
        
    for i in range(0, n):   
        max_row = np.argmax(np.abs(Ac[i:n, i])) + i
        if i != max_row:
            # Intercambiamos las filas en Ac
            Ac[[i, max_row], :] = Ac[[max_row, i], :]
            
            # Intercambiamos las filas en P
            P[[i, max_row], :] = P[[max_row, i], :]
            
            # Intercambiamos las filas en L hasta la columna i (excluyendo la diagonal)
            if i > 0:
                L[[i, max_row], :i] = L[[max_row, i], :i]
            
        for j in range(i+1, n):
            piv = Ac[j][i] / Ac[i][i]
            Ac[j] = Ac[j] - piv * Ac[i]
            L[j][i] = piv

    U = Ac
    return L, U, P


def inversaLU(L, U, P):
    """
    Calculo de la inversa de una matriz a partir de su factorización LU.

    Args:
        L (numpy.ndarray): Matriz triangular inferior L.
        U (numpy.ndarray): Matriz triangular superior U.
        P (numpy.ndarray): Matriz de permutación P.

    Returns:
        numpy.ndarray: Matriz inversa de la matriz original A que fue factorizada en L, U y P.
    """
    n = L.shape[0]
    Inv = np.zeros((n, 0)) 

    if P is None:
        P = np.eye(n)  # Si no se proporciona P, se usa la identidad
    for i in range(n):
        b = P[:, i]  
        x = solve_triangular(L, b, lower=True)
        y = solve_triangular(U, x, lower=False)
        Inv = np.column_stack((Inv, y))  # Agregamos la columna de la inversa

    return Inv
