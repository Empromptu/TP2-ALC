import numpy as np
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt

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


def metodoPotencia(A, num_iteraciones):
    """
    Calcula el autovalor dominante de una matriz A usando el Método de la Potencia.

    Args:
        A (ndarray): La Matriz a calcularle el autovalor 
        num_iteraciones (int): Número máximo de iteraciones para aproximar el autovalor.

    Returns:
        tuple: Una tupla que contiene el promedio, la desviación estándar, el máximo de los autovalores 
               y una lista de los autovalores calculados en cada iteración.
    """
    tolerancia = 1e-6
    autovalores = []
    n = A.shape[0]
    x0 = np.random.rand(n)
    x0 = x0 / np.linalg.norm(x0)

    for _ in range(num_iteraciones):
        x1 = A @ x0
        x1 = x1 / np.linalg.norm(x1)

        autovalor = (x0.T @ A @ x0) / (x0.T @ x0) 
        autovalores.append(autovalor)
        if np.linalg.norm(x1 - x0) < tolerancia:
            break
        x0 = x1

    promedio = np.mean(autovalores)
    desviacion_estandar = np.std(autovalores)
    maxaval = max(autovalores)

    return promedio, desviacion_estandar, maxaval, autovalores

def metodoPotenciaRecursivo(C, k, epsilon=1e-6, autovalores=None, autovectores=None):
    """
    Calcula los primeros k autovalores y autovectores de una matriz de covarianza C 
    usando el Método de la Potencia Recursivo, basado en la consigna 7.

    Args:
        C (ndarray): Matriz de Covarianza
        k (int): Número de autovalores y autovectores a calcular.
        epsilon (float, optional): Tolerancia para la convergencia. Default es 1e-6.
        autovalores (list, optional): Lista para almacenar los autovalores encontrados. Vacia en el Caso Base
        autovectores (list, optional): Lista para almacenar los autovectores encontrados. Vacia en el Caso Base

    Returns:
        tuple: Una tupla que contiene dos listas: autovalores y autovectores.
    """
    if autovalores is None:
        autovalores = []
    if autovectores is None:
        autovectores = []
    # Caso base
    if len(autovalores) == k:
        return autovalores, autovectores

    # Generamos un vector aleatorio normalizado
    n = C.shape[0]
    x = np.random.rand(n)
    x = x / np.linalg.norm(x)

    # Iteramos
    x_next = C @ x
    x_next = x_next / np.linalg.norm(x_next)

    while np.linalg.norm(x_next - x) >= (1 - epsilon):
        x = x_next
        x_next = C @ x
        x_next = x_next / np.linalg.norm(x_next)

    # Calculamos el autovalor usando el cociente de Rayleigh y lo guardamos con el autovector aproximado
    autovalor = (x.T @ C @ x) / (x.T @ x)
    autovalores.append(autovalor)
    autovectores.append(x)  # Aquí se guarda el autovector correspondiente

    # Construimos la nueva matriz C' = C - autovalor * x * x^T
    Cprima = C - autovalor * np.outer(x, x)

    return metodoPotenciaRecursivo(Cprima, k, epsilon, autovalores, autovectores)

def GraficoProyeccion(proyeccion, nombre: str, color):
    """
    Grafica la proyeccion de cada sector en los 2 autovectores principales

    Args:
        proyeccion: Filas de la Matriz proyectadas
        nombre (str): Nombre de la Matriz para el titulo
        color: Color de los puntos de la proyeccion
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(proyeccion[:, 0], proyeccion[:, 1], color=color, marker='o')
    plt.title(f"Proyección de filas de {nombre} usando los autovectores principales")
    for i in range(proyeccion.shape[0]):
        plt.text(proyeccion[i, 0], proyeccion[i, 1], str(i), fontsize=9, ha='right')
    plt.xlabel("Primera Componente Principal")
    plt.ylabel("Segunda Componente Principal")
    plt.grid()
    plt.show()

def GraficarFilas (Matriz, indice, titulo):
    """
    Grafico en barras de las filas de la Matriz

    Args:
        Matriz: Matriz a graficar
        indice: Fila de la Matriz graficar
        Titulo: Titulo del grafico
    """
    # Crear figura y ejes
    fig, axs = plt.subplots(1, 1, figsize=(6, 2.5))
    # Graficar la diferencia en barras
    axs.bar(range(len(Matriz[indice])), Matriz[indice], color='orange')
    axs.set_title(titulo)
    axs.set_xlabel('Sectores')
    axs.set_ylabel('Incidencia en Producción')

    plt.tight_layout()
    plt.show()