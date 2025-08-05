# strassen_pure_labels.py
# Algoritmo de Strassen sin umbral (recursión hasta 1×1)
# Gráfico con etiquetas de n sobre cada punto

import numpy as np
import time
import matplotlib.pyplot as plt
import sys

# ----------------------------------------------------------
# Funciones auxiliares
# ----------------------------------------------------------
def next_power_of_two(x: int) -> int:
    """Devuelve la mínima potencia de 2 mayor o igual a x."""
    return 1 << (x - 1).bit_length()

# ----------------------------------------------------------
# Strassen puro (recursivo hasta 1×1)
# ----------------------------------------------------------
def _strassen_recursive(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Multiplicación pura Strassen sin umbral (caso base 1×1)."""
    n = A.shape[0]
    if n == 1:
        return A * B

    mid = n // 2
    A11, A12 = A[:mid, :mid], A[:mid, mid:]
    A21, A22 = A[mid:, :mid], A[mid:, mid:]
    B11, B12 = B[:mid, :mid], B[:mid, mid:]
    B21, B22 = B[mid:, :mid], B[mid:, mid:]

    # 7 productos intermedios
    P1 = _strassen_recursive(A11 + A22, B11 + B22)
    P2 = _strassen_recursive(A21 + A22, B11)
    P3 = _strassen_recursive(A11, B12 - B22)
    P4 = _strassen_recursive(A22, B21 - B11)
    P5 = _strassen_recursive(A11 + A12, B22)
    P6 = _strassen_recursive(A21 - A11, B11 + B12)
    P7 = _strassen_recursive(A12 - A22, B21 + B22)

    # 4 bloques de la matriz resultado
    C11 = P1 + P4 - P5 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 - P2 + P3 + P6

    return np.block([[C11, C12],
                     [C21, C22]])

# ----------------------------------------------------------
# Wrapper público con padding para matrices rectangulares
# ----------------------------------------------------------
def strassen_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Multiplica A @ B usando exclusivamente el algoritmo de Strassen
    (recursión hasta 1×1).
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    m, k = A.shape
    k2, n = B.shape
    if k != k2:
        raise ValueError("Dimensiones incompatibles para la multiplicación.")

    max_dim = max(m, k, n)
    size = next_power_of_two(max_dim)

    # Padding con ceros hasta la potencia de 2
    A_pad = np.pad(A, ((0, size - m), (0, size - k)))
    B_pad = np.pad(B, ((0, size - k), (0, size - n)))

    C_pad = _strassen_recursive(A_pad, B_pad)
    return C_pad[:m, :n]

# ----------------------------------------------------------
# Benchmark
# ----------------------------------------------------------
if __name__ == "__main__":
    sys.setrecursionlimit(10_000)  # Evita RecursionError para n ≤ 1024-2048

    #sizes = [2, 4, 8, 16, 32, 64, 128, 256]
    sizes = [2, 4, 8, 16, 32, 64]
    #sizes = [128,256,512,1024]
    times = []
    repeats = 10  # Número de repeticiones
    for n in sizes:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        total_time = 0
        for _ in range(repeats):
            t0 = time.perf_counter()
            strassen_multiply(A, B)
            total_time += time.perf_counter() - t0
        elapsed = total_time / repeats  # Promedio
        times.append(elapsed)
        print(f"Tamaño {n}: {elapsed:.5f} segundos")

    # ----------------------------------------------------------
    # Gráfico con etiquetas de n sobre cada punto
    # ----------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, times, 'o-', color='tab:blue')

    # Etiqueta con el valor de n sobre cada punto
    for n, t in zip(sizes, times):
        plt.text(n, t, str(n), ha='center', va='bottom', fontsize=9)

    plt.title('Tiempo de ejecución – Algoritmo Strassen puro')
    plt.xlabel('Tamaño de la matriz (n)')
    plt.ylabel('Tiempo (s)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()