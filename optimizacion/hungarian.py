"""
optimizacion/hungarian.py
---------------------------
Implementación del algoritmo Húngaro para asignación óptima
de cajas a objetivos, usado por las heurísticas de A*.

El algoritmo húngaro resuelve el problema de asignación:
dado un conjunto de cajas y un conjunto de objetivos,
encuentra la asignación uno-a-uno que minimiza la suma
total de distancias (a diferencia de la heurística "mínima"
simple, que puede asignar dos cajas al mismo objetivo).

Uso:
    from optimizacion.hungarian import hungarian_distance

    total = hungarian_distance(boxes, goals, metric="manhattan")
"""

import math


def _manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def _euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def hungarian_distance(boxes, goals, metric="manhattan"):
    """
    Calcula la distancia total mínima asignando cada caja a un
    objetivo distinto (algoritmo húngaro / asignación óptima).

    Parámetros:
        boxes  : lista o set de tuplas (x, y) con posiciones de cajas
        goals  : lista o set de tuplas (x, y) con posiciones de objetivos
        metric : "manhattan" o "euclidean"

    Retorna:
        float: suma de distancias de la asignación óptima
    """
    boxes = list(boxes)
    goals = list(goals)
    n = len(boxes)

    if n == 0:
        return 0
    if n != len(goals):
        # Si no coinciden en cantidad (no debería pasar en un nivel válido),
        # usamos el mínimo entre ambos y el resto se ignora.
        n = min(len(boxes), len(goals))
        boxes = boxes[:n]
        goals = goals[:n]

    dist_fn = _manhattan if metric == "manhattan" else _euclidean

    # Matriz de costos
    cost = [[dist_fn(boxes[i], goals[j]) for j in range(n)] for i in range(n)]

    return _hungarian_algorithm(cost)


def _hungarian_algorithm(cost_matrix):
    """
    Implementación del algoritmo Húngaro (método de Kuhn-Munkres)
    usando el algoritmo simplificado O(n^3) con potenciales.

    Retorna el costo total mínimo de la asignación óptima.
    """
    n = len(cost_matrix)
    INF = float('inf')

    # Padding: el algoritmo clásico de Kuhn-Munkres usa indices 1..n
    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)   # p[j] = fila asignada a la columna j
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1

            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost_matrix[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    total = 0
    for j in range(1, n + 1):
        if p[j] != 0:
            total += cost_matrix[p[j] - 1][j - 1]

    return total