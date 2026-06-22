# cnn/generar_datos.py
# Genera datos de entrenamiento para la CNN.
# Para cada nivel, corre un solver y registra cada (estado, accion)
# de la solucion encontrada. Se puede generar con BFS (como el paper,
# que usa Backtracking) o con A* mejorado (la propuesta del proyecto).
#
# Cada estado se guarda como una matriz 32x32 con 4 canales (paredes,
# objetivos, cajas, jugador) y la accion como un indice 0-3 (U,R,D,L).
#
# Tambien se aplica aumento de datos: rotaciones de 90 y reflexion,
# multiplicando el dataset x8 como hace el paper.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import tablero

BOARD_SIZE = 32  # tamaño fijo de la representacion (igual al paper)
ACTIONS = ['U', 'R', 'D', 'L']  # orden fijo de las acciones de salida
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}


def estado_a_matriz(state, sdata, nrows, ncols):
    """
    Convierte un estado del tablero a una matriz BOARD_SIZE x BOARD_SIZE
    con 4 canales: pared, objetivo, caja, jugador.
    El tablero original se centra en la matriz fija de 32x32.
    """
    canales = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    # offset para centrar el tablero real dentro de la matriz fija
    off_y = (BOARD_SIZE - nrows) // 2
    off_x = (BOARD_SIZE - ncols) // 2

    for y in range(nrows):
        for x in range(ncols):
            i = y * ncols + x
            sy, sx = y + off_y, x + off_x
            if not (0 <= sy < BOARD_SIZE and 0 <= sx < BOARD_SIZE):
                continue
            if sdata[i] == '#':
                canales[0, sy, sx] = 1
            if sdata[i] == '.':
                canales[1, sy, sx] = 1
            if state[i] == '*':
                canales[2, sy, sx] = 1
            if state[i] == '@':
                canales[3, sy, sx] = 1

    return canales


def solucion_a_pares(level_path, solver_fn):
    """
    Corre un solver sobre un nivel y devuelve la lista de pares
    (matriz_estado, accion) recorriendo la solucion paso a paso.
    Solo se quedan los movimientos de jugador en mayuscula/minuscula
    traducidos a una de las 4 direcciones U/R/D/L.
    """
    tablero.init(level_path)
    sdata, nrows, ncols = tablero.sdata, tablero.nrows, tablero.ncols

    solution, _ = solver_fn()
    if solution in ("No solution", "Timeout"):
        return []

    pares = []
    state = tablero.ddata
    x, y = tablero.start_x, tablero.start_y

    dir_map = {
        'u': (0, -1, 'U'), 'U': (0, -1, 'U'),
        'r': (1,  0, 'R'), 'R': (1,  0, 'R'),
        'd': (0,  1, 'D'), 'D': (0,  1, 'D'),
        'l': (-1, 0, 'L'), 'L': (-1, 0, 'L'),
    }

    for move in solution:
        dx, dy, accion = dir_map[move]
        matriz = estado_a_matriz(state, sdata, nrows, ncols)
        pares.append((matriz, ACTION_TO_IDX[accion]))

        nx, ny = x + dx, y + dy
        if move.isupper():
            new_state = tablero.push(x, y, dx, dy, state)
        else:
            new_state = tablero.move_player(x, y, dx, dy, state)

        if new_state is None:
            break
        state = new_state
        x, y = nx, ny

    return pares


def _rotar_90(matriz):
    """Rota una matriz de canales 90 grados (sobre los ejes espaciales)."""
    return np.array([np.rot90(c) for c in matriz])


def _reflejar(matriz):
    """Refleja horizontalmente una matriz de canales."""
    return np.array([np.fliplr(c) for c in matriz])


ROTACION_ACCION = {
    # como rota la accion cuando se rota el tablero 90 grados (sentido horario)
    'U': 'R', 'R': 'D', 'D': 'L', 'L': 'U'
}
REFLEJO_ACCION = {
    # como cambia la accion al reflejar horizontalmente
    'U': 'U', 'D': 'D', 'L': 'R', 'R': 'L'
}


def aumentar_datos(pares):
    """
    Aplica rotaciones (0,90,180,270) y reflexion a cada par,
    multiplicando el dataset x8, igual que en el paper.
    """
    aumentados = []
    for matriz, accion_idx in pares:
        accion = ACTIONS[accion_idx]
        m, a = matriz, accion

        for _ in range(4):
            aumentados.append((m, ACTION_TO_IDX[a]))
            m_reflejada = _reflejar(m)
            a_reflejada = REFLEJO_ACCION[a]
            aumentados.append((m_reflejada, ACTION_TO_IDX[a_reflejada]))

            m = _rotar_90(m)
            a = ROTACION_ACCION[a]

    return aumentados


def generar_dataset(level_paths, solver_fn, aumentar=True):
    """
    Genera el dataset completo recorriendo varios niveles con el
    solver indicado. Retorna (X, y) como arrays de numpy listos
    para entrenar.
    """
    todos_los_pares = []
    for path in level_paths:
        pares = solucion_a_pares(path, solver_fn)
        todos_los_pares.extend(pares)

    if aumentar:
        todos_los_pares = aumentar_datos(todos_los_pares)

    if not todos_los_pares:
        return np.empty((0, 4, BOARD_SIZE, BOARD_SIZE), dtype=np.float32), np.empty((0,), dtype=np.int64)

    X = np.stack([p[0] for p in todos_los_pares])
    y = np.array([p[1] for p in todos_los_pares], dtype=np.int64)
    return X, y


def guardar_dataset(X, y, path):
    np.savez_compressed(path, X=X, y=y)
    print(f"Dataset guardado en {path}: {len(y)} ejemplos")


def cargar_dataset(path):
    data = np.load(path)
    return data["X"], data["y"]
