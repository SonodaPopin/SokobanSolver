# cnn/predecir.py
# Usa un modelo CNN entrenado para resolver un nivel jugando paso a paso.
# A diferencia de los algoritmos de busqueda, la CNN no explora estados:
# predice directamente la siguiente accion desde el estado actual.
#
# Como puede caer en ciclos (limitacion reportada en el paper), se aplica
# la correccion: si la accion propuesta repite un estado ya visitado,
# se toma la siguiente mejor opcion segun las probabilidades del modelo.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import tablero
from cnn.generar_datos import estado_a_matriz, ACTIONS
from cnn.modelo import crear_modelo

MAX_PASOS = 300  # limite de movimientos antes de declarar fallo


def cargar(path, n_filtros=16, device="cpu"):
    modelo = crear_modelo(n_filtros=n_filtros, device=device)
    modelo.load_state_dict(torch.load(path, map_location=device))
    modelo.eval()
    return modelo


def resolver_con_cnn(modelo, device="cpu"):
    """
    Resuelve el nivel actualmente cargado en tablero.py usando la CNN.
    El tablero debe estar inicializado con tablero.init() antes de llamar.

    Retorna (solucion, pasos_dados) igual que los otros solvers,
    para que main.py los pueda comparar con la misma interfaz.
    """
    state = tablero.ddata
    x, y = tablero.start_x, tablero.start_y
    sdata, nrows, ncols = tablero.sdata, tablero.nrows, tablero.ncols

    dirs = {'U': (0, -1), 'R': (1, 0), 'D': (0, 1), 'L': (-1, 0)}

    visitados = set()
    visitados.add(state)
    solucion = ""

    for _ in range(MAX_PASOS):
        if tablero.is_solved(state):
            return solucion, len(visitados)

        matriz = estado_a_matriz(state, sdata, nrows, ncols)
        entrada = torch.tensor(matriz).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = modelo(entrada)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        # ordenar acciones de mejor a peor segun el modelo
        orden_acciones = np.argsort(-probs)

        movido = False
        for idx_accion in orden_acciones:
            accion = ACTIONS[idx_accion]
            dx, dy = dirs[accion]
            nx, ny = x + dx, y + dy

            if state[tablero.idx(nx, ny)] == '*':
                nuevo_estado = tablero.push(x, y, dx, dy, state)
                letra = accion  # mayuscula = empuje
            else:
                nuevo_estado = tablero.move_player(x, y, dx, dy, state)
                letra = accion.lower()

            if nuevo_estado is None:
                continue

            # correccion de ciclos: si ya visitamos este estado, probar
            # la siguiente accion segun el modelo en vez de repetir
            if nuevo_estado in visitados:
                continue

            state = nuevo_estado
            x, y = nx, ny
            solucion += letra
            visitados.add(state)
            movido = True
            break

        if not movido:
            # todas las acciones llevan a estados ya visitados: forzar
            # la mejor accion del modelo aunque repita (evita trabarse)
            accion = ACTIONS[orden_acciones[0]]
            dx, dy = dirs[accion]
            nx, ny = x + dx, y + dy
            if state[tablero.idx(nx, ny)] == '*':
                nuevo_estado = tablero.push(x, y, dx, dy, state)
                letra = accion
            else:
                nuevo_estado = tablero.move_player(x, y, dx, dy, state)
                letra = accion.lower()
            if nuevo_estado is None:
                break
            state = nuevo_estado
            x, y = nx, ny
            solucion += letra

    return "No solution", len(visitados)


def solve(modelo_path="cnn/modelo_bfs.pt", n_filtros=16):
    """Wrapper generico con la misma interfaz que los demas solvers (solve())."""
    modelo = cargar(modelo_path, n_filtros=n_filtros)
    return resolver_con_cnn(modelo)


def solve_cnn_bfs():
    """CNN entrenada con datos generados por BFS."""
    return solve(modelo_path="cnn/modelo_bfs.pt")


def solve_cnn_astar():
    """CNN entrenada con datos generados por A* (Hungaro+Manhattan+Deadlock)."""
    return solve(modelo_path="cnn/modelo_astar.pt")
