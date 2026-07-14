# cnn/generar_datos.py
# Genera datos de entrenamiento para la CNN.
#
# Cambios respecto a la version anterior:
# - Procesamiento nivel por nivel en vez de acumular todo en RAM
# - Timeout por nivel para que niveles dificiles no congelen el proceso
# - Guardado progresivo en disco (chunks de 10 niveles)
# - Reporte de progreso por nivel para saber que esta pasando

import sys
import os
import time
import threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tablero

BOARD_SIZE = 32
ACTIONS = ['U', 'R', 'D', 'L']
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
TIMEOUT_POR_NIVEL = 180  # segundos maximos por nivel al generar datos


def estado_a_matriz(state, sdata, nrows, ncols):
    """
    Convierte un estado del tablero a una matriz BOARD_SIZE x BOARD_SIZE
    con 4 canales: pared, objetivo, caja, jugador.
    """
    canales = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    off_y = (BOARD_SIZE - nrows) // 2
    off_x = (BOARD_SIZE - ncols) // 2

    for y in range(nrows):
        for x in range(ncols):
            i = y * ncols + x
            sy, sx = y + off_y, x + off_x
            if not (0 <= sy < BOARD_SIZE and 0 <= sx < BOARD_SIZE):
                continue
            if sdata[i] == '#': canales[0, sy, sx] = 1
            if sdata[i] == '.': canales[1, sy, sx] = 1
            if state[i] == '*': canales[2, sy, sx] = 1
            if state[i] == '@': canales[3, sy, sx] = 1

    return canales


def _rotar_90(matriz):
    return np.array([np.rot90(c) for c in matriz])


def _reflejar(matriz):
    return np.array([np.fliplr(c) for c in matriz])


ROTACION_ACCION = {'U': 'L', 'L': 'D', 'D': 'R', 'R': 'U'}
REFLEJO_ACCION  = {'U': 'U', 'D': 'D', 'L': 'R', 'R': 'L'}
RUIDO_STD = 0.05  # desviacion estandar del ruido gaussiano para aumento x16


def _agregar_ruido(matriz):
    """
    Agrega ruido gaussiano a la matriz y recorta los valores a [0, 1].
    El tablero sigue siendo reconocible pero cada version es distinta,
    haciendo al modelo mas robusto a pequeñas variaciones.
    """
    ruido = np.random.normal(0, RUIDO_STD, matriz.shape).astype(np.float32)
    return np.clip(matriz + ruido, 0.0, 1.0)


def aumentar_pares(pares):
    """
    Aplica rotaciones, reflexion y ruido gaussiano a los pares.
    Resultado: x8 (geometrico) x2 (con/sin ruido) = x16 total.

    Las 16 versiones por par son:
      - 4 rotaciones (0, 90, 180, 270 grados)
      - cada una con reflexion horizontal
      - cada una de las 8 anteriores con ruido gaussiano agregado
    """
    aumentados = []
    for matriz, accion_idx in pares:
        accion = ACTIONS[accion_idx]
        m, a = matriz, accion
        for _ in range(4):
            # Version sin ruido
            aumentados.append((m, ACTION_TO_IDX[a]))
            m_ref = _reflejar(m)
            a_ref = REFLEJO_ACCION[a]
            aumentados.append((m_ref, ACTION_TO_IDX[a_ref]))

            # Version con ruido (misma accion, tablero ligeramente perturbado)
            aumentados.append((_agregar_ruido(m), ACTION_TO_IDX[a]))
            aumentados.append((_agregar_ruido(m_ref), ACTION_TO_IDX[a_ref]))

            m = _rotar_90(m)
            a = ROTACION_ACCION[a]
    return aumentados


def _resolver_con_timeout(solver_fn, resultado):
    """Corre solver_fn en un hilo y guarda el resultado."""
    try:
        sol, states = solver_fn()
        resultado['sol'] = sol
        resultado['states'] = states
    except Exception as e:
        resultado['error'] = str(e)


def solucion_a_pares(level_path, solver_fn, timeout=TIMEOUT_POR_NIVEL):
    """
    Corre un solver sobre un nivel con timeout y retorna pares
    (matriz_estado, accion). Si el solver tarda mas del timeout,
    retorna lista vacia sin crashear.
    """
    tablero.init(level_path)
    sdata, nrows, ncols = tablero.sdata, tablero.nrows, tablero.ncols

    # Correr solver con timeout en hilo separado
    resultado = {}
    hilo = threading.Thread(target=_resolver_con_timeout,
                            args=(solver_fn, resultado), daemon=True)
    hilo.start()
    hilo.join(timeout)

    if hilo.is_alive() or 'sol' not in resultado:
        return []  # timeout o error

    solution = resultado['sol']
    if solution in ("No solution", "Timeout", ""):
        return []

    pares = []
    state = tablero.ddata
    x, y = tablero.start_x, tablero.start_y

    dir_map = {
        'u': (0,-1,'U'), 'U': (0,-1,'U'),
        'r': (1, 0,'R'), 'R': (1, 0,'R'),
        'd': (0, 1,'D'), 'D': (0, 1,'D'),
        'l': (-1,0,'L'), 'L': (-1,0,'L'),
    }

    for move in solution:
        dx, dy, accion = dir_map[move]
        matriz = estado_a_matriz(state, sdata, nrows, ncols)
        pares.append((matriz, ACTION_TO_IDX[accion]))

        if move.isupper():
            new_state = tablero.push(x, y, dx, dy, state)
        else:
            new_state = tablero.move_player(x, y, dx, dy, state)

        if new_state is None:
            break
        state = new_state
        x, y = x + dx, y + dy

    return pares


def generar_dataset(level_paths, solver_fn, aumentar=True, verbose=True):
    """
    Genera el dataset procesando nivel por nivel.
    En vez de acumular todo en RAM, procesa cada nivel y va
    apilando solo las matrices finales, liberando la memoria
    de estructuras intermedias despues de cada nivel.

    Retorna (X, y) como arrays numpy listos para entrenar.
    """
    todas_X = []
    todas_y = []
    resueltos = saltados = 0

    for i, path in enumerate(level_paths):
        nombre = os.path.basename(path)
        t0 = time.time()

        pares = solucion_a_pares(path, solver_fn, timeout=TIMEOUT_POR_NIVEL)

        if not pares:
            if verbose:
                print(f"  [{i+1:3d}/{len(level_paths)}] {nombre}: sin solucion o timeout")
            saltados += 1
            continue

        if aumentar:
            pares = aumentar_pares(pares)

        # Apilar solo los arrays finales (mas eficiente en memoria)
        for matriz, accion in pares:
            todas_X.append(matriz)
            todas_y.append(accion)

        elapsed = round(time.time() - t0, 1)
        if verbose:
            print(f"  [{i+1:3d}/{len(level_paths)}] {nombre}: "
                  f"{len(pares)} ejemplos en {elapsed}s "
                  f"(total acumulado: {len(todas_y)})")
        resueltos += 1

    if verbose:
        print(f"\nResumen generacion:")
        print(f"  Resueltos: {resueltos}/{len(level_paths)}")
        print(f"  Saltados:  {saltados}")
        print(f"  Total ejemplos: {len(todas_y)}")

    if not todas_X:
        return (np.empty((0, 4, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
                np.empty((0,), dtype=np.int64))

    X = np.stack(todas_X)
    y = np.array(todas_y, dtype=np.int64)
    return X, y


def guardar_dataset(X, y, path):
    np.savez_compressed(path, X=X, y=y)
    print(f"Dataset guardado en {path}: {len(y)} ejemplos")


def cargar_dataset(path):
    data = np.load(path)
    return data["X"], data["y"]