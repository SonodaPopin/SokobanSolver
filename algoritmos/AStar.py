"""
algorithms/astar.py
--------------------
A* para Sokoban.
Usa los módulos de tablero, hashing, deadlocks y heuristics.

A* = UCS + heurística. En vez de explorar solo por costo acumulado,
usa f(n) = g(n) + h(n), donde:
    g(n) = costo acumulado real desde el inicio
    h(n) = estimación heurística del costo restante

El paper evalúa 4 heurísticas (Tabla 1). Aquí se agregan las mismas
4 más una quinta heurística propia con penalización por deadlocks.
"""

import heapq
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tablero
from optimizacion.hashing import StateHash
from optimizacion.deadlocks import init_deadlocks, has_deadlock
from optimizacion.heuristicas import HEURISTICAS


def solve(heuristic_name="Hung. + Manh."):
    """
    Resuelve el nivel actual usando A* con la heurística indicada.
    El tablero debe estar inicializado con tablero.init() antes de llamar.

    Parámetros:
        heuristic_name : nombre de la heurística (ver optimizacion.heuristicas.HEURISTICAS)

    Retorna:
        (solution, states_explored)
    """
    init_deadlocks(tablero.sdata, tablero.nrows, tablero.ncols)
    heuristic_fn = HEURISTICAS[heuristic_name]

    visited = StateHash()
    visited.add(tablero.ddata)

    # Cola de prioridad: (f_costo, contador, g_costo, estado, solucion, x, y)
    counter = 0
    h0 = heuristic_fn(tablero.ddata)
    pq = [(h0, counter, 0, tablero.ddata, "", tablero.start_x, tablero.start_y)]
    dirs = ((0, -1, 'u', 'U'), (1, 0, 'r', 'R'),
            (0,  1, 'd', 'D'), (-1, 0, 'l', 'L'))

    while pq:
        f, _, g, cur, sol, x, y = heapq.heappop(pq)

        for dx, dy, ml, mu in dirs:
            nx, ny = x + dx, y + dy

            if cur[tablero.idx(nx, ny)] == '*':
                new_state = tablero.push(x, y, dx, dy, cur)
                if new_state and new_state not in visited:
                    if has_deadlock(new_state):
                        visited.add(new_state)
                        continue
                    if tablero.is_solved(new_state):
                        return sol + mu, len(visited)

                    new_g = g + 1
                    new_h = heuristic_fn(new_state)
                    counter += 1
                    heapq.heappush(pq, (new_g + new_h, counter, new_g,
                                        new_state, sol + mu, nx, ny))
                    visited.add(new_state)
            else:
                if (tablero.sdata[tablero.idx(nx, ny)] == '#' or
                        cur[tablero.idx(nx, ny)] != ' '):
                    continue
                d2 = bytearray(cur.encode())
                d2[tablero.idx(x, y)]   = ord(' ')
                d2[tablero.idx(nx, ny)] = ord('@')
                new_state = d2.decode()
                if new_state not in visited:
                    if tablero.is_solved(new_state):
                        return sol + ml, len(visited)

                    new_g = g + 1
                    new_h = heuristic_fn(new_state)
                    counter += 1
                    heapq.heappush(pq, (new_g + new_h, counter, new_g,
                                        new_state, sol + ml, nx, ny))
                    visited.add(new_state)

    return "No solution", len(visited)


# ── Wrappers para cada heurística (facilita comparación en main.py) ──

def solve_euclidean_min():
    return solve("Eucl. min")

def solve_manhattan_min():
    return solve("Manh. min")

def solve_hungarian_euclidean():
    return solve("Hung. + Eucl.")

def solve_hungarian_manhattan():
    return solve("Hung. + Manh.")

def solve_hungarian_manhattan_deadlock():
    return solve("Hung. + Manh. + Deadlock")