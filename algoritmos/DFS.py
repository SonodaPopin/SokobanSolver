"""
algoritmos/DFS.py
-----------------
Depth First Search para Sokoban.
Usa los módulos de tablero, hashing y deadlocks.

Diferencia con BFS:
- BFS usa cola (FIFO) → explora por capas de profundidad → solución más corta
- DFS usa pila (LIFO) → va lo más profundo posible antes de retroceder → más rápido
  en encontrar ALGUNA solución, pero no garantiza que sea la óptima.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tablero
from optimizacion.hashing import StateHash
from optimizacion.deadlocks import init_deadlocks, has_deadlock

# Profundidad máxima para evitar búsqueda infinita
MAX_DEPTH = 150


def solve():
    """
    Resuelve el nivel actual usando DFS con hashing y detección de deadlocks.
    El tablero debe estar inicializado con tablero.init() antes de llamar.

    Retorna:
        (solution, states_explored)
        solution : string de movimientos (minúsculas=mover, MAYÚSCULAS=empujar)
        states   : número de estados explorados
    """
    init_deadlocks(tablero.sdata, tablero.nrows, tablero.ncols)

    visited = StateHash()
    visited.add(tablero.ddata)

    # Pila en vez de cola — única diferencia estructural con BFS
    stack = [(tablero.ddata, "", tablero.start_x, tablero.start_y, 0)]
    dirs = ((0, -1, 'u', 'U'), (1, 0, 'r', 'R'),
            (0,  1, 'd', 'D'), (-1, 0, 'l', 'L'))

    while stack:
        cur, sol, x, y, depth = stack.pop()  # pop() en vez de popleft()

        # Límite de profundidad para evitar búsqueda infinita
        if depth >= MAX_DEPTH:
            continue

        for dx, dy, ml, mu in dirs:
            nx, ny = x + dx, y + dy

            if cur[tablero.idx(nx, ny)] == '*':
                # Hay caja → intentar empujar
                new_state = tablero.push(x, y, dx, dy, cur)
                if new_state and new_state not in visited:
                    if has_deadlock(new_state):
                        visited.add(new_state)
                        continue
                    if tablero.is_solved(new_state):
                        return sol + mu, len(visited)
                    stack.append((new_state, sol + mu, nx, ny, depth + 1))
                    visited.add(new_state)
            else:
                # Casilla libre → mover jugador
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
                    stack.append((new_state, sol + ml, nx, ny, depth + 1))
                    visited.add(new_state)

    return "No solution", len(visited)