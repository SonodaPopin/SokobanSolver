"""
algoritmos/BFS.py
-----------------
Breadth First Search para Sokoban.
Usa los módulos de tablero, hashing y deadlocks.
"""

from collections import deque
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tablero
from optimizacion.hashing import StateHash
from optimizacion.deadlocks import init_deadlocks, has_deadlock


def solve():
    """
    Resuelve el nivel actual usando BFS con hashing y detección de deadlocks.
    El tablero debe estar inicializado con tablero.init() antes de llamar.

    Retorna:
        (solution, states_explored)
        solution : string de movimientos (minúsculas=mover, MAYÚSCULAS=empujar)
        states   : número de estados explorados
    """
    init_deadlocks(tablero.sdata, tablero.nrows, tablero.ncols)

    visited = StateHash()
    visited.add(tablero.ddata)
    queue = deque([(tablero.ddata, "", tablero.start_x, tablero.start_y)])
    dirs = ((0, -1, 'u', 'U'), (1, 0, 'r', 'R'),
            (0,  1, 'd', 'D'), (-1, 0, 'l', 'L'))

    while queue:
        cur, sol, x, y = queue.popleft()

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
                    queue.append((new_state, sol + mu, nx, ny))
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
                    queue.append((new_state, sol + ml, nx, ny))
                    visited.add(new_state)

    return "No solution", len(visited)
