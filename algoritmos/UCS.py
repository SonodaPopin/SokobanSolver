"""
algorithms/UCS.py
-----------------
Uniform Cost Search para Sokoban.
Usa los módulos de tablero, hashing y deadlocks.

Diferencia con BFS y DFS:
- BFS: explora por capas de profundidad (todos los nodos a distancia 1, luego 2, etc.)
- DFS: va lo más profundo posible antes de retroceder
- UCS: explora en orden de costo acumulado usando una cola de prioridad.
  Garantiza encontrar la solución de MENOR COSTO, no necesariamente menor pasos.

El paper evalúa dos funciones de costo:
  Costo 1 (cost1): 
    - Empujar caja fuera de objetivo → costo muy alto (100)
    - Mover caja hacia objetivo      → costo medio (10)
    - Mover jugador                  → costo bajo (1)

  Costo 2 (cost2):
    - Mover jugador                  → costo 1
    - Empujar caja (cualquier caso)  → costo 1  (igual que mover)
    → Esto hace que UCS se comporte similar a BFS

Según el paper, cost1 tuvo mejor rendimiento que cost2.
"""

import heapq
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tablero
from optimizacion.hashing import StateHash
from optimizacion.deadlocks import init_deadlocks, has_deadlock


# ── Funciones de costo (replicando el paper) ─────────────────

def cost1(is_push, pushing_off_goal, pushing_to_goal):
    """
    Costo 1 del paper:
    - Empujar caja fuera de objetivo → costo alto
    - Empujar caja en general        → costo medio
    - Mover jugador                  → costo bajo
    """
    if is_push:
        if pushing_off_goal:
            return 100   # muy penalizado: aleja caja de objetivo
        elif pushing_to_goal:
            return 1     # premiado: acerca caja a objetivo
        else:
            return 10    # costo normal de empuje
    return 1             # mover jugador


def cost2(is_push, pushing_off_goal, pushing_to_goal):
    """
    Costo 2 del paper:
    - Todo cuesta igual (1)
    - Hace que UCS se comporte como BFS
    """
    return 1


def solve(cost_fn=cost1):
    """
    Resuelve el nivel actual usando UCS con hashing y detección de deadlocks.
    El tablero debe estar inicializado con tablero.init() antes de llamar.

    Parámetros:
        cost_fn : función de costo a usar (cost1 o cost2)

    Retorna:
        (solution, states_explored)
        solution : string de movimientos (minúsculas=mover, MAYÚSCULAS=empujar)
        states   : número de estados explorados
    """
    init_deadlocks(tablero.sdata, tablero.nrows, tablero.ncols)

    visited = StateHash()
    visited.add(tablero.ddata)

    # Cola de prioridad: (costo_acumulado, estado, solucion, x, y)
    # heapq en Python es un min-heap: siempre saca el de menor costo primero
    counter = 0  # desempate cuando los costos son iguales
    pq = [(0, counter, tablero.ddata, "", tablero.start_x, tablero.start_y)]
    dirs = ((0, -1, 'u', 'U'), (1, 0, 'r', 'R'),
            (0,  1, 'd', 'D'), (-1, 0, 'l', 'L'))

    while pq:
        cost, _, cur, sol, x, y = heapq.heappop(pq)

        for dx, dy, ml, mu in dirs:
            nx, ny = x + dx, y + dy

            if cur[tablero.idx(nx, ny)] == '*':
                # Hay caja → intentar empujar
                new_state = tablero.push(x, y, dx, dy, cur)
                if new_state and new_state not in visited:
                    if has_deadlock(new_state):
                        visited.add(new_state)
                        continue

                    # Calcular costo del movimiento
                    box_was_on_goal = tablero.sdata[tablero.idx(nx, ny)] == '.'
                    box_now_on_goal = tablero.sdata[tablero.idx(nx+dx, ny+dy)] == '.'
                    move_cost = cost_fn(
                        is_push=True,
                        pushing_off_goal=box_was_on_goal and not box_now_on_goal,
                        pushing_to_goal=box_now_on_goal
                    )

                    if tablero.is_solved(new_state):
                        return sol + mu, len(visited)

                    counter += 1
                    heapq.heappush(pq, (cost + move_cost, counter,
                                        new_state, sol + mu, nx, ny))
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
                    move_cost = cost_fn(
                        is_push=False,
                        pushing_off_goal=False,
                        pushing_to_goal=False
                    )
                    if tablero.is_solved(new_state):
                        return sol + ml, len(visited)
                    counter += 1
                    heapq.heappush(pq, (cost + move_cost, counter,
                                        new_state, sol + ml, nx, ny))
                    visited.add(new_state)

    return "No solution", len(visited)


def solve_cost1():
    """Wrapper para usar UCS con función de costo 1 (mejor según el paper)."""
    return solve(cost_fn=cost1)


def solve_cost2():
    """Wrapper para usar UCS con función de costo 2."""
    return solve(cost_fn=cost2)