"""
optimizacion/heuristics.py
-----------------------------
Heurísticas para A*, replicando las 4 del paper más una mejora propia.

Heurísticas del paper (Tabla 1):
    1. Eucl. mín.      → distancia euclidiana al objetivo más cercano
    2. Manh. mín.       → distancia Manhattan al objetivo más cercano
    3. Húng. + Eucl.    → asignación óptima (húngaro) con distancia euclidiana
    4. Húng. + Manh.    → asignación óptima (húngaro) con distancia Manhattan

Mejora propuesta (no está en el paper):
    5. Húng. + Manh. + Deadlock Penalty
       → igual a la 4, pero suma una penalización fuerte si alguna
         caja está cerca de convertirse en deadlock, guiando la
         búsqueda lejos de movimientos peligrosos antes de que
         el deadlock ocurra.

Uso:
    from optimizacion.heuristicas import HEURISTICAS

    h = HEURISTICAS["hungarian_manhattan_deadlock"](state)
"""

import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tablero
from optimizacion.hungarian import hungarian_distance
from optimizacion.deadlocks import get_dead_squares

_goals_cache = None
_goals_cache_key = None


def _get_goals():
    """Cachea la lista de objetivos ya que no cambia durante la búsqueda."""
    global _goals_cache, _goals_cache_key
    key = id(tablero.sdata)
    if _goals_cache_key != key:
        _goals_cache = [(x, y) for y in range(tablero.nrows) for x in range(tablero.ncols)
                         if tablero.sdata[tablero.idx(x, y)] == '.']
        _goals_cache_key = key
    return _goals_cache


def _get_boxes_fast(state):
    """
    Extrae posiciones de cajas iterando el string directamente
    en vez de usar tablero.idx() por cada celda (más rápido).
    """
    ncols = tablero.ncols
    return [(i % ncols, i // ncols) for i, ch in enumerate(state) if ch == '*']


def _manhattan_min(boxes, goals):
    """Para cada caja, distancia Manhattan al objetivo más cercano. Suma todo."""
    total = 0
    for bx, by in boxes:
        total += min(abs(bx - gx) + abs(by - gy) for gx, gy in goals)
    return total


def _euclidean_min(boxes, goals):
    """Para cada caja, distancia euclidiana al objetivo más cercano. Suma todo."""
    total = 0
    for bx, by in boxes:
        total += min(math.sqrt((bx - gx)**2 + (by - gy)**2) for gx, gy in goals)
    return total


def h_euclidean_min(state):
    """Heurística 1 del paper: distancia euclidiana mínima."""
    boxes = _get_boxes_fast(state)
    goals = _get_goals()
    return _euclidean_min(boxes, goals)


def h_manhattan_min(state):
    """Heurística 2 del paper: distancia Manhattan mínima."""
    boxes = _get_boxes_fast(state)
    goals = _get_goals()
    return _manhattan_min(boxes, goals)


def h_hungarian_euclidean(state):
    """Heurística 3 del paper: asignación óptima con distancia euclidiana."""
    boxes = _get_boxes_fast(state)
    goals = _get_goals()
    return hungarian_distance(boxes, goals, metric="euclidean")


def h_hungarian_manhattan(state):
    """Heurística 4 del paper: asignación óptima con distancia Manhattan."""
    boxes = _get_boxes_fast(state)
    goals = _get_goals()
    return hungarian_distance(boxes, goals, metric="manhattan")


# ── Mejora propuesta: penalización por proximidad a deadlock ────

DEADLOCK_PENALTY = 10  # peso de la penalización


def _near_deadlock_penalty(boxes):
    """
    Penaliza cajas que están adyacentes a una dead square
    (un movimiento más y quedarían atrapadas).
    No es un deadlock confirmado, sino una advertencia temprana
    que guía a A* a evitar acercarse a esas zonas.
    """
    dead_squares = get_dead_squares()
    if not dead_squares:
        return 0

    penalty = 0
    dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    for bx, by in boxes:
        for dx, dy in dirs:
            if (bx + dx, by + dy) in dead_squares:
                penalty += DEADLOCK_PENALTY
                break  # una sola penalización por caja es suficiente
    return penalty


def h_hungarian_manhattan_deadlock(state):
    """
    Heurística 5 (mejora propuesta): asignación óptima (húngaro)
    con distancia Manhattan + penalización por cercanía a deadlocks.

    El objetivo es que A* evite acercar cajas a zonas peligrosas
    incluso antes de que se conviertan en deadlock real.
    """
    boxes = _get_boxes_fast(state)
    goals = _get_goals()
    base = hungarian_distance(boxes, goals, metric="manhattan")
    penalty = _near_deadlock_penalty(boxes)
    return base + penalty


# ── Registro de heurísticas disponibles ──────────────────────
HEURISTICAS = {
    "Eucl. min":              h_euclidean_min,
    "Manh. min":               h_manhattan_min,
    "Hung. + Eucl.":           h_hungarian_euclidean,
    "Hung. + Manh.":           h_hungarian_manhattan,
    "Hung. + Manh. + Deadlock": h_hungarian_manhattan_deadlock,
}