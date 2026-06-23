"""
deadlocks.py
------------
Módulo independiente de detección de deadlocks para Sokoban.
Puede ser importado por cualquier algoritmo de búsqueda.

Uso:
    from deadlocks import init_deadlocks, has_deadlock

    init_deadlocks(sdata, nrows, ncols)   # llamar una vez al inicio
    if has_deadlock(state):               # llamar por cada estado nuevo
        ...
"""

from collections import deque

# Estado interno del módulo (se inicializa con init_deadlocks)
_sdata = ""
_nrows = 0
_ncols = 0
_dead_squares = set()


def init_deadlocks(sdata, nrows, ncols):
    """
    Inicializa el módulo con el mapa estático del nivel.
    Debe llamarse una vez antes de usar has_deadlock().

    Parámetros:
        sdata  : string del mapa estático (paredes # y objetivos .)
        nrows  : número de filas del tablero
        ncols  : número de columnas del tablero
    """
    global _sdata, _nrows, _ncols, _dead_squares
    _sdata = sdata
    _nrows = nrows
    _ncols = ncols
    _dead_squares = _compute_dead_squares()


def _idx(x, y):
    return y * _ncols + x


def _compute_dead_squares():
    """
    Detecta dead squares por esquinas:
    una casilla es dead square si tiene pared en al menos un lado
    horizontal (izq o der) Y al menos un lado vertical (arr o abajo),
    y no es objetivo. Una caja en esa posicion no puede salir.
    """
    goals = {(x, y) for y in range(_nrows) for x in range(_ncols)
             if _sdata[_idx(x, y)] == '.'}

    dead = set()
    for y in range(_nrows):
        for x in range(_ncols):
            if _sdata[_idx(x, y)] == '#':
                continue
            if (x, y) in goals:
                continue

            # pared en alguno de los lados horizontales
            blocked_h = (
                (x - 1 < 0 or _sdata[_idx(x-1, y)] == '#') or
                (x + 1 >= _ncols or _sdata[_idx(x+1, y)] == '#')
            )
            # pared en alguno de los lados verticales
            blocked_v = (
                (y - 1 < 0 or _sdata[_idx(x, y-1)] == '#') or
                (y + 1 >= _nrows or _sdata[_idx(x, y+1)] == '#')
            )

            if blocked_h and blocked_v:
                dead.add((x, y))

    return dead


def _is_freeze_deadlock(x, y, state, visited=None):
    """
    Detecta si la caja en (x,y) está congelada (no puede moverse
    en ningún eje). Si está congelada fuera de un objetivo es
    deadlock inmediato. Si está sobre un objetivo, revisa si las
    cajas que la bloquean también están congeladas (deadlock en cadena).
    """
    if visited is None:
        visited = set()
    if (x, y) in visited:
        return False
    visited.add((x, y))

    on_goal = _sdata[_idx(x, y)] == '.'

    blocked_h = any(
        _sdata[_idx(x + dx, y)] == '#' or state[_idx(x + dx, y)] == '*'
        for dx in [-1, 1] if 0 <= x + dx < _ncols
    )
    blocked_v = any(
        _sdata[_idx(x, y + dy)] == '#' or state[_idx(x, y + dy)] == '*'
        for dy in [-1, 1] if 0 <= y + dy < _nrows
    )

    if not (blocked_h and blocked_v):
        return False
    if not on_goal:
        return True

    for dx in [-1, 1]:
        nx = x + dx
        if 0 <= nx < _ncols and state[_idx(nx, y)] == '*':
            if _is_freeze_deadlock(nx, y, state, visited):
                return True
    for dy in [-1, 1]:
        ny = y + dy
        if 0 <= ny < _nrows and state[_idx(x, ny)] == '*':
            if _is_freeze_deadlock(x, ny, state, visited):
                return True

    return False


def has_deadlock(state):
    """
    Función principal exportable.
    Retorna True si el estado tiene algún deadlock, False si no.

    Verifica en orden:
      1. Dead Square (O(1) por caja, muy rápido)
      2. Freeze deadlock (más costoso, solo si pasa el paso 1)

    Parámetros:
        state : string del estado dinámico actual
    """
    for y in range(_nrows):
        for x in range(_ncols):
            if state[_idx(x, y)] == '*':
                if (x, y) in _dead_squares:
                    return True
                if _is_freeze_deadlock(x, y, state):
                    return True
    return False


def get_dead_squares():
    """Retorna el conjunto de dead squares (útil para visualización o debug)."""
    return frozenset(_dead_squares)