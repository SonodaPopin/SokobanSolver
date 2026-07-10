# optimizacion/deadlocks.py
# Modulo de deteccion de deadlocks para Sokoban.
#
# Implementa tres chequeos, de mas barato a mas caro:
#   1. Esquinas simples: una caja en una esquina real (pared en un
#      lado horizontal Y un lado vertical) nunca puede salir de ahi.
#   2. Lineas muertas: si una caja esta pegada a una pared y todo
#      el tramo continuo de esa pared no tiene ningun objetivo,
#      la caja no podra llegar a ningun objetivo moviendose por ahi.
#   3. Freeze deadlock: una caja esta "congelada" si no puede ser
#      empujada en ninguna direccion.
#
# Uso:
#   init_deadlocks(sdata, nrows, ncols)   # una vez al cargar el nivel
#   has_deadlock(state)                   # por cada estado nuevo

_sdata = ""
_nrows = 0
_ncols = 0
_dead_squares = set()
_dead_lines = set()


def init_deadlocks(sdata, nrows, ncols):
    global _sdata, _nrows, _ncols, _dead_squares, _dead_lines
    _sdata = sdata
    _nrows = nrows
    _ncols = ncols
    _dead_squares = _compute_corner_deadsquares()
    _dead_lines = _compute_dead_lines()


def _idx(x, y):
    return y * _ncols + x


def _es_pared(x, y):
    if not (0 <= x < _ncols and 0 <= y < _nrows):
        return True
    return _sdata[_idx(x, y)] == '#'


def _es_objetivo(x, y):
    if not (0 <= x < _ncols and 0 <= y < _nrows):
        return False
    return _sdata[_idx(x, y)] == '.'


def _compute_corner_deadsquares():
    """
    Esquinas simples: una caja en (x,y) que no es objetivo y tiene
    pared en al menos un lado horizontal Y al menos un lado vertical
    queda atrapada para siempre.
    """
    dead = set()
    for y in range(_nrows):
        for x in range(_ncols):
            if _es_pared(x, y) or _es_objetivo(x, y):
                continue

            wall_l = _es_pared(x-1, y)
            wall_r = _es_pared(x+1, y)
            wall_u = _es_pared(x, y-1)
            wall_d = _es_pared(x, y+1)

            if (wall_l or wall_r) and (wall_u or wall_d):
                dead.add((x, y))

    return dead


def _compute_dead_lines():
    """
    Deteccion de lineas muertas desactivada temporalmente.
    La logica de tramos pegados a pared genera falsos positivos
    en celdas aisladas entre dos paredes laterales que en realidad
    son transitables verticalmente (o viceversa). Requiere una
    reformulacion mas cuidadosa para distinguir un pasillo real
    de una celda con paredes incidentales en ambos lados.
    """
    return set()


def _puede_ser_empujada(bx, by, dx, dy, state):
    """
    Verifica si la caja en (bx,by) puede ser empujada en direccion
    (dx,dy): el destino debe estar libre y el lado opuesto (donde
    se para el jugador) tambien debe estar libre.
    """
    tx, ty = bx + dx, by + dy
    jx, jy = bx - dx, by - dy

    if _es_pared(tx, ty):
        return False
    if 0 <= tx < _ncols and 0 <= ty < _nrows and state[_idx(tx, ty)] == '*':
        return False
    if _es_pared(jx, jy):
        return False
    if 0 <= jx < _ncols and 0 <= jy < _nrows and state[_idx(jx, jy)] == '*':
        return False
    return True


def _is_freeze_deadlock(bx, by, state, visited=None):
    """
    Una caja esta bloqueada si no puede ser empujada en ninguna
    direccion en este instante. Eso solo es un deadlock PERMANENTE
    si las cajas que la bloquean tambien estan bloqueadas de forma
    permanente (si alguna vecina bloqueante SI puede moverse,
    el bloqueo actual es temporal y no es deadlock).
    """
    if visited is None:
        visited = set()
    if (bx, by) in visited:
        # ciclo de cajas bloqueandose mutuamente sin salida: deadlock
        return True
    visited.add((bx, by))

    puede_moverse = (
        _puede_ser_empujada(bx, by, 1, 0, state) or
        _puede_ser_empujada(bx, by, -1, 0, state) or
        _puede_ser_empujada(bx, by, 0, 1, state) or
        _puede_ser_empujada(bx, by, 0, -1, state)
    )

    if puede_moverse:
        return False

    # bloqueada en este instante: identificar que la bloquea en cada eje
    # y revisar si esas cajas bloqueantes pueden moverse (liberandola)
    bloqueantes = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = bx + dx, by + dy
        if 0 <= nx < _ncols and 0 <= ny < _nrows and state[_idx(nx, ny)] == '*':
            bloqueantes.append((nx, ny))

    if not bloqueantes:
        # bloqueada solo por paredes/bordes, sin cajas vecinas de por
        # medio: nunca podra moverse, es deadlock permanente
        return True

    # si CUALQUIER caja bloqueante puede liberar el bloqueo (no esta
    # ella misma congelada), entonces este bloqueo es temporal
    for nx, ny in bloqueantes:
        if not _is_freeze_deadlock(nx, ny, state, visited):
            return False

    # todas las cajas bloqueantes estan tambien permanentemente congeladas
    return True


def has_deadlock(state):
    """
    Retorna True si el estado tiene algun deadlock detectado.

    Solo se usa deteccion de esquinas (la mas confiable y sin falsos
    positivos: una caja en una esquina real, sin objetivo, nunca puede
    salir de ahi sin importar el resto del tablero).

    El freeze deadlock y las lineas muertas quedaron desactivados:
    en las pruebas generaban falsos positivos en estados intermedios
    validos (p. ej. una caja momentaneamente bloqueada por otra caja
    que se puede mover despues), lo que hacia fallar niveles que si
    tenian solucion. Detectar correctamente esos casos requeriria
    verificar dependencias dinamicas entre cajas, que es esencialmente
    parte de lo que hace dificil resolver Sokoban en primer lugar.
    """
    for i, ch in enumerate(state):
        if ch != '*':
            continue
        x, y = i % _ncols, i // _ncols
        if (x, y) in _dead_squares:
            return True
    return False


def get_dead_squares():
    """Retorna el conjunto combinado de esquinas y lineas muertas (para debug/visualizacion)."""
    return frozenset(_dead_squares | _dead_lines)