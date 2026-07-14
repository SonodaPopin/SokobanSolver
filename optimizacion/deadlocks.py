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

def init_deadlocks(sdata, nrows, ncols):
    global _sdata, _nrows, _ncols, _dead_squares
    _sdata = sdata
    _nrows = nrows
    _ncols = ncols
    _dead_squares = _compute_corner_deadsquares()
    _dead_squares |= _compute_dead_lines()


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

            wall_l = _es_pared(x - 1, y)
            wall_r = _es_pared(x + 1, y)
            wall_u = _es_pared(x, y - 1)
            wall_d = _es_pared(x, y + 1)

            if (wall_l or wall_r) and (wall_u or wall_d):
                dead.add((x, y))

    return dead

def _scan_line_runs(length, is_wall_at, is_touching_wall, is_goal_at):
    """
    Escanea una linea (una fila o una columna) buscando tramos donde:
      - ninguna celda es pared
      - TODAS las celdas tocan la misma pared perpendicular
        (ej: pared arriba en todo el tramo, para una fila)
      - el tramo esta acotado por pared real en AMBOS extremos
      - no hay ningun objetivo en el tramo

    Retorna una lista de tramos (cada uno, lista de indices) que
    cumplen las 4 condiciones -> cualquier caja ahi es deadlock.
    """
    runs = []
    i = 0
    while i < length:
        if is_wall_at(i):
            i += 1
            continue

        run = []
        while i < length and not is_wall_at(i) and is_touching_wall(i):
            run.append(i)
            i += 1

        if run:
            end = run[-1]
            start = run[0]
            left_bound  = is_wall_at(start - 1)
            right_bound = is_wall_at(end + 1)
            has_goal = any(is_goal_at(j) for j in run)
            if left_bound and right_bound and not has_goal:
                runs.append(run)
        else:
            i += 1

    return runs


def _compute_dead_lines():
    dead = set()
    # Tramos horizontales: pared arriba o abajo
    for wall_side in ('u', 'd'):
        for y in range(_nrows):
            def is_wall_at(x, y=y):
                return _es_pared(x, y)

            def is_touching(x, y=y, side=wall_side):
                return _es_pared(x, y - 1) if side == 'u' else _es_pared(x, y + 1)

            def is_goal_at(x, y=y):
                return _es_objetivo(x, y)

            for run in _scan_line_runs(_ncols, is_wall_at, is_touching, is_goal_at):
                dead.update((x, y) for x in run)

    # Tramos verticales: pared a la izquierda o derecha 
    for wall_side in ('l', 'r'):
        for x in range(_ncols):
            def is_wall_at(y, x=x):
                return _es_pared(x, y)

            def is_touching(y, x=x, side=wall_side):
                return _es_pared(x - 1, y) if side == 'l' else _es_pared(x + 1, y)

            def is_goal_at(y, x=x):
                return _es_objetivo(x, y)

            for run in _scan_line_runs(_nrows, is_wall_at, is_touching, is_goal_at):
                dead.update((x, y) for y in run)

    return dead

def _hay_caja(x, y, state):
    return 0 <= x < _ncols and 0 <= y < _nrows and state[_idx(x, y)] == '*'


def _pared_o_congelada(nx, ny, state, memo_x, memo_y, en_curso):
    # comprueba si es una pared o es una caja permanentemente bloqueada
    if _es_pared(nx, ny):
        return True
    if _hay_caja(nx, ny, state):
        bloqueada_h = _bloqueada_x(nx, ny, state, memo_x, memo_y, en_curso)
        bloqueada_v = _bloqueada_y(nx, ny, state, memo_x, memo_y, en_curso)
        return bloqueada_h and bloqueada_v
    return False


def _bloqueada_x(bx, by, state, memo_x, memo_y, en_curso):
    key = (bx, by)
    if key in memo_x:
        return memo_x[key]

    stack_key = (bx, by, 'x')
    if stack_key in en_curso:
        return True

    en_curso.add(stack_key)
    resultado = (
        _pared_o_congelada(bx - 1, by, state, memo_x, memo_y, en_curso) or
        _pared_o_congelada(bx + 1, by, state, memo_x, memo_y, en_curso)
    )
    en_curso.discard(stack_key)
    memo_x[key] = resultado
    return resultado


def _bloqueada_y(bx, by, state, memo_x, memo_y, en_curso):
    key = (bx, by)
    if key in memo_y:
        return memo_y[key]

    stack_key = (bx, by, 'y')
    if stack_key in en_curso:
        return True

    en_curso.add(stack_key)
    resultado = (
        _pared_o_congelada(bx, by - 1, state, memo_x, memo_y, en_curso) or
        _pared_o_congelada(bx, by + 1, state, memo_x, memo_y, en_curso)
    )
    en_curso.discard(stack_key)
    memo_y[key] = resultado
    return resultado


def _hay_freeze_deadlock(state, cajas):
    """
    Revisa si alguna caja que NO esta sobre un objetivo quedo
    bloqueada en ambos ejes de forma permanente.
    """
    if not cajas:
        return False

    memo_x, memo_y = {}, {}
    for bx, by in cajas:
        if _es_objetivo(bx, by):
            continue
        en_curso = set()
        if (_bloqueada_x(bx, by, state, memo_x, memo_y, en_curso) and _bloqueada_y(bx, by, state, memo_x, memo_y, en_curso)):
            return True
    return False


def has_deadlock(state):
    """
    Retorna True si el estado tiene algun deadlock detectado.

    1. Esquinas simples + lineas muertas contra pared (baratas,
    precomputadas al cargar el nivel.
    2. Freeze deadlock por ejes (mas caro):
    solo usa pared y el estado de otras cajas.
    """
    cajas = []
    for i, ch in enumerate(state):
        if ch != '*':
            continue
        x, y = i % _ncols, i // _ncols
        if (x, y) in _dead_squares:
            return True
        cajas.append((x, y))

    return _hay_freeze_deadlock(state, cajas)


def get_dead_squares():
    """Retorna el conjunto de esquinas/lineas muertas (para debug/visualizacion)."""
    return frozenset(_dead_squares)