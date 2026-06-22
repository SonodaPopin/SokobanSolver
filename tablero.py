# tablero.py
# Logica central del tablero de Sokoban. Todos los algoritmos importan
# este modulo para manipular el estado del juego.
#
# Notacion:
#   #  pared
#   ' ' espacio libre
#   .  objetivo
#   @  jugador
#   $  caja
#   *  caja sobre objetivo
#   +  jugador sobre objetivo

nrows = 0
ncols = 0
sdata = ""      # mapa estatico: paredes y objetivos
ddata = ""      # estado dinamico inicial: jugador y cajas
start_x = 0
start_y = 0

_goal_indices = None
_goal_indices_key = None


def idx(x, y):
    return y * ncols + x


def init(nivel):
    """Inicializa el tablero a partir de un string o path a archivo .txt."""
    global nrows, ncols, sdata, ddata, start_x, start_y, _goal_indices, _goal_indices_key
    _goal_indices = None
    _goal_indices_key = None
    sdata = ""
    ddata = ""

    if nivel.strip().startswith("#") or "\n" in nivel:
        raw = nivel
    else:
        with open(nivel, "r") as f:
            raw = f.read()

    data = list(filter(None, raw.splitlines()))
    ncols = max(len(r) for r in data)
    nrows = len(data)

    maps = {' ': ' ', '.': '.', '@': ' ', '#': '#',
            '$': ' ', '*': '.', '+': '.'}
    mapd = {' ': ' ', '.': ' ', '@': '@', '#': ' ',
            '$': '*', '*': '*', '+': '@'}

    for r, row in enumerate(data):
        row = row.ljust(ncols)
        for c, ch in enumerate(row):
            sdata += maps.get(ch, ' ')
            ddata += mapd.get(ch, ' ')
            if ch in ('@', '+'):
                start_x, start_y = c, r


def push(x, y, dx, dy, state):
    """Empuja la caja adyacente en direccion (dx, dy). Retorna None si no es valido."""
    nx, ny = x + dx, y + dy
    nnx, nny = x + 2*dx, y + 2*dy

    if sdata[idx(nnx, nny)] == '#' or state[idx(nnx, nny)] != ' ':
        return None

    d2 = bytearray(state.encode())
    d2[idx(x, y)] = ord(' ')
    d2[idx(nx, ny)] = ord('@')
    d2[idx(nnx, nny)] = ord('*')
    return d2.decode()


def move_player(x, y, dx, dy, state):
    """Mueve el jugador a la casilla adyacente sin empujar caja."""
    nx, ny = x + dx, y + dy

    if sdata[idx(nx, ny)] == '#' or state[idx(nx, ny)] != ' ':
        return None

    d2 = bytearray(state.encode())
    d2[idx(x, y)] = ord(' ')
    d2[idx(nx, ny)] = ord('@')
    return d2.decode()


def is_solved(state):
    """True cuando todas las cajas estan sobre objetivos. Cachea los indices de objetivo."""
    global _goal_indices, _goal_indices_key
    key = id(sdata)
    if _goal_indices_key != key:
        _goal_indices = frozenset(i for i, c in enumerate(sdata) if c == '.')
        _goal_indices_key = key

    box_count = 0
    for i in _goal_indices:
        if state[i] != '*':
            return False
        box_count += 1
    return state.count('*') == box_count


def get_cajas(state):
    return {(x, y) for y in range(nrows) for x in range(ncols)
            if state[idx(x, y)] == '*'}


def get_player(state):
    for y in range(nrows):
        for x in range(ncols):
            if state[idx(x, y)] == '@':
                return x, y
    return None


def load_level(path):
    init(path)


def print_tablero(state):
    """Imprime el tablero combinando mapa estatico y estado dinamico."""
    for y in range(nrows):
        row = ""
        for x in range(ncols):
            s = sdata[idx(x, y)]
            d = state[idx(x, y)]
            if d == '@':
                row += '+' if s == '.' else '@'
            elif d == '*':
                row += '*' if s == '.' else '$'
            else:
                row += s
        print(row)
