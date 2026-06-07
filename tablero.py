"""
tablero.py
--------
Lógica central del tablero de Sokoban.
Todos los algoritmos importan este módulo para manipular el estado del juego.

Notación del tablero:
    #  → pared
    ' '→ espacio libre
    .  → objetivo
    @  → jugador
    $  → caja
    *  → caja sobre objetivo
    +  → jugador sobre objetivo
"""

# ── Estado global del tablero ────────────────────────────────
nrows = 0       # número de filas
ncols = 0       # número de columnas
sdata = ""      # mapa estático: paredes (#) y objetivos (.)
ddata = ""      # estado inicial dinámico: jugador (@) y cajas (*)
start_x = 0     # posición inicial X del jugador
start_y = 0     # posición inicial Y del jugador


def idx(x, y):
    """Convierte coordenadas (x, y) a índice lineal del tablero."""
    return y * ncols + x


def init(board):
    """
    Inicializa el tablero a partir de un string o path a archivo .txt.

    Parámetros:
        board : string del nivel o path a archivo .txt
    """
    global nrows, ncols, sdata, ddata, start_x, start_y
    sdata = ""
    ddata = ""

    # Aceptar string directo o path a archivo
    if board.strip().startswith("#") or "\n" in board:
        raw = board
    else:
        with open(board, "r") as f:
            raw = f.read()

    data = list(filter(None, raw.splitlines()))
    ncols = max(len(r) for r in data)
    nrows = len(data)

    # maps: extrae el mapa estático (paredes y objetivos)
    # mapd: extrae el estado dinámico (jugador y cajas)
    maps = {' ': ' ', '.': '.', '@': ' ', '#': '#',
            '$': ' ', '*': '.', '+': '.'}
    mapd = {' ': ' ', '.': ' ', '@': '@', '#': ' ',
            '$': '*', '*': '*', '+': '@'}

    for r, row in enumerate(data):
        for c, ch in enumerate(row):
            sdata += maps.get(ch, ' ')
            ddata += mapd.get(ch, ' ')
            if ch in ('@', '+'):
                start_x, start_y = c, r


def push(x, y, dx, dy, state):
    """
    Intenta empujar la caja adyacente en dirección (dx, dy).
    Retorna el nuevo estado como string, o None si el movimiento es inválido.
    """
    nx, ny = x + dx, y + dy       # posición de la caja
    nnx, nny = x + 2*dx, y + 2*dy  # posición destino de la caja

    if (sdata[idx(nnx, nny)] == '#' or state[idx(nnx, nny)] != ' '):
        return None

    d2 = bytearray(state.encode())
    d2[idx(x, y)]     = ord(' ')
    d2[idx(nx, ny)]   = ord('@')
    d2[idx(nnx, nny)] = ord('*')
    return d2.decode()


def move_player(x, y, dx, dy, state):
    """
    Mueve el jugador a la casilla adyacente (sin empujar caja).
    Retorna el nuevo estado, o None si el movimiento es inválido.
    """
    nx, ny = x + dx, y + dy

    if (sdata[idx(nx, ny)] == '#' or state[idx(nx, ny)] != ' '):
        return None

    d2 = bytearray(state.encode())
    d2[idx(x, y)]   = ord(' ')
    d2[idx(nx, ny)] = ord('@')
    return d2.decode()


def is_solved(state):
    """Retorna True cuando todas las cajas están sobre objetivos."""
    return all(
        (sdata[i] == '.') == (state[i] == '*')
        for i in range(len(state))
    )


def get_boxes(state):
    """Retorna un set con las posiciones (x, y) de todas las cajas."""
    return {(x, y)
            for y in range(nrows)
            for x in range(ncols)
            if state[idx(x, y)] == '*'}


def get_player(state):
    """Retorna la posición (x, y) del jugador en el estado dado."""
    for y in range(nrows):
        for x in range(ncols):
            if state[idx(x, y)] == '@':
                return x, y
    return None


def load_level(path):
    """Carga e inicializa un nivel desde un archivo .txt."""
    init(path)


def print_board(state):
    """Imprime el tablero combinando mapa estático y estado dinámico."""
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
