"""
optimizations/tunnel_macros.py
-------------------------------
Implementación de Tunnel Macros para Sokoban.

Un túnel es un pasillo donde una caja solo puede moverse en una dirección
(no tiene espacio para salir por los lados). En esos casos, en vez de
calcular cada paso del túnel individualmente, se colapsa toda la secuencia
en un único macro-movimiento, reduciendo la profundidad efectiva del árbol.

Uso:
    from optimizations.tunnel_macros import find_tunnels, apply_tunnel_macro

    tunnels = find_tunnels()          # llamar una vez al inicio del nivel
    result = apply_tunnel_macro(x, y, dx, dy, state, tunnels)
"""

import tablero


def find_tunnels():
    """
    Precalcula todos los túneles del nivel actual.
    Un túnel en dirección (dx, dy) desde (x, y) existe cuando:
      - La casilla es libre (no pared)
      - Ambos lados perpendiculares están bloqueados por paredes

    Retorna un dict: {(x, y, dx, dy): largo_del_tunel}
    """
    tunnels = {}
    dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    for y in range(tablero.nrows):
        for x in range(tablero.ncols):
            if tablero.sdata[tablero.idx(x, y)] == '#':
                continue

            for dx, dy in dirs:
                # Verificar que la casilla siguiente en la dirección es libre
                nx, ny = x + dx, y + dy
                if not (0 <= nx < tablero.ncols and 0 <= ny < tablero.nrows):
                    continue
                if tablero.sdata[tablero.idx(nx, ny)] == '#':
                    continue

                # Verificar que los lados perpendiculares están bloqueados
                # (eso define que es un túnel)
                perp = [(-dy, dx), (dy, -dx)]  # direcciones perpendiculares
                if all(
                    tablero.sdata[tablero.idx(x + px, y + py)] == '#'
                    for px, py in perp
                    if 0 <= x + px < tablero.ncols and 0 <= y + py < tablero.nrows
                ):
                    # Calcular largo del túnel siguiendo la dirección
                    length = 0
                    cx, cy = nx, ny
                    while (0 <= cx < tablero.ncols and
                           0 <= cy < tablero.nrows and
                           tablero.sdata[tablero.idx(cx, cy)] != '#'):
                        # Verificar que sigue siendo túnel (lados bloqueados)
                        still_tunnel = all(
                            tablero.sdata[tablero.idx(cx + px, cy + py)] == '#'
                            for px, py in perp
                            if (0 <= cx + px < tablero.ncols and
                                0 <= cy + py < tablero.nrows)
                        )
                        if not still_tunnel:
                            break
                        length += 1
                        cx += dx
                        cy += dy

                    if length > 1:
                        tunnels[(x, y, dx, dy)] = length

    return tunnels


def apply_tunnel_macro(x, y, dx, dy, state, tunnels):
    """
    Si hay un túnel desde (x, y) en dirección (dx, dy),
    aplica el macro-movimiento completo: mueve la caja
    hasta el final del túnel en un solo paso.

    Retorna:
        (new_state, new_x, new_y, steps) si se aplicó el macro
        None si no hay túnel aplicable
    """
    key = (x, y, dx, dy)
    if key not in tunnels:
        return None

    length = tunnels[key]
    cur_state = state
    cur_x, cur_y = x, y

    for _ in range(length):
        new_state = tablero.push(cur_x, cur_y, dx, dy, cur_state)
        if new_state is None:
            break
        cur_state = new_state
        cur_x += dx
        cur_y += dy

    if cur_state == state:
        return None

    return cur_state, cur_x, cur_y, length


def get_tunnel_moves(x, y, dx, dy, tunnels):
    """
    Retorna el string de movimientos que representa el macro-movimiento
    (útil para construir la solución final con la notación del paper).

    Parámetros:
        x, y     : posición inicial del jugador
        dx, dy   : dirección del túnel
        tunnels  : dict de túneles precalculados
    """
    dir_map = {
        (0, -1): 'U',
        (1,  0): 'R',
        (0,  1): 'D',
        (-1, 0): 'L'
    }
    key = (x, y, dx, dy)
    if key not in tunnels:
        return ""
    return dir_map.get((dx, dy), '?') * tunnels[key]
