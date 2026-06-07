"""
main.py
-------
Punto de entrada del proyecto Sokoban Solver.
Permite correr y comparar algoritmos sobre niveles individuales o en lote.

Uso:
    python main.py                         # corre todos los niveles con BFS
    python main.py --level levels/level4.txt
    python main.py --compare               # compara todos los algoritmos
"""

import argparse
import os
import tablero
from resultados.metricas import run, compare_all, print_result

# ── Importar algoritmos disponibles ─────────────────────────
from algoritmos.BFS import solve as bfs_solve
from algoritmos.DFS import solve as dfs_solve

# Cuando implementen los demás, descomentar:
# from algorithms.ucs    import solve as ucs_solve
# from algorithms.astar  import solve as astar_solve

# ── Registro de solvers disponibles ─────────────────────────
SOLVERS = {
    "BFS": bfs_solve,
    "DFS": dfs_solve,
    # "UCS":   ucs_solve,
    # "A*":    astar_solve,
}

# ── Niveles disponibles ──────────────────────────────────────
LEVELS_DIR = "niveles"
LEVEL_FILES = [
    os.path.join(LEVELS_DIR, f"level{i}.txt") for i in range(1, 6)
]
LEVEL_NUMBERS = list(range(1, 6))


def run_single(level_path, level_number=None):
    """Corre todos los solvers sobre un nivel y muestra resultados."""
    print(f"\nNivel: {level_path}")
    with open(level_path) as f:
        tablero.init(f.read())
    tablero.print_board(tablero.ddata)

    for name, solver in SOLVERS.items():
        tablero.init(level_path)
        result = run(solver, level_path, level_number)
        print_result(result, algorithm_name=name)


def run_compare():
    """Corre todos los solvers sobre todos los niveles y muestra tabla comparativa."""
    available = [l for l in LEVEL_FILES if os.path.exists(l)]
    numbers = LEVEL_NUMBERS[:len(available)]
    compare_all(available, SOLVERS, numbers)


def main():
    parser = argparse.ArgumentParser(description="Sokoban Solver")
    parser.add_argument(
        "--level", type=str, default=None,
        help="Path al archivo del nivel (ej: levels/level1.txt)"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compara todos los algoritmos en todos los niveles"
    )
    parser.add_argument(
        "--level-number", type=int, default=None,
        help="Número del nivel para comparar con el paper (1-5)"
    )
    args = parser.parse_args()

    if args.compare:
        run_compare()
    elif args.level:
        if not os.path.exists(args.level):
            print(f"Error: no se encontró el archivo '{args.level}'")
            return
        run_single(args.level, args.level_number)
    else:
        # Por defecto: corre el level4 (el más interesante del paper)
        default_level = os.path.join(LEVELS_DIR, "level4.txt")
        if os.path.exists(default_level):
            run_single(default_level, level_number=4)
        else:
            print("No se encontró el nivel por defecto. Usa --level <path>")


if __name__ == "__main__":
    main()
