import argparse
import os
import time
import tablero
from resultados.metricas import (
    run, compare_all, print_result,
    save_results, load_results, clear_results
)

from algoritmos.BFS import solve as bfs_solve
from algoritmos.DFS import solve as dfs_solve
from algoritmos.UCS import solve_cost1 as ucs_cost1_solve
from algoritmos.UCS import solve_cost2 as ucs_cost2_solve
from algoritmos.AStar import (
    solve_euclidean_min as astar_eucl,
    solve_manhattan_min as astar_manh,
    solve_hungarian_euclidean as astar_hung_eucl,
    solve_hungarian_manhattan as astar_hung_manh,
    solve_hungarian_manhattan_deadlock as astar_hung_manh_dl,
)

SOLVERS = {
    "BFS": bfs_solve,
    "DFS": dfs_solve,
    "UCS Cost1": ucs_cost1_solve,
    "UCS Cost2": ucs_cost2_solve,
    "A* Eucl": astar_eucl,
    "A* Manh": astar_manh,
    "A* Hung+Eucl": astar_hung_eucl,
    "A* Hung+Manh": astar_hung_manh,
    "A* Hung+Manh+DL": astar_hung_manh_dl,
}

# La CNN solo se agrega si hay un modelo entrenado guardado.
# Entrenar con: python cnn/entrenar.py
_modelo_cnn_path = os.path.join("cnn", "modelo_bfs.pt")
if os.path.exists(_modelo_cnn_path):
    from cnn.predecir import solve as cnn_solve
    SOLVERS["CNN"] = cnn_solve

LEVELS_DIR = "niveles"
LEVEL_FILES = [os.path.join(LEVELS_DIR, f"level{i}.txt") for i in range(1, 26)]
LEVEL_NUMBERS = list(range(1, 26))


def run_single(level_path, level_number=None, save=False, timeout=30, verbose=False):
    print(f"\nNivel: {level_path}")
    with open(level_path) as f:
        tablero.init(f.read())
    tablero.print_tablero(tablero.ddata)

    resultados = []
    for name, solver in SOLVERS.items():
        if verbose:
            print(f"\nCorriendo {name} en {level_path} (timeout={timeout}s)...", flush=True)
        tablero.init(level_path)

        start = time.time()
        result = run(solver, level_path, level_number, timeout=timeout)
        elapsed = round(time.time() - start, 1)

        if verbose:
            status = "TIMEOUT" if result.get("timed_out") else "OK" if result["solved"] else "SIN SOLUCION"
            print(f"{name} terminó en {elapsed}s -> {status}", flush=True)

        print_result(result, algorithm_name=name)
        result["algorithm"] = name
        result["level"] = level_path
        resultados.append(result)

    if save:
        save_results(resultados)


def run_compare(save=False, timeout=30, verbose=False):
    available = [l for l in LEVEL_FILES if os.path.exists(l)]
    numbers = LEVEL_NUMBERS[:len(available)]
    if verbose:
        total = len(available) * len(SOLVERS)
        print(f"Plan: {len(available)} niveles x {len(SOLVERS)} algoritmos = {total} corridas (timeout={timeout}s c/u)")
    compare_all(available, SOLVERS, numbers, save=save, timeout=timeout, verbose=verbose)


def show_history():
    data = load_results()
    if not data:
        return
    print(f"\n{'Fecha':<20} {'Algoritmo':<12} {'Nivel':<20} "
          f"{'Resuelto':<9} {'Tiempo(ms)':>11} {'Estados':>10} {'Pasos':>7}")
    print("-" * 95)
    for r in data:
        ts = r.get("timestamp", "")[:19]
        alg = r.get("algorithm", "?")
        lvl = os.path.basename(r.get("level", "?"))
        solved = "Si" if r.get("solved") else "No"
        t = r.get("time_ms", 0)
        s = r.get("states", 0)
        p = r.get("steps", 0)
        print(f"{ts:<20} {alg:<12} {lvl:<20} {solved:<9} {t:>11.1f} {s:>10} {p:>7}")
    print(f"\nTotal de runs guardados: {len(data)}")


def main():
    parser = argparse.ArgumentParser(description="Sokoban Solver")
    parser.add_argument("--level", type=str, default=None, help="Path al archivo del nivel")
    parser.add_argument("--compare", action="store_true", help="Compara todos los algoritmos en todos los niveles")
    parser.add_argument("--level-number", type=int, default=None, help="Numero del nivel para comparar con el paper (1-5)")
    parser.add_argument("--save", action="store_true", help="Guarda resultados en resultados/run_history.json y .csv")
    parser.add_argument("--history", action="store_true", help="Muestra el historico de resultados guardados")
    parser.add_argument("--clear-history", action="store_true", help="Borra el historico de resultados guardados")
    parser.add_argument("--timeout", type=int, default=180, help="Segundos maximos por algoritmo (default: 180)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Muestra progreso en vivo")
    args = parser.parse_args()

    if args.clear_history:
        clear_results()
    elif args.history:
        show_history()
    elif args.compare:
        run_compare(save=args.save, timeout=args.timeout, verbose=args.verbose)
    elif args.level:
        if not os.path.exists(args.level):
            print(f"Error: no se encontró el archivo '{args.level}'")
            return
        run_single(args.level, args.level_number, save=args.save,
                   timeout=args.timeout, verbose=args.verbose)
    else:
        default_level = os.path.join(LEVELS_DIR, "level4.txt")
        if os.path.exists(default_level):
            run_single(default_level, level_number=4, save=args.save,
                       timeout=args.timeout, verbose=args.verbose)
        else:
            print("No se encontró el nivel por defecto. Usa --level <path>")


if __name__ == "__main__":
    main()
