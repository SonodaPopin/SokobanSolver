"""
results/metricas.py
------------------
Módulo de medición y comparación de algoritmos para Sokoban.
Mide tiempo, estados explorados y pasos de solución,
y los compara contra los resultados del paper (Tabla 1).

Uso:
    from results.metricas import run, compare_all, print_table

    result = run(solve_fn, level_path)
    compare_all(level_path)
"""

import time
import board

# ── Resultados del paper (Tabla 1) ───────────────────────────
# Formato: {nivel: {algoritmo: (tiempo_ms, estados, pasos)}}
PAPER_RESULTS = {
    1: {
        "A* Eucl. min":    (230.5,  733,   None),
        "A* Manh. min":    (222.6,  622,   None),
        "A* Hung. Eucl":   (326.1,  678,   None),
        "A* Hung. Manh":   (274.8,  583,   None),
    },
    2: {
        "A* Eucl. min":    (1549.5, 4874,  None),
        "A* Manh. min":    (1438.0, 4435,  None),
        "A* Hung. Eucl":   (1948.1, 4792,  None),
        "A* Hung. Manh":   (1849.0, 4499,  None),
    },
    3: {
        "A* Eucl. min":    (389.5,  1138,  None),
        "A* Manh. min":    (169.0,  344,   None),
        "A* Hung. Eucl":   (929.6,  713,   None),
        "A* Hung. Manh":   (270.7,  294,   None),
    },
    4: {
        "A* Eucl. min":    (8098.6, 13913, None),
        "A* Manh. min":    (6064.3, 9978,  None),
        "A* Hung. Eucl":   (8657.6, 13394, None),
        "A* Hung. Manh":   (6572.6, 9800,  None),
    },
    5: {
        "A* Eucl. min":    (2178.7, 6162,  None),
        "A* Manh. min":    (1541.5, 4087,  None),
        "A* Hung. Eucl":   (3116.3, 5501,  None),
        "A* Hung. Manh":   (2153.2, 3702,  None),
    },
}

# Pasos óptimos (oracle) para los niveles 1-5
ORACLE_STEPS = {1: 33, 2: 43, 3: 57, 4: 82, 5: 51}


def run(solve_fn, level, level_number=None):
    """
    Ejecuta un algoritmo sobre un nivel y retorna sus métricas.

    Parámetros:
        solve_fn     : función solve() del algoritmo (debe retornar (sol, estados))
        level        : string del nivel o path a archivo .txt
        level_number : número del nivel (para comparar con el paper)

    Retorna dict con:
        solution  : string de movimientos
        steps     : número de pasos
        states    : estados explorados
        time_ms   : tiempo en milisegundos
        solved    : True/False
    """
    board.init(level)

    start = time.time()
    solution, states = solve_fn()
    elapsed_ms = round((time.time() - start) * 1000, 1)

    solved = solution != "No solution"
    steps = len(solution) if solved else 0

    result = {
        "solution": solution,
        "steps":    steps,
        "states":   states,
        "time_ms":  elapsed_ms,
        "solved":   solved,
    }

    if level_number and level_number in ORACLE_STEPS:
        result["oracle_steps"] = ORACLE_STEPS[level_number]
        result["steps_vs_oracle"] = (
            f"+{steps - ORACLE_STEPS[level_number]}"
            if solved else "N/A"
        )

    return result


def compare_all(levels, solvers, level_numbers=None):
    """
    Corre todos los solvers sobre todos los niveles y muestra una tabla comparativa.

    Parámetros:
        levels       : lista de strings o paths de niveles
        solvers      : dict {nombre: función solve()}
        level_numbers: lista de números de nivel (para mostrar datos del paper)

    Ejemplo:
        compare_all(
            levels=["levels/level1.txt", "levels/level2.txt"],
            solvers={"BFS": bfs_solve, "A*": astar_solve},
            level_numbers=[1, 2]
        )
    """
    if level_numbers is None:
        level_numbers = [None] * len(levels)

    for i, (level, level_num) in enumerate(zip(levels, level_numbers)):
        print(f"\n{'='*60}")
        print(f"  Nivel {level_num if level_num else i+1}")
        print(f"{'='*60}")
        print(f"  {'Algoritmo':<20} {'Tiempo (ms)':>12} {'Estados':>10} {'Pasos':>7}")
        print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*7}")

        for name, solve_fn in solvers.items():
            try:
                r = run(solve_fn, level, level_num)
                status = "✓" if r["solved"] else "✗"
                print(f"  {status} {name:<19} {r['time_ms']:>12.1f} "
                      f"{r['states']:>10} {r['steps']:>7}")
            except Exception as e:
                print(f"  ✗ {name:<19} {'ERROR':>12} {'---':>10} {'---':>7}  ({e})")

        # Mostrar datos del paper si disponibles
        if level_num and level_num in PAPER_RESULTS:
            print(f"\n  --- Paper (referencia) ---")
            for name, (t, s, _) in PAPER_RESULTS[level_num].items():
                print(f"    {name:<20} {t:>12.1f} {s:>10} {'N/A':>7}")

        if level_num and level_num in ORACLE_STEPS:
            print(f"\n  Oracle (pasos óptimos): {ORACLE_STEPS[level_num]}")


def print_result(result, algorithm_name=""):
    """Imprime el resultado de un único run de forma legible."""
    print(f"\n{'─'*40}")
    if algorithm_name:
        print(f"  Algoritmo : {algorithm_name}")
    print(f"  Resuelto  : {'Sí' if result['solved'] else 'No'}")
    print(f"  Tiempo    : {result['time_ms']} ms")
    print(f"  Estados   : {result['states']}")
    print(f"  Pasos     : {result['steps']}")
    if "oracle_steps" in result:
        print(f"  Oracle    : {result['oracle_steps']} pasos")
        print(f"  Diferencia: {result['steps_vs_oracle']}")
    if result["solved"]:
        print(f"  Solución  : {result['solution']}")
    print(f"{'─'*40}")
