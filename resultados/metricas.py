# resultados/metricas.py
# Medicion, comparacion y guardado de resultados de los solvers.
# Los resultados se guardan en JSON y CSV para no tener que correr
# todo de nuevo cuando se necesiten para el informe o graficos.

import time
import json
import csv
import os
import threading
from datetime import datetime
import tablero

DEFAULT_TIMEOUT = 60

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(RESULTS_DIR, "run_history.json")
CSV_PATH = os.path.join(RESULTS_DIR, "run_history.csv")

# Resultados reportados en el paper (Tabla 1), para comparacion
PAPER_RESULTS = {
    1: {
        "A* Eucl. min":  (230.5,  733,   None),
        "A* Manh. min":  (222.6,  622,   None),
        "A* Hung. Eucl": (326.1,  678,   None),
        "A* Hung. Manh": (274.8,  583,   None),
    },
    2: {
        "A* Eucl. min":  (1549.5, 4874,  None),
        "A* Manh. min":  (1438.0, 4435,  None),
        "A* Hung. Eucl": (1948.1, 4792,  None),
        "A* Hung. Manh": (1849.0, 4499,  None),
    },
    3: {
        "A* Eucl. min":  (389.5,  1138,  None),
        "A* Manh. min":  (169.0,  344,   None),
        "A* Hung. Eucl": (929.6,  713,   None),
        "A* Hung. Manh": (270.7,  294,   None),
    },
    4: {
        "A* Eucl. min":  (8098.6, 13913, None),
        "A* Manh. min":  (6064.3, 9978,  None),
        "A* Hung. Eucl": (8657.6, 13394, None),
        "A* Hung. Manh": (6572.6, 9800,  None),
    },
    5: {
        "A* Eucl. min":  (2178.7, 6162,  None),
        "A* Manh. min":  (1541.5, 4087,  None),
        "A* Hung. Eucl": (3116.3, 5501,  None),
        "A* Hung. Manh": (2153.2, 3702,  None),
    },
}

ORACLE_STEPS = {1: 33, 2: 43, 3: 57, 4: 82, 5: 51}


def run(solve_fn, level, level_number=None, timeout=DEFAULT_TIMEOUT):
    """
    Ejecuta un algoritmo sobre un nivel y retorna sus metricas.
    Si tarda mas que 'timeout' segundos se aborta y se marca como timeout
    (para que --compare no se quede colgado en niveles pesados).
    """
    tablero.init(level)

    result_container = {}

    def target():
        try:
            solution, states = solve_fn()
            result_container["solution"] = solution
            result_container["states"] = states
        except Exception as e:
            result_container["error"] = str(e)

    start = time.time()
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    elapsed_ms = round((time.time() - start) * 1000, 1)

    if thread.is_alive():
        solution = "Timeout"
        states = -1
    elif "error" in result_container:
        solution = "No solution"
        states = 0
    else:
        solution = result_container.get("solution", "No solution")
        states = result_container.get("states", 0)

    solved = solution not in ("No solution", "Timeout")
    steps = len(solution) if solved else 0

    result = {
        "solution": solution if solved else "",
        "steps": steps,
        "states": states,
        "time_ms": elapsed_ms,
        "solved": solved,
        "timed_out": solution == "Timeout",
    }

    if level_number and level_number in ORACLE_STEPS:
        result["oracle_steps"] = ORACLE_STEPS[level_number]
        result["steps_vs_oracle"] = (
            steps - ORACLE_STEPS[level_number] if solved else None
        )

    return result


def save_results(results, json_path=JSON_PATH, csv_path=CSV_PATH, append=True):
    """Guarda una lista de resultados en JSON (historico completo) y CSV (tabular)."""
    timestamp = datetime.now().isoformat(timespec="seconds")
    for r in results:
        r["timestamp"] = timestamp

    existing = []
    if append and os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []

    all_results = existing + results
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    fieldnames = ["timestamp", "algorithm", "level", "solved",
                  "time_ms", "states", "steps",
                  "oracle_steps", "steps_vs_oracle"]

    file_exists = os.path.exists(csv_path)
    mode = "a" if (append and file_exists) else "w"

    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if mode == "w" or not file_exists:
            writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\nResultados guardados en:")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")


def load_results(json_path=JSON_PATH):
    if not os.path.exists(json_path):
        print(f"No se encontró {json_path}. Aún no hay resultados guardados.")
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def clear_results(json_path=JSON_PATH, csv_path=CSV_PATH):
    for p in (json_path, csv_path):
        if os.path.exists(p):
            os.remove(p)
    print("Histórico de resultados borrado.")


def compare_all(levels, solvers, level_numbers=None, save=False,
                 timeout=DEFAULT_TIMEOUT, verbose=False):
    """Corre todos los solvers sobre todos los niveles y muestra tabla comparativa."""
    if level_numbers is None:
        level_numbers = [None] * len(levels)

    all_results = []

    for i, (level, level_num) in enumerate(zip(levels, level_numbers)):
        print(f"\n{'='*60}")
        print(f"  Nivel {level_num if level_num else i+1}")
        print(f"{'='*60}")
        print(f"  {'Algoritmo':<20} {'Tiempo (ms)':>12} {'Estados':>10} {'Pasos':>7}")
        print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*7}")

        for name, solve_fn in solvers.items():
            if verbose:
                print(f"  -> corriendo {name} ...", end=" ", flush=True)
            try:
                t0 = time.time()
                r = run(solve_fn, level, level_num, timeout=timeout)
                real_elapsed = round(time.time() - t0, 1)

                if verbose:
                    tag = ("TIMEOUT" if r.get("timed_out")
                           else "OK" if r["solved"] else "SIN SOLUCION")
                    print(f"listo en {real_elapsed}s ({tag})", flush=True)

                if r.get("timed_out"):
                    status = "T"
                    states_display = "TIMEOUT"
                else:
                    status = "OK" if r["solved"] else "X"
                    states_display = r["states"]
                print(f"  {status:<2} {name:<19} {r['time_ms']:>12.1f} "
                      f"{str(states_display):>10} {r['steps']:>7}")

                r["algorithm"] = name
                r["level"] = level
                all_results.append(r)
            except Exception as e:
                if verbose:
                    print(f"ERROR: {e}", flush=True)
                print(f"  X  {name:<19} {'ERROR':>12} {'---':>10} {'---':>7}  ({e})")

        if level_num and level_num in PAPER_RESULTS:
            print(f"\n  --- Paper (referencia) ---")
            for name, (t, s, _) in PAPER_RESULTS[level_num].items():
                print(f"    {name:<20} {t:>12.1f} {s:>10} {'N/A':>7}")

        if level_num and level_num in ORACLE_STEPS:
            print(f"\n  Oracle (pasos óptimos): {ORACLE_STEPS[level_num]}")

    if save and all_results:
        save_results(all_results)

    return all_results


def print_result(result, algorithm_name=""):
    print(f"\n{'-'*40}")
    if algorithm_name:
        print(f"  Algoritmo : {algorithm_name}")
    print(f"  Resuelto  : {'Sí' if result['solved'] else 'No'}")
    print(f"  Tiempo    : {result['time_ms']} ms")
    print(f"  Estados   : {result['states']}")
    print(f"  Pasos     : {result['steps']}")
    if "oracle_steps" in result:
        print(f"  Oracle    : {result['oracle_steps']} pasos")
        if result["steps_vs_oracle"] is not None:
            diff = result["steps_vs_oracle"]
            sign = "+" if diff >= 0 else ""
            print(f"  Diferencia: {sign}{diff}")
    if result["solved"]:
        print(f"  Solución  : {result['solution']}")
    print(f"{'-'*40}")
