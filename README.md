# Sokoban Solver

Proyecto de IA que implementa y compara distintos algoritmos para resolver el juego Sokoban, basado en el paper *"AI in Game Playing: Sokoban Solver"* (Venkatesan, Jain & Grewal, 2018).

## Descripción

Sokoban es un juego de puzzles donde el jugador debe empujar cajas hasta posiciones objetivo dentro de un tablero con paredes. Es un problema NP-Hard y PSPACE-Completo, lo que lo hace ideal para evaluar y comparar distintas estrategias de búsqueda en IA.

Este proyecto replica los algoritmos del paper original y propone mejoras propias:
- Detección de deadlocks por esquinas (Dead Square)
- Heurística A* con penalización por deadlocks
- CNN entrenada con BFS y con A* para comparación

---

## Requisitos

```bash
pip install torch numpy
```

Python 3.8 o superior.

---

## Estructura del proyecto

```
SokobanSolver/
├── main.py                    # Punto de entrada principal
├── tablero.py                 # Lógica del tablero (estado, movimientos)
│
├── algoritmos/
│   ├── BFS.py                 # Breadth First Search
│   ├── DFS.py                 # Depth First Search
│   ├── UCS.py                 # Uniform Cost Search (Cost1 y Cost2)
│   └── AStar.py               # A* con 5 heurísticas
│
├── optimizacion/
│   ├── hashing.py             # Tabla hash de estados visitados
│   ├── deadlocks.py           # Detección de deadlocks (esquinas + freeze)
│   ├── heuristicas.py         # Heurísticas para A*
│   ├── hungarian.py           # Algoritmo húngaro para asignación óptima
│   └── tunnelMacros.py        # Tunnel macros (implementado, no integrado aún)
│
├── cnn/
│   ├── generar_datos.py       # Genera dataset desde soluciones de algoritmos
│   ├── modelo.py              # Arquitectura CNN (PyTorch)
│   ├── entrenar.py            # Entrenamiento con BFS y A*
│   ├── predecir.py            # Inferencia: resuelve niveles con CNN
│   ├── modelo_bfs.pt          # Pesos entrenados con BFS (generado al entrenar)
│   └── modelo_astar.pt        # Pesos entrenados con A* (generado al entrenar)
│
├── resultados/
│   └── metricas.py            # Medición, comparación y guardado de resultados
│
└── niveles/
    ├── level1.txt             # Niveles originales del paper (CodaLab)
    ├── level2.txt
    └── ...
```

---

## Uso

### Correr un nivel específico

```bash
python main.py --level niveles/level1.txt --level-number 1
```

### Comparar todos los algoritmos en todos los niveles

```bash
python main.py --compare
```

### Ver progreso en tiempo real (modo debug)

```bash
python main.py --compare --verbose
```

### Guardar resultados en CSV/JSON

```bash
python main.py --compare --save
```

### Controlar el tiempo máximo por algoritmo

```bash
python main.py --compare --timeout 60
```

### Ver histórico de resultados guardados

```bash
python main.py --history
```

### Borrar histórico

```bash
python main.py --clear-history
```

---

## Algoritmos implementados

| Algoritmo | Descripción |
|---|---|
| BFS | Breadth First Search. Garantiza el camino más corto. |
| DFS | Depth First Search. Rápido pero no garantiza solución óptima. Límite de 150 pasos. |
| UCS Cost1 | Uniform Cost Search con costos diferenciados (empuje fuera de objetivo penalizado). |
| UCS Cost2 | Uniform Cost Search con costo uniforme (equivalente a BFS). |
| A* Eucl | A* con distancia euclidiana mínima como heurística. |
| A* Manh | A* con distancia Manhattan mínima como heurística. |
| A* Hung+Eucl | A* con asignación óptima húngara (euclidiana). |
| A* Hung+Manh | A* con asignación óptima húngara (Manhattan). Mejor según el paper. |
| A* Hung+Manh+DL | A* húngaro Manhattan con penalización por cercanía a deadlocks. Mejora propia. |
| CNN-BFS | Red neuronal convolucional entrenada con datos de BFS. |
| CNN-A* | Red neuronal convolucional entrenada con datos de A*. Mejora propuesta. |

---

## Optimizaciones

**Hashing de estados**: todos los algoritmos usan un set de estados visitados para evitar reexplorar configuraciones ya vistas.

**Detección de deadlocks (mejora sobre el paper)**: el código original del paper (`Level.py`) tiene `isFailure()` que siempre retorna `False` — es decir, no implementa ninguna detección de deadlocks. Este proyecto agrega:
- **Dead Square**: esquinas absolutas donde una caja nunca puede salir (pared en al menos un lado horizontal y un lado vertical, sin ser objetivo).
- Esto reduce entre 49% y 82% los estados explorados según el nivel.

---

## CNN

La CNN replica la arquitectura del paper con una mejora: **16 filtros convolucionales** en vez de 8.

```
Entrada: 4 canales × 32×32 (paredes, objetivos, cajas, jugador)
    ↓
Conv2d (16 filtros 3×3) + ReLU
    ↓
Flatten → Linear(32) → ReLU → Linear(16) → ReLU → Linear(4)
    ↓
Salida: 4 acciones (U, R, D, L)
```

### Generar datos y entrenar ambos modelos

```bash
python cnn/entrenar.py
```

Esto:
1. Corre BFS y A* sobre todos los niveles en `niveles/`
2. Registra cada par (estado, acción) de las soluciones
3. Aplica aumento de datos ×8 (rotaciones y reflexiones)
4. Entrena dos modelos separados (100 épocas, lr=0.001)
5. Muestra tabla comparativa de precisión vs paper

Una vez entrenados, `CNN-BFS` y `CNN-A*` aparecen automáticamente en `main.py --compare`.

### Limitación conocida

La CNN predice una acción a la vez sin capacidad de búsqueda ni retroceso. Aunque el test accuracy ronda el 45-50%, resolver un nivel completo requiere acertar 30-80+ acciones consecutivas, lo que hace que la tasa de éxito sea baja. Esta es la misma limitación reportada por el paper original.

---

## Comparación con el paper

Los niveles 1-5 corresponden a los niveles originales del paper (obtenidos del workspace CodaLab: `0x2412ae8944eb449db74ce9bc0b9463fe`), lo que permite comparación directa con la Tabla 1 del paper.

| Métrica | Paper (A* Hung+Manh, nivel 4) | Este proyecto |
|---|---|---|
| Tiempo (ms) | 6572.6 | — (distinto hardware) |
| Estados explorados | 9,800 | ~11,593 |
| Pasos solución | N/A | 31 (oracle=82) |
| CNN train acc | 58% | ~91% |
| CNN test acc | 45% | ~48% |

> **Nota**: los tiempos no son directamente comparables porque el paper no especifica el hardware usado.

---

## Referencia

Venkatesan, A., Jain, A., & Grewal, R. (2018). *AI in Game Playing: Sokoban Solver*. CS 221 Project Progress Report. arXiv:1807.00049.

Código original del paper: https://worksheets.codalab.org/worksheets/0x2412ae8944eb449db74ce9bc0b9463fe/
