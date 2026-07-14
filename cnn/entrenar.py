
# cnn/entrenar.py
# Entrena la CNN con split train/val/test correcto:
#   - Train (80%): el modelo aprende con estos datos
#   - Val   (10%): se evalua cada epoca para detectar overfitting
#                  pero NO se usa para ajustar pesos
#   - Test  (10%): se evalua UNA SOLA VEZ al final, nunca durante
#                  el entrenamiento, para dar el accuracy definitivo

import sys
import os
import glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import tablero
from cnn.modelo import crear_modelo
from cnn.generar_datos import generar_dataset, guardar_dataset


def configurar_gpu():
    if not torch.cuda.is_available():
        print("GPU no disponible, usando CPU.")
        return "cpu"
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name} ({vram_total:.1f} GB VRAM)")
    torch.cuda.set_per_process_memory_fraction(0.7)
    torch.backends.cudnn.benchmark = True
    print(f"VRAM limitada al 70% ({vram_total * 0.7:.1f} GB)")
    return device


def entrenar(X, y, epochs=100, batch_size=64, lr=0.0005,
             val_split=0.1, test_split=0.1, device=None, verbose=True):
    """
    Entrena la CNN con split train/val/test.

    Split por defecto:
        80% train  - el modelo aprende con esto
        10% val    - se monitorea cada epoca (detecta overfitting)
        10% test   - se evalua solo al final (accuracy definitivo)

    Retorna:
        (modelo, historial, test_acc)
        historial tiene 'train_acc' y 'val_acc' por epoca
        test_acc es el accuracy final sobre el conjunto de test
    """
    if device is None:
        device = configurar_gpu()

    print(f"\nEntrenando en: {device.upper()}")
    print(f"Dataset total: {len(y)} ejemplos")

    # Split en tres partes en numpy (antes de mover a GPU)
    n = len(X)
    idx = np.random.permutation(n)

    n_test = int(n * test_split)
    n_val  = int(n * val_split)
    n_train = n - n_test - n_val

    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    print(f"  Train: {n_train} ejemplos (80%)")
    print(f"  Val:   {n_val}   ejemplos (10%)")
    print(f"  Test:  {n_test}  ejemplos (10%) — solo se usa al final")

    X_train = torch.tensor(X[train_idx])
    y_train = torch.tensor(y[train_idx])
    X_val   = torch.tensor(X[val_idx])
    y_val   = torch.tensor(y[val_idx])
    X_test  = torch.tensor(X[test_idx])
    y_test  = torch.tensor(y[test_idx])

    pin = device == "cuda"
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True,
                              pin_memory=pin, num_workers=0)
    val_loader   = DataLoader(TensorDataset(X_val, y_val),
                              batch_size=batch_size, shuffle=False,
                              pin_memory=pin, num_workers=0)
    test_loader  = DataLoader(TensorDataset(X_test, y_test),
                              batch_size=batch_size, shuffle=False,
                              pin_memory=pin, num_workers=0)

    modelo = crear_modelo(device=device)
    optimizer = torch.optim.Adam(modelo.parameters(), lr=lr, weight_decay=1e-4)
    criterio  = nn.CrossEntropyLoss()

    historial = {"train_acc": [], "val_acc": []}
    mejor_val_acc = 0
    mejor_epoch   = 0
    mejor_estado = None

    for epoch in range(epochs):
        # Entrenamiento
        modelo.train()
        correctos = total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = modelo(xb)
            loss = criterio(out, yb)
            loss.backward()
            optimizer.step()
            correctos += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)
        train_acc = correctos / total if total else 0

        # Validacion — solo para monitorear, no ajusta pesos
        modelo.eval()
        correctos = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = modelo(xb)
                correctos += (out.argmax(1) == yb).sum().item()
                total += yb.size(0)
        val_acc = correctos / total if total else 0

        historial["train_acc"].append(train_acc)
        historial["val_acc"].append(val_acc)

        if val_acc > mejor_val_acc:
            mejor_val_acc = val_acc
            mejor_epoch = epoch + 1

            # Guardar copia de los pesos de esta época
            mejor_estado = {
                k: v.cpu().clone()
                for k, v in modelo.state_dict().items()
    }

        if device == "cuda":
            torch.cuda.empty_cache()

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch+1:3d}/{epochs}  "
                  f"train={train_acc:.3f}  val={val_acc:.3f}")
            
    # Restaurar el mejor modelo segun validacion
    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)
        modelo.to(device)

    print(f"Modelo restaurado desde epoch {mejor_epoch} "
        f"con val_acc={mejor_val_acc:.3f}")
    
    # Test — se evalua UNA SOLA VEZ aqui, al terminar el entrenamiento
    print(f"\nEvaluando en conjunto de TEST (primera y unica vez)...")
    modelo.eval()
    correctos = total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = modelo(xb)
            correctos += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)
    test_acc = correctos / total if total else 0

    print(f"\nResultados finales:")
    print(f"  Train acc: {historial['train_acc'][-1]:.3f}")
    print(f"  Val acc:   {historial['val_acc'][-1]:.3f}  "
          f"(mejor: {mejor_val_acc:.3f} en epoch {mejor_epoch})")
    print(f"  Test acc:  {test_acc:.3f}  <- numero definitivo")

    return modelo, historial, test_acc


def validar_en_niveles(modelo_path, level_paths, device="cuda"):
    """
    Prueba el modelo resolviendo niveles reales completos.
    Esta es la metrica mas importante en la practica:
    no cuantas acciones predice bien de forma aislada,
    sino cuantos puzzles logra resolver de principio a fin.
    """
    from cnn.predecir import cargar, resolver_con_cnn

    if not os.path.exists(modelo_path):
        print(f"No se encontro el modelo en {modelo_path}")
        return 0, 0

    modelo = cargar(modelo_path, device=device)
    resueltos = []
    fallidos   = []

    print(f"\nValidando en {len(level_paths)} niveles reales...")
    for path in level_paths:
        nombre = os.path.basename(path)
        tablero.init(path)
        sol, estados = resolver_con_cnn(modelo, device=device)
        if sol != "No solution":
            resueltos.append((nombre, len(sol), estados))
        else:
            fallidos.append(nombre)

    total = len(level_paths)
    tasa  = len(resueltos) / total * 100 if total else 0

    print(f"\n  Resueltos: {len(resueltos)}/{total} ({tasa:.1f}%)")
    if resueltos:
        print(f"  Niveles resueltos:")
        for nombre, pasos, estados in resueltos:
            print(f"    {nombre}: {pasos} pasos, {estados} estados")
    if fallidos:
        print(f"  No resueltos: {len(fallidos)}")

    return len(resueltos), total


def guardar_modelo(modelo, path):
    torch.save(modelo.state_dict(), path)
    print(f"Modelo guardado en {path}")


def cargar_modelo(path, device="cuda"):
    modelo = crear_modelo(device=device)
    modelo.load_state_dict(torch.load(path, map_location=device))
    modelo.eval()
    return modelo


if __name__ == "__main__":
    from algoritmos.BFS import solve as bfs_solve
    from algoritmos.AStar import solve_hungarian_manhattan_deadlock as astar_solve

    niveles_paper    = sorted(glob.glob("niveles/paper/level*.txt"))
    niveles_microban = sorted(glob.glob("niveles/microban/level*.txt"))
    niveles_todos    = niveles_paper + niveles_microban

    print(f"Niveles del paper:  {len(niveles_paper)}")
    print(f"Niveles Microban:   {len(niveles_microban)}")
    print(f"Total:              {len(niveles_todos)}")

    if not niveles_paper:
        print("No se encontraron niveles en la carpeta 'niveles/'.")
        sys.exit(1)

    configs = [
        ("BFS", bfs_solve,   "cnn/dataset_bfs.npz",   "cnn/modelo_bfs.pt"),
        ("A*",  astar_solve, "cnn/dataset_astar.npz", "cnn/modelo_astar.pt"),
    ]

    resultados = {}

    for nombre, solver, dataset_path, modelo_path in configs:
        print(f"\n{'='*55}")
        print(f"  {nombre}: generando dataset...")
        print(f"{'='*55}")

        X, y = generar_dataset(niveles_todos, solver, aumentar=True, verbose=True)

        if len(y) == 0:
            print("Dataset vacio, se omite.")
            continue

        guardar_dataset(X, y, dataset_path)

        print(f"\nEntrenando CNN-{nombre}...")
        modelo, historial, test_acc = entrenar(
            X, y, epochs=100, batch_size=64, lr=0.0005
        )
        guardar_modelo(modelo, modelo_path)

        # Validacion en niveles reales (solo niveles del paper)
        resueltos, total_niveles = validar_en_niveles(
            modelo_path, niveles_paper, device=configurar_gpu()
        )

        resultados[nombre] = {
            "n_ejemplos":        len(y),
            "train_acc":         historial["train_acc"][-1],
            "val_acc":           historial["val_acc"][-1],
            "test_acc":          test_acc,
            "niveles_resueltos": resueltos,
            "niveles_total":     total_niveles,
        }

    # Resumen final comparativo
    print(f"\n{'='*55}")
    print("  Resumen comparativo")
    print(f"{'='*55}")
    print(f"{'Modelo':<8} {'Ejemplos':>10} {'Train':>7} "
          f"{'Val':>7} {'Test':>7} {'Niveles':>10}")
    print(f"{'-'*8} {'-'*10} {'-'*7} {'-'*7} {'-'*7} {'-'*10}")
    for nombre, r in resultados.items():
        niveles_str = f"{r['niveles_resueltos']}/{r['niveles_total']}"
        print(f"{nombre:<8} {r['n_ejemplos']:>10} "
              f"{r['train_acc']:>7.3f} {r['val_acc']:>7.3f} "
              f"{r['test_acc']:>7.3f} {niveles_str:>10}")
    print(f"\nNota: val_acc se monitorea durante el entrenamiento.")
    print(f"      test_acc se evalua una sola vez al terminar (numero definitivo).")