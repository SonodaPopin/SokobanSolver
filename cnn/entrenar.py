# cnn/entrenar.py
# Entrena la CNN con los datos generados y guarda el modelo entrenado.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from cnn.modelo import crear_modelo


def entrenar(X, y, n_filtros=16, epochs=100, batch_size=32, lr=0.001,
             test_split=0.1, device=None, verbose=True):
    """
    Entrena la CNN con los datos dados.

    Parametros:
        X, y       : arrays de numpy del dataset (ver generar_datos.py)
        n_filtros  : filtros de la conv (16 = mejora propuesta, 8 = paper)
        epochs     : numero de epocas
        batch_size : tamaño de batch
        lr         : learning rate
        test_split : fraccion para test

    Retorna:
        (modelo, historial) donde historial tiene 'train_acc' y 'test_acc' por epoca
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    n = len(X)
    idx = np.random.permutation(n)
    n_test = int(n * test_split)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    X_train = torch.tensor(X[train_idx])
    y_train = torch.tensor(y[train_idx])
    X_test = torch.tensor(X[test_idx])
    y_test = torch.tensor(y[test_idx])

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    modelo = crear_modelo(n_filtros=n_filtros, device=device)
    optimizer = torch.optim.Adam(modelo.parameters(), lr=lr)
    criterio = nn.CrossEntropyLoss()

    historial = {"train_acc": [], "test_acc": []}

    for epoch in range(epochs):
        modelo.train()
        correctos_train, total_train = 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            salida = modelo(xb)
            loss = criterio(salida, yb)
            loss.backward()
            optimizer.step()

            pred = salida.argmax(dim=1)
            correctos_train += (pred == yb).sum().item()
            total_train += yb.size(0)

        train_acc = correctos_train / total_train if total_train else 0

        modelo.eval()
        with torch.no_grad():
            xb, yb = X_test.to(device), y_test.to(device)
            salida = modelo(xb)
            pred = salida.argmax(dim=1)
            test_acc = (pred == yb).float().mean().item() if len(yb) else 0

        historial["train_acc"].append(train_acc)
        historial["test_acc"].append(test_acc)

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch+1}/{epochs}  train_acc={train_acc:.3f}  test_acc={test_acc:.3f}")

    return modelo, historial


def guardar_modelo(modelo, path):
    torch.save(modelo.state_dict(), path)
    print(f"Modelo guardado en {path}")


def cargar_modelo(path, n_filtros=16, device="cpu"):
    modelo = crear_modelo(n_filtros=n_filtros, device=device)
    modelo.load_state_dict(torch.load(path, map_location=device))
    modelo.eval()
    return modelo


if __name__ == "__main__":
    from cnn.generar_datos import generar_dataset, guardar_dataset
    from algoritmos.BFS import solve as bfs_solve

    niveles = [f"niveles/level{i}.txt" for i in range(1, 6)]
    niveles = [n for n in niveles if os.path.exists(n)]

    print("Generando dataset con BFS...")
    X, y = generar_dataset(niveles, bfs_solve, aumentar=True)
    print(f"Dataset: {len(y)} ejemplos")

    if len(y) == 0:
        print("No se generaron datos. Revisa los niveles y el solver.")
    else:
        guardar_dataset(X, y, "cnn/dataset_bfs.npz")
        modelo, historial = entrenar(X, y, n_filtros=16, epochs=50)
        guardar_modelo(modelo, "cnn/modelo_bfs.pt")
