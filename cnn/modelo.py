# cnn/modelo.py
# Arquitectura de la CNN para predecir la accion del jugador en Sokoban.
#
# El paper usa 8 filtros conv 3x3 -> capas ocultas (32, 16) -> softmax (4).
# Aqui se implementa la version mejorada propuesta: 16 filtros en vez de 8,
# manteniendo el resto de la estructura para que la comparacion sea clara.

import torch
import torch.nn as nn

BOARD_SIZE = 32
N_CHANNELS = 4   # pared, objetivo, caja, jugador
N_ACTIONS = 4    # U, R, D, L


class SokobanCNN(nn.Module):
    """
    CNN para Sokoban.

    Parametros:
        n_filtros : cantidad de filtros conv (8 = paper original, 16 = mejora propuesta)
    """

    def __init__(self, n_filtros=16):
        super().__init__()
        self.n_filtros = n_filtros

        self.conv = nn.Sequential(
            nn.Conv2d(N_CHANNELS, n_filtros, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        flat_size = n_filtros * BOARD_SIZE * BOARD_SIZE

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, N_ACTIONS),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x  # logits, softmax se aplica en la loss (CrossEntropyLoss)


def crear_modelo(n_filtros=16, device="cpu"):
    modelo = SokobanCNN(n_filtros=n_filtros).to(device)
    return modelo
