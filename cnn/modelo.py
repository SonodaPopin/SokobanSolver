# cnn/modelo.py
# Arquitectura de la CNN para predecir la accion del jugador en Sokoban.
#
# El paper usa 8 filtros conv 3x3 -> capas ocultas (32, 16) -> softmax (4).
# Esta versión mejorada utiliza:
# - 16 y 32 filtros convolucionales
# - MaxPooling para reducir dimensionalidad
# - Dropout para disminuir overfitting
# - capas fully connected más pequeñas

import torch
import torch.nn as nn

BOARD_SIZE = 32
N_CHANNELS = 4   # pared, objetivo, caja, jugador
N_ACTIONS = 4    # U, R, D, L


class SokobanCNN(nn.Module):
    """
    CNN para Sokoban.

    """

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(N_CHANNELS, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        flat_size = 32 * BOARD_SIZE * BOARD_SIZE

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, N_ACTIONS),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x  # logits, softmax se aplica en la loss (CrossEntropyLoss)


def crear_modelo(device="cpu"):
    modelo = SokobanCNN().to(device)
    return modelo
