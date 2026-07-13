from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

csv_file = BASE_DIR / "run_history.csv"

# Leer el archivo CSV
df = pd.read_csv(csv_file)

df["level"] = (
    df["level"]
    .apply(lambda x: Path(x).stem)      # level1
    .str.replace("level", "", regex=False)
    .astype(int)
)

# Ordenar por nivel para que las líneas se vean correctamente
df = df.sort_values("level")

# Se dibuja una línea por algoritmo
algorithms = df["algorithm"].unique()

colors = plt.get_cmap("tab20").colors

# Función para crear un gráfico
def plot_metric(y_column, y_label, output_file):
    plt.figure(figsize=(8, 5))

    for i, alg in enumerate(algorithms):
        data = df[df["algorithm"] == alg]
        plt.plot(
            data["level"],
            np.log(data[y_column]),
            marker="o",
            color=colors[i],
            label=alg
        )

    plt.xlabel("Level")
    plt.ylabel(y_label)
    plt.title(f"Level vs {y_label}")
    plt.grid(True)
    plt.legend(
        title="Algorithm",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5)
    )
    plt.tight_layout()
    plt.savefig(BASE_DIR / output_file, dpi=300, bbox_inches="tight")
    plt.show()

# Crear los tres gráficos
#plot_metric("time_ms", "Time (ms)", "level_vs_time.png")
#plot_metric("steps", "Steps", "level_vs_steps.png")
plot_metric("states", "Log(States)", "level_vs_Log(states).png")

print("Gráficos creados correctamente.")