from pathlib import Path

ENTRADA = Path("niveles/microban.txt")
SALIDA = Path("niveles/microban")

SALIDA.mkdir(parents=True, exist_ok=True)

with open(ENTRADA, encoding="utf-8") as f:
    lineas = f.readlines()

nivel = []
numero = None

for linea in lineas:
    linea = linea.rstrip("\n")

    if linea.startswith(";"):
        # guardar el nivel anterior
        if numero is not None and nivel:
            with open(SALIDA / f"level{numero:03}.txt", "w") as out:
                out.write("\n".join(nivel))

        numero = int(linea[1:].strip().split()[0])
        nivel = []

    elif linea.startswith("'"):
        # comentario del nivel
        continue

    elif linea.strip() == "":
        continue

    else:
        nivel.append(linea)

# guardar el último
if numero is not None and nivel:
    with open(SALIDA / f"level{numero:03}.txt", "w") as out:
        out.write("\n".join(nivel))

print("Importados", numero, "niveles.")