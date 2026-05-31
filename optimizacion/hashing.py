"""
hashing.py
----------
Módulo independiente de hashing de estados para Sokoban.
Puede ser importado por cualquier algoritmo de búsqueda.

Uso:
    from hashing import StateHash

    visited = StateHash()
    visited.add(state)
    if state in visited:
        ...
    visited.reset()
"""


class StateHash:
    """
    Tabla hash de estados visitados.
    Envuelve un set de Python con una interfaz clara
    para que cualquier algoritmo lo use sin conocer
    los detalles de implementación.
    """

    def __init__(self):
        self._visited = set()

    def add(self, state):
        """Registra un estado como visitado."""
        self._visited.add(state)

    def __contains__(self, state):
        """Permite usar 'if state in visited'."""
        return state in self._visited

    def reset(self):
        """Limpia todos los estados (útil para reutilizar entre niveles)."""
        self._visited.clear()

    def __len__(self):
        """Retorna cuántos estados han sido visitados."""
        return len(self._visited)