"""
FASE 3: INDEX
B+ tree: estructura para búsqueda rápida e indexación
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import pickle
import os


@dataclass
class BTreeNode:
    """Nodo de un B+ tree"""
    is_leaf: bool
    keys: List  # valores key de búsqueda
    values: List  # RID (row ids) para hojas, referencias a hijos para internos
    
    def is_full(self, order: int) -> bool:
        return len(self.keys) >= 2 * order - 1
    
    def is_underfull(self, order: int) -> bool:
        return len(self.keys) < order - 1


class BTree:
    """
    B+ tree: árbol de búsqueda balanceado para búsquedas rápidas.
    Cada nodo puede contener múltiples keys.
    
    Las hojas contienen (key -> RID) para acceso a filas.
    Los nodos internos solo tienen keys para guiar la búsqueda.
    """
    
    def __init__(self, index_file: str, order: int = 4):
        """
        Args:
            index_file: archivo donde se persiste el árbol
            order: grado del árbol (afecta el factor de ramificación)
        """
        self.index_file = index_file
        self.order = order
        self.root = None
        
        if os.path.exists(index_file):
            self.load()
        else:
            # Crear árbol vacío
            self.root = BTreeNode(is_leaf=True, keys=[], values=[])
    
    def search(self, key) -> List[Tuple[int, int]]:
        """
        Busca todas las filas con el key dado.
        
        Returns:
            Lista de (page_id, offset) donde está la fila
        """
        if not self.root:
            return []
        
        rids = []
        self._search_recursive(self.root, key, rids)
        return rids
    
    def _search_recursive(self, node: BTreeNode, key, rids: List) -> None:
        """Búsqueda recursiva en el árbol"""
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        
        if node.is_leaf:
            # En hoja: buscar coincidencias
            for j in range(len(node.keys)):
                if node.keys[j] == key:
                    rids.append(node.values[j])
        else:
            # En nodo interno: recursar al hijo apropiado
            if i < len(node.values):
                self._search_recursive(node.values[i], key, rids)
    
    def insert(self, key, rid: Tuple[int, int]) -> None:
        """
        Inserta un (key, RID) en el árbol.
        
        Args:
            key: valor a indexar
            rid: (page_id, offset) de la fila
        """
        if self.root.is_full(self.order):
            # Crear nueva raíz
            new_root = BTreeNode(is_leaf=False, keys=[], values=[self.root])
            self._split_child(new_root, 0)
            self.root = new_root
        
        self._insert_non_full(self.root, key, rid)
        self.save()
    
    def _insert_non_full(self, node: BTreeNode, key, rid: Tuple[int, int]) -> None:
        """Inserta en un nodo que no está lleno"""
        i = len(node.keys) - 1
        
        if node.is_leaf:
            # Insertar directamente en hoja
            node.keys.append(None)
            node.values.append(None)
            while i >= 0 and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                node.values[i + 1] = node.values[i]
                i -= 1
            node.keys[i + 1] = key
            node.values[i + 1] = rid
        else:
            # Encontrar hijo donde ir
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            
            child = node.values[i]
            if child.is_full(self.order):
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            
            self._insert_non_full(node.values[i], key, rid)
    
    def _split_child(self, parent: BTreeNode, index: int) -> None:
        """Divide un hijo lleno"""
        order = self.order
        full_child = parent.values[index]
        
        mid = order - 1
        new_child = BTreeNode(
            is_leaf=full_child.is_leaf,
            keys=full_child.keys[mid + 1:],
            values=full_child.values[mid + 1:] if not full_child.is_leaf else full_child.values[mid + 1:]
        )
        
        parent.keys.insert(index, full_child.keys[mid])
        parent.values.insert(index + 1, new_child)
        
        full_child.keys = full_child.keys[:mid]
        full_child.values = full_child.values[:mid + 1] if not full_child.is_leaf else full_child.values[:mid + 1]
    
    def save(self) -> None:
        """Persiste el árbol en disco"""
        with open(self.index_file, 'wb') as f:
            pickle.dump(self.root, f)
    
    def load(self) -> None:
        """Carga el árbol desde disco"""
        with open(self.index_file, 'rb') as f:
            self.root = pickle.load(f)
    
    def __repr__(self) -> str:
        return f"<BTree order={self.order} root_keys={len(self.root.keys) if self.root else 0}>"
