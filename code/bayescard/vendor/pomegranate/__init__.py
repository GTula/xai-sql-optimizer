"""Stub mínimo de pomegranate para mantener importable el código BayesCard.

La implementación real de BayesCard depende de pomegranate, pero el motor
SQL usa el árbol copiado principalmente para mantener el código local.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional


@dataclass
class BayesianNetwork:
    """Versión mínima compatible con la API usada por BayesCard."""

    structure: List[Any]

    @classmethod
    def from_samples(cls, *_args, **_kwargs):
        return cls(structure=[])

    def fit(self, *_args, **_kwargs):
        return self

    def add_node(self, *_args, **_kwargs):
        return self

    def to_junction_tree(self):
        return self
