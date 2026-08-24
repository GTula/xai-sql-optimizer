"""Fachada única para el código BayesCard vendorizado en motor_SQL.

Este paquete expone el optimizador local y registra alias para los nombres
históricos que usaban los módulos copiados desde BayesCard.
"""

from __future__ import annotations

import importlib
import sys

from .cost_model import BayesCardCostModel
from .query_optimizer import BayesCardQueryOptimizer


def _install_alias(alias_name: str, target_name: str) -> None:
	module = importlib.import_module(target_name)
	sys.modules.setdefault(alias_name, module)


_ALIAS_TARGETS = {
	"DataPrepare": "bayescard.vendor.DataPrepare",
	"DeepDBUtils": "bayescard.vendor.DeepDBUtils",
	"Evaluation": "bayescard.vendor.Evaluation",
	"Inference": "bayescard.vendor.Inference",
	"Models": "bayescard.vendor.Models",
	"Parameter": "bayescard.vendor.Parameter",
	"Pgmpy": "bayescard.vendor.Pgmpy",
	"Schemas": "bayescard.vendor.Schemas",
	"Testing": "bayescard.vendor.Testing",
	"pgmpy": "bayescard.vendor.Pgmpy",
	"pomegranate": "bayescard.vendor.pomegranate",
}

for alias_name, target_name in _ALIAS_TARGETS.items():
	_install_alias(alias_name, target_name)

__all__ = ["BayesCardCostModel", "BayesCardQueryOptimizer"]