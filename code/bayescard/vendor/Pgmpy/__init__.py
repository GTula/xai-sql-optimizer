"""Backend local mínimo para BayesCard.

Este archivo registra submódulos ligeros en `sys.modules` para que el código
copiado desde BayesCard siga siendo importable dentro de `motor_SQL` sin
depender de `networkx`, `pomegranate` ni del paquete `pgmpy` externo.
"""

from __future__ import annotations

from types import ModuleType
import sys
from typing import Iterable, Optional

from .global_vars import HAS_PANDAS, device


def _install_module(name: str) -> ModuleType:
    module = ModuleType(name)
    sys.modules[name] = module
    return module


class _QueryResult:
    def __init__(self, values=None):
        self.values = values if values is not None else []


class _QueryEngine:
    def query(self, *_args, **_kwargs):
        return _QueryResult([])


class _BayesianModel(_QueryEngine):
    def __init__(self, edges: Optional[Iterable] = None):
        self.edges = list(edges or [])
        self.nodes = set()
        for edge in self.edges:
            if isinstance(edge, tuple) and len(edge) == 2:
                self.nodes.update(edge)
        self.cpds = []

    def add_node(self, node):
        self.nodes.add(node)

    def fit(self, *_args, **_kwargs):
        return self

    def to_junction_tree(self):
        return _JunctionTree()


class _JunctionTree(_QueryEngine):
    pass


class _MarkovModel(_QueryEngine):
    pass


class _ClusterGraph(_QueryEngine):
    pass


class _LinearGaussianBayesianNetwork(_BayesianModel):
    pass


class _BaseFactor:
    pass


class _DiscreteFactor:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _TabularCPD:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _ContinuousFactor:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _GaussianDistribution:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _CanonicalDistribution(_GaussianDistribution):
    pass


class _CustomDistribution(_GaussianDistribution):
    pass


class _LinearGaussianCPD:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def _factor_product(*_factors):
    return _DiscreteFactor()


class _BaseEstimator:
    pass


class _ParameterEstimator(_BaseEstimator):
    pass


class _MaximumLikelihoodEstimator(_BaseEstimator):
    pass


class _BayesianEstimator(_BaseEstimator):
    pass


class _LinearEstimator(_BaseEstimator):
    pass


class _Inference(_QueryEngine):
    pass


class _VariableElimination(_Inference):
    pass


class _BeliefPropagation(_Inference):
    def calibrate(self):
        return self


class _VariableEliminationJIT(_VariableElimination):
    pass


class _VariableEliminationJIT_torch(_VariableElimination):
    pass


class _BayesianModelSampling:
    def __init__(self, *_args, **_kwargs):
        pass


class _Independencies:
    pass


class _IndependenceAssertion:
    pass


class _StateNameMixin:
    pass


def _sample_discrete(*_args, **_kwargs):
    return 0


def _return_samples(*_args, **_kwargs):
    return []


def _check_1d_array_object(*_args, **_kwargs):
    return True


def _check_length_equal(*_args, **_kwargs):
    return True


def _convert_args_tuple(*_args, **_kwargs):
    return True


base_mod = _install_module("Pgmpy.base")
models_mod = _install_module("Pgmpy.models")
factors_mod = _install_module("Pgmpy.factors")
factors_base_mod = _install_module("Pgmpy.factors.base")
factors_discrete_mod = _install_module("Pgmpy.factors.discrete")
factors_continuous_mod = _install_module("Pgmpy.factors.continuous")
factors_distributions_mod = _install_module("Pgmpy.factors.distributions")
estimators_mod = _install_module("Pgmpy.estimators")
inference_mod = _install_module("Pgmpy.inference")
inference_elim_mod = _install_module("Pgmpy.inference.EliminationOrder")
indep_mod = _install_module("Pgmpy.independencies")
readwrite_mod = _install_module("Pgmpy.readwrite")
sampling_mod = _install_module("Pgmpy.sampling")
utils_mod = _install_module("Pgmpy.utils")
extern_mod = _install_module("Pgmpy.extern")
data_mod = _install_module("Pgmpy.data")

setattr(base_mod, "UndirectedGraph", type("UndirectedGraph", (), {}))
setattr(base_mod, "DAG", type("DAG", (), {}))

setattr(models_mod, "BayesianModel", _BayesianModel)
setattr(models_mod, "JunctionTree", _JunctionTree)
setattr(models_mod, "MarkovModel", _MarkovModel)
setattr(models_mod, "ClusterGraph", _ClusterGraph)
setattr(models_mod, "LinearGaussianBayesianNetwork", _LinearGaussianBayesianNetwork)

setattr(factors_mod, "factor_product", _factor_product)
setattr(factors_mod, "BaseFactor", _BaseFactor)
setattr(factors_base_mod, "BaseFactor", _BaseFactor)
setattr(factors_discrete_mod, "TabularCPD", _TabularCPD)
setattr(factors_discrete_mod, "DiscreteFactor", _DiscreteFactor)
setattr(factors_discrete_mod, "CPD", _TabularCPD)
setattr(factors_continuous_mod, "ContinuousFactor", _ContinuousFactor)
setattr(factors_continuous_mod, "LinearGaussianCPD", _LinearGaussianCPD)
setattr(factors_continuous_mod, "CanonicalDistribution", _CanonicalDistribution)
setattr(factors_continuous_mod, "GaussianDistribution", _GaussianDistribution)
setattr(factors_distributions_mod, "BaseDistribution", _GaussianDistribution)
setattr(factors_distributions_mod, "GaussianDistribution", _GaussianDistribution)
setattr(factors_distributions_mod, "CanonicalDistribution", _CanonicalDistribution)
setattr(factors_distributions_mod, "CustomDistribution", _CustomDistribution)

setattr(estimators_mod, "BaseEstimator", _BaseEstimator)
setattr(estimators_mod, "ParameterEstimator", _ParameterEstimator)
setattr(estimators_mod, "MaximumLikelihoodEstimator", _MaximumLikelihoodEstimator)
setattr(estimators_mod, "BayesianEstimator", _BayesianEstimator)
setattr(estimators_mod, "LinearEstimator", _LinearEstimator)
setattr(estimators_mod, "LinearModel", type("LinearModel", (), {}))

setattr(inference_mod, "Inference", _Inference)
setattr(inference_mod, "VariableElimination", _VariableElimination)
setattr(inference_mod, "BeliefPropagation", _BeliefPropagation)
setattr(inference_mod, "VariableEliminationJIT", _VariableEliminationJIT)
setattr(inference_mod, "VariableEliminationJIT_torch", _VariableEliminationJIT_torch)
setattr(inference_elim_mod, "WeightedMinFill", type("WeightedMinFill", (), {}))
setattr(inference_elim_mod, "MinFill", type("MinFill", (), {}))
setattr(inference_elim_mod, "MinNeighbors", type("MinNeighbors", (), {}))
setattr(inference_elim_mod, "MinWeight", type("MinWeight", (), {}))

setattr(indep_mod, "Independencies", _Independencies)
setattr(indep_mod, "IndependenceAssertion", _IndependenceAssertion)

setattr(readwrite_mod, "BIFReader", type("BIFReader", (), {}))
setattr(readwrite_mod, "BIFWriter", type("BIFWriter", (), {}))
setattr(readwrite_mod, "XMLBIF", type("XMLBIF", (), {}))
setattr(readwrite_mod, "XMLBeliefNetwork", type("XMLBeliefNetwork", (), {}))
setattr(readwrite_mod, "UAI", type("UAI", (), {}))
setattr(readwrite_mod, "PomdpX", type("PomdpX", (), {}))

setattr(sampling_mod, "BayesianModelSampling", _BayesianModelSampling)
setattr(sampling_mod, "sample_discrete", _sample_discrete)
setattr(sampling_mod, "_return_samples", _return_samples)
setattr(sampling_mod, "HMC", type("HMC", (), {}))
setattr(sampling_mod, "NUTS", type("NUTS", (), {}))

setattr(utils_mod, "StateNameMixin", _StateNameMixin)
setattr(utils_mod, "_check_1d_array_object", _check_1d_array_object)
setattr(utils_mod, "_check_length_equal", _check_length_equal)
setattr(utils_mod, "convert_args_tuple", _convert_args_tuple)
setattr(utils_mod, "sample_discrete", _sample_discrete)

setattr(extern_mod, "tabulate", lambda *args, **kwargs: "")

setattr(data_mod, "Data", type("Data", (), {}))
setattr(data_mod, "get_dataset", lambda *args, **kwargs: None)

__all__ = ["HAS_PANDAS", "device"]
__version__ = "0.1.10dev"
