"""
XAI: Validación de fidelidad SHAP vs ejecución real.

Instrumenta la ejecución del plan SQL para recolectar métricas reales
(filas por paso de JOIN) y las compara contra las estimaciones que SHAP usó.

Pregunta central:
  ¿Las features que SHAP marcó como importantes describen lo que realmente
  ocurrió durante la ejecución, o sólo describen el modelo de costo interno?

Flujo:
  1. engine.execute_with_trace(sql)   → resultados + raw_trace (filas reales)
  2. RuntimeTrace.from_engine_trace() → estructura tipada del trace
  3. compare_shap_vs_runtime()        → FidelityReport con score y detalle
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Estructuras de datos
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class JoinStepTrace:
    """Métricas reales de un paso de JOIN durante la ejecución."""
    step: int
    input_rows: int    # filas acumuladas antes del JOIN (izquierda del NLJ)
    table_rows: int    # filas leídas de la tabla unida (sin filtro WHERE local)
    output_rows: int   # filas producidas tras aplicar la condición de JOIN


@dataclass
class RuntimeTrace:
    """
    Métricas reales recolectadas durante la ejecución del plan.

    Nota sobre diferencia con las estimaciones:
    - El engine aplica la condición WHERE DESPUÉS de todos los JOINs.
    - El estimador aplica selectividades WHERE-derivadas DURANTE cada paso.
    - Por eso 'table_rows' en el trace es el tamaño real de la tabla (sin
      filtro WHERE), mientras que 'join_N_table_rows' en las features estimadas
      puede ser menor si el optimizador aplica selectividad local.
    - Esta diferencia es parte de lo que la validación de fidelidad detecta.
    """
    base_rows: int
    join_steps: List[JoinStepTrace] = field(default_factory=list)
    final_rows: int = 0
    plan_order: List[str] = field(default_factory=list)

    @classmethod
    def from_engine_trace(cls, raw: Dict[str, Any]) -> "RuntimeTrace":
        """Construye un RuntimeTrace desde el dict crudo devuelto por execute_with_trace."""
        steps = [
            JoinStepTrace(
                step=s["step"],
                input_rows=s["input_rows"],
                table_rows=s["table_rows"],
                output_rows=s["output_rows"],
            )
            for s in raw.get("steps", [])
        ]
        return cls(
            base_rows=raw.get("base_rows", 0),
            join_steps=steps,
            final_rows=raw.get("final_rows", 0),
            plan_order=list(raw.get("plan_order", [])),
        )

    def to_features(self) -> Dict[str, float]:
        """
        Convierte el trace en el mismo formato de features que extract_plan_features.

        Solo incluye las features que realmente se pueden medir en runtime
        (features de steps de JOIN y agregados intermedios). Features
        estructurales fijas (join_count, table_count, etc.) se omiten porque
        no varían entre planes y no aportan a la comparación.
        """
        outputs = [float(s.output_rows) for s in self.join_steps]
        costs = outputs  # el proxy usa output_rows como costo del paso

        features: Dict[str, float] = {
            "base_filtered_rows": float(self.base_rows),
        }

        if outputs:
            features["sum_intermediate_rows"] = sum(outputs)
            features["max_intermediate_rows"] = max(outputs)
            features["avg_intermediate_rows"] = sum(outputs) / len(outputs)
            most_expensive_idx = costs.index(max(costs))
            features["most_expensive_join_cost"] = float(max(costs))
            features["most_expensive_join_position"] = float(most_expensive_idx + 1)

        for s in self.join_steps:
            prefix = f"join_{s.step}"
            features[f"{prefix}_input_rows"] = float(s.input_rows)
            features[f"{prefix}_table_rows"] = float(s.table_rows)
            features[f"{prefix}_output_rows"] = float(s.output_rows)
            features[f"{prefix}_cost"] = float(s.output_rows)

        return features


# ─────────────────────────────────────────────────────────────────────────────
# Fidelidad por feature
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FeatureFidelity:
    """Fidelidad de una feature individual (estimada vs real)."""
    feature: str
    estimated: float
    real: float
    error_pct: float     # |estimado - real| / max(|real|, 1) * 100
    shap_rank: int       # posición en el ranking SHAP (1 = más importante)
    shap_value: float
    is_faithful: bool    # True si error_pct <= threshold


@dataclass
class FidelityReport:
    """Reporte completo de fidelidad SHAP vs runtime."""
    overall_fidelity_score: float          # 0.0–1.0; 1.0 = estimaciones perfectas
    feature_fidelities: List[FeatureFidelity]
    faithful_top_features: List[str]       # top features SHAP con bajo error
    misleading_top_features: List[str]     # top features SHAP con alto error
    summary: str


# ─────────────────────────────────────────────────────────────────────────────
# Comparación principal
# ─────────────────────────────────────────────────────────────────────────────

def compare_shap_vs_runtime(
    shap_values: Dict[str, float],
    estimated_features: Dict[str, float],
    runtime_trace: RuntimeTrace,
    top_k: int = 5,
    fidelity_threshold_pct: float = 20.0,
) -> FidelityReport:
    """
    Compara las features que SHAP consideró importantes contra sus valores reales.

    Para cada feature en el top_k SHAP:
    - Toma el valor estimado (el que usó SHAP para su explicación).
    - Toma el valor real medido durante la ejecución.
    - Calcula el error de estimación porcentual.

    Una feature es "fiel" si su error < fidelity_threshold_pct.
    Un score global de fidelidad promedia los errores: 1.0 = sin error, 0.0 = error total.

    Args:
        shap_values: {feature: shap_value} del plan elegido.
        estimated_features: {feature: valor_estimado} usado por SHAP.
        runtime_trace: métricas reales de ejecución.
        top_k: cuántas top features de SHAP considerar.
        fidelity_threshold_pct: % de error máximo para considerar fiel.

    Returns:
        FidelityReport con score global y tabla de detalle por feature.
    """
    real_features = runtime_trace.to_features()

    # Rankear features SHAP por importancia absoluta
    ranked = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
    top_shap = ranked[:top_k]

    fidelities: List[FeatureFidelity] = []
    errors: List[float] = []

    for rank, (feat, sv) in enumerate(top_shap, start=1):
        estimated = estimated_features.get(feat)
        real = real_features.get(feat)

        # Solo comparar features que existen en ambos lados
        if estimated is None or real is None:
            continue

        denom = max(abs(real), 1.0)
        error_pct = abs(estimated - real) / denom * 100.0
        errors.append(error_pct)

        fidelities.append(FeatureFidelity(
            feature=feat,
            estimated=float(estimated),
            real=float(real),
            error_pct=error_pct,
            shap_rank=rank,
            shap_value=float(sv),
            is_faithful=error_pct <= fidelity_threshold_pct,
        ))

    if errors:
        # Score: 1 - error_promedio_normalizado. Error promedio es el promedio
        # de error_pct / 100, así que score = 1 - mean(error_pct) / 100.
        overall_score = max(0.0, 1.0 - sum(errors) / (len(errors) * 100.0))
    else:
        overall_score = 1.0

    faithful = [f.feature for f in fidelities if f.is_faithful]
    misleading = [f.feature for f in fidelities if not f.is_faithful]

    summary = _build_fidelity_summary(
        overall_score, fidelities, faithful, misleading,
        fidelity_threshold_pct, runtime_trace.plan_order,
    )

    return FidelityReport(
        overall_fidelity_score=overall_score,
        feature_fidelities=fidelities,
        faithful_top_features=faithful,
        misleading_top_features=misleading,
        summary=summary,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Formato del reporte
# ─────────────────────────────────────────────────────────────────────────────

def _build_fidelity_summary(
    score: float,
    fidelities: List[FeatureFidelity],
    faithful: List[str],
    misleading: List[str],
    threshold: float,
    plan_order: List[str],
) -> str:
    SEP = "─" * 72
    lines = [
        "VALIDACIÓN DE FIDELIDAD: SHAP vs Ejecución Real",
        SEP,
    ]

    if plan_order:
        lines.append(f"Orden real de ejecución : {' -> '.join(plan_order)}")

    lines += [
        f"Score de fidelidad      : {score:.2f} / 1.00",
        f"(Threshold error fiel   : < {threshold:.0f}%  |  "
        f"Features comparadas: {len(fidelities)}  |  "
        f"Fieles: {len(faithful)}  Engañosas: {len(misleading)})",
        "",
    ]

    if not fidelities:
        lines.append(
            "No hay features de SHAP comparables con el runtime "
            "(las features del plan no tienen equivalente medible en ejecución)."
        )
        return "\n".join(lines)

    header = (
        f"  {'#':>3}  {'Feature':<30}  {'Estimado':>10}  {'Real':>10}  "
        f"{'Error%':>7}  {'Fiel':>5}  SHAP"
    )
    lines.append(header)
    lines.append("  " + "─" * 70)

    for f in fidelities:
        fiel = "✓" if f.is_faithful else "✗"
        lines.append(
            f"  {f.shap_rank:>3}  {f.feature:<30}  {f.estimated:>10.2f}  "
            f"{f.real:>10.2f}  {f.error_pct:>6.1f}%  {fiel:>5}  {f.shap_value:+.4f}"
        )

    lines.append("")

    if not misleading:
        lines += [
            "Conclusión ► Las features que SHAP usó para explicar la decisión",
            "             coinciden con la ejecución real. La explicación es válida.",
        ]
    elif len(misleading) >= len(faithful):
        lines += [
            "Conclusión ► La mayoría de las features top-SHAP tienen errores de",
            "             estimación altos. SHAP describe el modelo de costo,",
            "             no necesariamente la realidad de ejecución.",
        ]
        if misleading:
            lines.append(f"             Features problemáticas : {', '.join(misleading)}")
        if faithful:
            lines.append(f"             Features confiables    : {', '.join(faithful)}")
    else:
        lines += [
            "Conclusión ► La explicación SHAP es parcialmente fiel.",
        ]
        if faithful:
            lines.append(f"             Confiables           : {', '.join(faithful)}")
        if misleading:
            lines.append(f"             Alta discrepancia    : {', '.join(misleading)}")

    lines += [
        "",
        "Nota: 'Real' mide filas en el engine (WHERE se aplica al final).",
        "      'Estimado' aplica selectividades WHERE durante el paso → diferencia es normal.",
    ]

    return "\n".join(lines)
