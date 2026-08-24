from typing import Dict, List, Optional


class Explanation:
	"""Contenedor simple para respuestas de explicabilidad del optimizador."""

	def __init__(
		self,
		optimizer_name: str,
		chosen_plan: List[str] = None,           # nombre de las tablas en el orden elegido por el optimizador
		top_features: List[str] = None,
		metadata: Dict[str, str] = None,          # parámetros de la ejecución que generó esta explicación
		# --- XAI extensions (all optional for backward compatibility) ---
		selected_plan_score: Optional[float] = None,
		compared_plans_count: Optional[int] = None,
		shap_used: bool = False,
		lime_used: bool = False,
		feature_values_for_selected_plan: Optional[Dict[str, float]] = None,
		shap_values_for_selected_plan: Optional[Dict[str, float]] = None,
		lime_values_for_selected_plan: Optional[Dict[str, float]] = None,
		lime_top_features: Optional[List[str]] = None,
		natural_language_summary: str = "",
		warnings: Optional[List[str]] = None,
		# --- Interpretation layer ---
		executive_summary: str = "",
		factors_positive: Optional[List[str]] = None,
		factors_negative: Optional[List[str]] = None,
		runner_up_plan: Optional[List[str]] = None,
		runner_up_score: Optional[float] = None,
		explanation_quality: str = "",
		explanation_quality_reasons: Optional[List[str]] = None,
		feature_impact_pct: Optional[Dict[str, float]] = None,
		feature_percentiles: Optional[Dict[str, float]] = None,
		technical_detail: str = "",
		contrastive_explanation: str = "",
	):
		self.optimizer_name = optimizer_name
		self.chosen_plan = chosen_plan if chosen_plan is not None else []
		self.top_features = top_features if top_features is not None else []
		self.metadata = metadata if metadata is not None else {}
		self.selected_plan_score = selected_plan_score
		self.compared_plans_count = compared_plans_count
		self.shap_used = shap_used
		self.lime_used = lime_used
		self.feature_values_for_selected_plan = feature_values_for_selected_plan if feature_values_for_selected_plan is not None else {}
		self.shap_values_for_selected_plan = shap_values_for_selected_plan if shap_values_for_selected_plan is not None else {}
		self.lime_values_for_selected_plan = lime_values_for_selected_plan if lime_values_for_selected_plan is not None else {}
		self.lime_top_features = lime_top_features if lime_top_features is not None else []
		self.natural_language_summary = natural_language_summary
		self.warnings = warnings if warnings is not None else []
		self.executive_summary = executive_summary
		self.factors_positive = factors_positive if factors_positive is not None else []
		self.factors_negative = factors_negative if factors_negative is not None else []
		self.runner_up_plan = runner_up_plan if runner_up_plan is not None else []
		self.runner_up_score = runner_up_score
		self.explanation_quality = explanation_quality
		self.explanation_quality_reasons = explanation_quality_reasons if explanation_quality_reasons is not None else []
		self.feature_impact_pct = feature_impact_pct if feature_impact_pct is not None else {}
		self.feature_percentiles = feature_percentiles if feature_percentiles is not None else {}
		self.technical_detail = technical_detail
		self.contrastive_explanation = contrastive_explanation

	def to_dict(self) -> Dict[str, object]:
		"""Exporta la explicacion en formato diccionario."""
		return {
			"optimizer_name": self.optimizer_name,
			"chosen_plan": self.chosen_plan,
			"top_features": self.top_features,
			"metadata": self.metadata,
			"selected_plan_score": self.selected_plan_score,
			"compared_plans_count": self.compared_plans_count,
			"shap_used": self.shap_used,
			"lime_used": self.lime_used,
			"feature_values_for_selected_plan": self.feature_values_for_selected_plan,
			"shap_values_for_selected_plan": self.shap_values_for_selected_plan,
			"lime_values_for_selected_plan": self.lime_values_for_selected_plan,
			"lime_top_features": self.lime_top_features,
			"natural_language_summary": self.natural_language_summary,
			"warnings": self.warnings,
			"executive_summary": self.executive_summary,
			"factors_positive": self.factors_positive,
			"factors_negative": self.factors_negative,
			"runner_up_plan": self.runner_up_plan,
			"runner_up_score": self.runner_up_score,
			"explanation_quality": self.explanation_quality,
			"explanation_quality_reasons": self.explanation_quality_reasons,
			"feature_impact_pct": self.feature_impact_pct,
			"feature_percentiles": self.feature_percentiles,
			"technical_detail": self.technical_detail,
			"contrastive_explanation": self.contrastive_explanation,
		}

	def __str__(self) -> str:
		"""Representacion simple y legible de la explicacion."""
		return (
			f"Explanation(optimizer_name={self.optimizer_name}, "
			f"chosen_plan={self.chosen_plan}, "
			f"top_features={self.top_features}, "
			f"shap_used={self.shap_used}, "
			f"lime_used={self.lime_used}, "
			f"selected_plan_score={self.selected_plan_score}, "
			f"explanation_quality={self.explanation_quality!r}, "
			f"metadata={self.metadata})"
		)
    

  
    
    
