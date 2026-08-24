"""
Demo de análisis batch de consultas SQL.

Ejecuta el batch_explainer con el archivo CONSULTAS_100.sql y muestra
cómo todas las consultas se persisten en archivos consolidados.

Uso:
    python demo_batch_analysis.py

Archivos generados en databases/feature_analysis/:
    - summary_CONSULTAS_100_{timestamp}.csv
    - estimated_CONSULTAS_100_batch_{timestamp}.csv (features estimadas)
    - runtime_CONSULTAS_100_batch_{timestamp}.csv (features reales)
    - metadata_CONSULTAS_100_{timestamp}.json
"""

from __future__ import annotations

import os
import sys

# Ajustar path para importar desde la raíz del proyecto
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from xai.batch_explainer import run_batch_analysis


def main() -> None:
    """Ejecuta el análisis batch de las 100 consultas demo."""
    
    queries_file = "docs/CONSULTAS_100.sql"
    
    if not os.path.exists(queries_file):
        print(f"Error: Archivo {queries_file} no encontrado.")
        return
    
    print("Iniciando análisis batch de consultas...")
    print(f"Archivo de entrada: {queries_file}\n")
    
    try:
        summary = run_batch_analysis(
            queries_file=queries_file,
            output_dir="databases/feature_analysis",
            dataset_name="CONSULTAS_100",
            optimizers=["selinger", "bayes"],
            max_plans=50,
            verbose=True,
        )
        
        print("\n" + "=" * 60)
        print("ANÁLISIS COMPLETADO")
        print("=" * 60)
        print(f"\nDataset           : {summary.dataset_name}")
        print(f"Consultas totales : {summary.total_queries}")
        print(f"Exitosas          : {summary.successful_queries}")
        print(f"Fallidas          : {summary.failed_queries}")
        print(f"Duración          : {summary.total_duration_seconds:.2f}s")
        
        print(f"\nArchivos generados en: {summary.output_directory}")
        print(f"  1. {summary.summary_csv}")
        print(f"     (Resumen de todas las consultas)")
        
        if summary.estimated_features_csv:
            print(f"  2. {summary.estimated_features_csv}")
            print(f"     (Features estimadas consolidadas)")
        
        if summary.runtime_features_csv:
            print(f"  3. {summary.runtime_features_csv}")
            print(f"     (Features reales consolidadas)")
        
        print(f"  4. {summary.metadata_json}")
        print(f"     (Metadata del batch)")
        
        print("\nPuedes analizar los archivos con pandas:")
        print("  import pandas as pd")
        if summary.estimated_features_csv:
            print(f"  df_est = pd.read_csv('{summary.output_directory}/{summary.estimated_features_csv}')")
            print("  df_est.groupby(['optimizer', 'query_id']).size()")
        if summary.runtime_features_csv:
            print(f"  df_rt = pd.read_csv('{summary.output_directory}/{summary.runtime_features_csv}')")
            print("  df_rt.groupby('optimizer')['fidelity_score'].mean()")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"\nError durante el análisis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
