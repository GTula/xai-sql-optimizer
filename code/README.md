# motor_SQL

A small relational database engine written in Python, built to study how SQL
optimizers choose join orders and how to explain those choices with SHAP.

It parses SQL into an AST, validates it, reorders joins with one of three
optimizers, and executes the query over CSV-backed tables. The `xai/` package
explains the optimizer's estimated cost for each candidate plan.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Python 3.9+. The core engine only uses the standard library; the dependencies
are for BayesCard and the XAI experiments. `xai/demo_explainer.py` also needs
`matplotlib`.

## Usage

```powershell
python main.py repl        # interactive console (statements end with ;)
python main.py example     # scripted example
python test_minidb.py      # end-to-end test of the engine
```

From Python:

```python
from database import Database

with Database("research.db", optimizer_type="selinger") as db:
    db.execute("CREATE TABLE department (id INTEGER, name VARCHAR(50))")
    db.execute("INSERT INTO department VALUES (1, 'Physics')")
    print(db.execute("SELECT id, name FROM department"))
```

`optimizer_type` can be `basic` (keeps the written order), `selinger`
(dynamic programming over row counts and distinct values) or `bayes`
(BayesCard estimates). Run `ANALYSE <table>;` first so the optimizers have
statistics.

Supported: `CREATE TABLE`, `INSERT`, `SELECT` with filters and multiple
`JOIN`s, `CREATE INDEX`, `ANALYSE`.

## Reproducing the paper

```powershell
python -m xai.research_evaluation --output-dir databases/research_evaluation --orders 4 --bootstrap 250
```

This generates the six-table synthetic schema, runs the 112-query workload for
both optimizers over 5 seeds, 3 sizes and 3 distributions, and writes the CSVs
with Ridge fidelity, SHAP stability and timing results. For a quick check:

```powershell
python -m xai.research_evaluation --output-dir databases/research_smoke --seeds 1 --sizes small --distributions uniform --orders 2 --query-limit 22 --bootstrap 20
```

Other scripts in `xai/`: `demo_explainer.py` (global SHAP importance),
`demo_bayes_runtime_validation.py` (estimates vs. runtime, q-error) and
`demo_batch_analysis.py` (runs `docs/CONSULTAS_100.sql`).

## Structure

| Folder | Contents |
| --- | --- |
| `sql/` | Tokenizer, parser, AST, validator and optimizers |
| `executor/` | Query execution |
| `catalog/` | Table metadata and statistics |
| `index/` | B-tree index |
| `bayescard/` | Adapted BayesCard estimator |
| `xai/` | Plan features, SHAP, explanations, runtime traces, experiments |
| `docs/` | Test workloads and file documentation |

Data is stored as CSV files under `databases/`.

## Limitations

This is a research prototype. Joins run as in-memory nested loops, indexes are
not used by `SELECT`, and `UPDATE`, `DELETE`, `ORDER BY`, `GROUP BY` and
subqueries are not implemented.

The engine produces structured explanations (SHAP values, main factors,
runner-up comparison). In the paper, these outputs are passed to an LLM
(Gemini) to write the natural-language summary; that step is not part of this
repository.

## References

Wu et al., *BayesCard: Revitalizing Bayesian Frameworks for Cardinality
Estimation*, 2020. [arXiv](https://arxiv.org/abs/2012.14743) ·
[original code](https://github.com/wuziniu/BayesCard)
