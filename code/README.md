# motor_SQL — Explainable SQL Engine

`motor_SQL` is a relational database prototype written in Python. The project
covers the complete query-processing pipeline: tokenization, AST construction,
semantic validation, `JOIN` order optimization, execution, and persistence.

In addition to the basic engine, it includes an experimental research track on
cardinality estimation and optimizer explainability. It compares a basic
execution order, a Selinger-inspired optimizer, and a BayesCard integration. The
XAI layer analyzes why a plan was selected and compares its estimates with
metrics observed at runtime.

## Features

- Custom SQL parser and validator.
- Execution of `SELECT`, `INSERT`, `CREATE TABLE`, `CREATE INDEX`, and `ANALYSE`.
- Queries with filters, Boolean expressions, and multiple `JOIN` clauses.
- Support for `INTEGER`, `VARCHAR`, `FLOAT`, `BOOLEAN`, and `DATE` types.
- File-based persistence for data, catalog metadata, statistics, and indexes.
- Three join-ordering strategies:
  - `basic`: preserves the order written in the query;
  - `selinger`: uses dynamic programming, cardinalities, and the number of
    distinct values (NDV);
  - `bayes`: uses BayesCard-based cardinality estimates.
- Local plan explanations using SHAP, including a natural-language summary,
  positive and negative factors, and a comparison with the runner-up plan.
- Instrumented execution for comparing estimated and observed features.
- Batch query analysis with CSV and JSON result persistence.

## Query Pipeline

```text
SQL
 └─> Tokenizer ─> Parser/AST ─> Validator ─> Optimizer ─> ExecutionEngine
                             catalog/statistics ↑       └─> CSV tables
                                        XAI ────────────> explanations and traces
```

The `Database` class is the facade that connects these components. When a
statement is executed, `ExecutionEngine` parses and validates it. For `SELECT`
queries, the optimizer may reorder joins before the engine executes them using
nested loops.

## Requirements and Installation

- Python 3.9 or later.
- A virtual environment is recommended.

Using PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The core engine mainly uses the Python standard library. The dependencies in
`requirements.txt` are required for BayesCard and the XAI experiments. The demo
that generates global feature-importance charts also requires `matplotlib`:

```powershell
python -m pip install matplotlib
```

## Quick Start

Run the main workflow test:

```powershell
python .\test_minidb.py
```

Run the programmatic example:

```powershell
python .\main.py example
```

Open the interactive console:

```powershell
python .\main.py repl
```

You can also provide the logical database name:

```powershell
python .\main.py repl my_database.db
```

Inside the REPL, every statement must end with `;`. Use `HELP;` to display help
and `EXIT;` to close the session.

```sql
CREATE TABLE instructor (
    id INTEGER,
    name VARCHAR(50),
    dept VARCHAR(30),
    salary FLOAT
);

INSERT INTO instructor VALUES
    (1, 'Einstein', 'Physics', 95000),
    (2, 'Mozart', 'Music', 40000),
    (3, 'Darwin', 'Biology', 60000);

ANALYSE instructor;

SELECT name, salary
FROM instructor
WHERE salary > 50000;
```

## Python API

```python
from database import Database

with Database("research.db", optimizer_type="selinger") as db:
    db.execute("CREATE TABLE department (id INTEGER, name VARCHAR(50))")
    db.execute("INSERT INTO department VALUES (1, 'Physics')")
    rows = db.execute("SELECT id, name FROM department")
    print(rows)
```

`optimizer_type` accepts `basic`, `selinger`, `bayes`, or `bayescard`. Running
`ANALYSE [table]` before experimenting with the optimizers updates the
cardinalities and NDVs used by their cost models.

## Explainability and Experiments

The `xai/` package treats the optimizer as the system being explained without
changing its decisions. For each query containing joins, it can enumerate
candidate plans, extract structural and cost features, calculate SHAP
contributions, and produce a contrastive explanation comparing the selected plan
with its closest alternative.

The main experiments are:

```powershell
# Global feature importance across the query workload
python .\xai\demo_explainer.py

# Comparison between Bayes estimates and runtime work
python .\xai\demo_bayes_runtime_validation.py

# Batch processing of docs/CONSULTAS_100.sql
python .\xai\demo_batch_analysis.py
```

The batch analysis writes its results to `databases/feature_analysis/`. These
outputs include estimated features for each plan, runtime traces, a query
summary, and experiment metadata. Fidelity is evaluated by comparing estimated
and observed work, including q-error and the intermediate rows produced by each
join.

## Persistence

Opening `Database("my_database.db")` creates the following structure next to the
provided path:

```text
databases/
├── catalog/
│   ├── databases.csv
│   ├── tables.csv
│   ├── columns.csv
│   ├── indexes.csv
│   ├── statistics.csv
│   └── column_stats.csv
└── my_database/
    ├── <table>.csv
    └── <index>.idx
```

The `.db` argument identifies the database, but data is currently stored in this
CSV and index hierarchy rather than inside a single database file.

## Repository Structure

| Path | Responsibility |
| --- | --- |
| `database.py` | Public facade for the engine. |
| `sql/` | Tokenizer, parser, AST, validation, and optimizers. |
| `catalog/` | CSV-based metadata and statistics. |
| `executor/` | AST execution and physical operators. |
| `index/` | Persistent B-tree implementation. |
| `bayescard/` | BayesCard estimator adaptation and source code. |
| `xai/` | Features, SHAP, explanations, traces, and batch analysis. |
| `demo_bn_db/` | Small dataset used by the demonstrations. |
| `docs/` | Technical inventory, explainability notes, and test queries. |

For a file-by-file inventory, see
[`docs/DOCUMENTACION_ARCHIVOS.md`](docs/DOCUMENTACION_ARCHIVOS.md).

## Current Scope and Limitations

This is an educational and research prototype, not a production DBMS.

- Tables are read from CSV files and joins are executed in memory using nested
  loops. There is no page-based storage or buffer pool.
- `CREATE INDEX` registers the index and creates its B-tree file, but it does not
  yet load existing rows or provide an access path for `SELECT`.
- The parser defines structures for `UPDATE` and `DELETE`, but the execution
  engine does not yet implement these statements.
- `ORDER BY`, `GROUP BY`, subqueries, and transactions are not currently
  supported.
- Experimental validation shows that the Bayes model may assume filters are
  applied before joins, while the current executor evaluates `WHERE` afterward.
  Runtime traces exist specifically to measure this difference.

## Academic Background

The probabilistic integration is based on **BayesCard: Revitilizing Bayesian
Frameworks for Cardinality Estimation**, by Ziniu Wu, Amir Shaikhha, Rong Zhu,
Kai Zeng, Yuxing Han, and Jingren Zhou. The paper proposes using Bayesian
networks to provide accurate, interpretable, and efficient cardinality estimates
for query optimization.

- [Paper on arXiv](https://arxiv.org/abs/2012.14743)
- [Original BayesCard implementation](https://github.com/wuziniu/BayesCard)
