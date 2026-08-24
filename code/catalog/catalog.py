"""
FASE 2: CATALOG
Metadatos del sistema persistidos como tablas CSV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import csv
import os


@dataclass
class Column:
    """Definicion de una columna."""

    name: str
    type: str


@dataclass
class Index:
    """Definicion de un indice."""

    name: str
    table_name: str
    column_name: str
    index_file: str


@dataclass
class Table:
    """Definicion de una tabla en el catalogo."""

    name: str
    columns: List[Column]
    table_file: str

    def get_column(self, name: str) -> Optional[Column]:
        for col in self.columns:
            if col.name == name:
                return col
        return None

    def get_column_index(self, name: str) -> Optional[int]:
        for i, col in enumerate(self.columns):
            if col.name == name:
                return i
        return None


class Catalog:
    """Catalogo persistido en CSV como una base de datos de metadatos."""

    DATABASES_TABLE = "databases.csv"
    TABLES_TABLE = "tables.csv"
    COLUMNS_TABLE = "columns.csv"
    INDEXES_TABLE = "indexes.csv"
    STATISTICS_TABLE = "statistics.csv"
    COLUMN_STATS_TABLE = "column_stats.csv"

    def __init__(self, catalog_dir: str, db_name: str):
        self.catalog_dir = catalog_dir
        self.db_name = db_name
        self.tables: Dict[str, Table] = {}
        self.indices: Dict[str, Index] = {}

        os.makedirs(self.catalog_dir, exist_ok=True)
        self._ensure_catalog_tables()
        self._ensure_database_registered()
        self.load()

    @property
    def databases_path(self) -> str:
        return os.path.join(self.catalog_dir, self.DATABASES_TABLE)

    @property
    def tables_path(self) -> str:
        return os.path.join(self.catalog_dir, self.TABLES_TABLE)

    @property
    def columns_path(self) -> str:
        return os.path.join(self.catalog_dir, self.COLUMNS_TABLE)

    @property
    def indexes_path(self) -> str:
        return os.path.join(self.catalog_dir, self.INDEXES_TABLE)

    @property
    def statistics_path(self) -> str:
        return os.path.join(self.catalog_dir, self.STATISTICS_TABLE)

    @property
    def column_stats_path(self) -> str:
        return os.path.join(self.catalog_dir, self.COLUMN_STATS_TABLE)

    def create_table(self, table_name: str, columns: List[Column], table_file: str) -> Table:
        if table_name in self.tables:
            raise ValueError(f"Tabla '{table_name}' ya existe")

        table = Table(name=table_name, columns=columns, table_file=table_file)

        # Registrar tabla en metadatos
        self._append_row(
            self.tables_path,
            {
                "db_name": self.db_name,
                "table_name": table_name,
                "table_path": table_file,
            },
        )

        for position, col in enumerate(columns):
            self._append_row(
                self.columns_path,
                {
                    "db_name": self.db_name,
                    "table_name": table_name,
                    "column_name": col.name,
                    "column_type": col.type,
                    "column_position": str(position),
                },
            )

        self.tables[table_name] = table
        return table

    def get_table(self, table_name: str) -> Optional[Table]:
        return self.tables.get(table_name)

    def get_table_row_count(self, table_name: str) -> int:
        try:
            rows = self._read_rows(self.statistics_path)
            for row in rows:
                if row["db_name"] == self.db_name and row["table_name"] == table_name:
                    return int(row["row_count"])
        except Exception:
            pass
        return 1

    def create_index(self, index_name: str, table_name: str, column_name: str, index_file: str) -> Index:
        if index_name in self.indices:
            raise ValueError(f"Indice '{index_name}' ya existe")

        if table_name not in self.tables:
            raise ValueError(f"Tabla '{table_name}' no existe")

        if self.tables[table_name].get_column(column_name) is None:
            raise ValueError(f"Columna '{column_name}' no existe en '{table_name}'")

        index = Index(
            name=index_name,
            table_name=table_name,
            column_name=column_name,
            index_file=index_file,
        )

        self._append_row(
            self.indexes_path,
            {
                "db_name": self.db_name,
                "index_name": index_name,
                "table_name": table_name,
                "column_name": column_name,
                "index_path": index_file,
            },
        )

        self.indices[index_name] = index
        return index

    def get_index(self, index_name: str) -> Optional[Index]:
        return self.indices.get(index_name)

    def load(self) -> None:
        self.tables = {}
        self.indices = {}

        table_rows = [r for r in self._read_rows(self.tables_path) if r["db_name"] == self.db_name]
        column_rows = [r for r in self._read_rows(self.columns_path) if r["db_name"] == self.db_name]
        index_rows = [r for r in self._read_rows(self.indexes_path) if r["db_name"] == self.db_name]

        columns_by_table: Dict[str, List[Column]] = {}
        for row in sorted(column_rows, key=lambda r: int(r["column_position"])):
            columns_by_table.setdefault(row["table_name"], []).append(
                Column(name=row["column_name"], type=row["column_type"])
            )

        for row in table_rows:
            table_name = row["table_name"]
            self.tables[table_name] = Table(
                name=table_name,
                columns=columns_by_table.get(table_name, []),
                table_file=row["table_path"],
            )

        for row in index_rows:
            self.indices[row["index_name"]] = Index(
                name=row["index_name"],
                table_name=row["table_name"],
                column_name=row["column_name"],
                index_file=row["index_path"],
            )

    def update_table_row_count(self, table_name: str, row_count: int) -> None:
        rows = self._read_rows(self.statistics_path)
        updated = False

        for row in rows:
            if row["db_name"] == self.db_name and row["table_name"] == table_name:
                row["row_count"] = str(row_count)
                updated = True
                break

        if not updated:
            rows.append(
                {
                    "db_name": self.db_name,
                    "table_name": table_name,
                    "row_count": str(row_count),
                }
            )

        self._write_rows(self.statistics_path, ["db_name", "table_name", "row_count"], rows)

    def update_column_ndv(self, table_name: str, column_name: str, ndv: int) -> None:
        self._ensure_csv(self.column_stats_path, ["db_name", "table_name", "column_name", "ndv"])
        rows = self._read_rows(self.column_stats_path)
        updated = False
        for r in rows:
            if r["db_name"] == self.db_name and r["table_name"] == table_name and r["column_name"] == column_name:
                r["ndv"] = str(ndv)
                updated = True
                break

        if not updated:
            rows.append(
                {
                    "db_name": self.db_name,
                    "table_name": table_name,
                    "column_name": column_name,
                    "ndv": str(ndv),
                }
            )

        self._write_rows(self.column_stats_path, ["db_name", "table_name", "column_name", "ndv"], rows)

    def get_column_ndv(self, table_name: str, column_name: str) -> int:
        try:
            self._ensure_csv(self.column_stats_path, ["db_name", "table_name", "column_name", "ndv"])
            rows = self._read_rows(self.column_stats_path)
            for r in rows:
                if r["db_name"] == self.db_name and r["table_name"] == table_name and r["column_name"] == column_name:
                    return int(r["ndv"])
        except Exception:
            pass
        return 1

    def _ensure_catalog_tables(self) -> None:
        self._ensure_csv(self.databases_path, ["db_name", "db_path"])
        self._ensure_csv(self.tables_path, ["db_name", "table_name", "table_path"])
        self._ensure_csv(
            self.columns_path,
            ["db_name", "table_name", "column_name", "column_type", "column_position"],
        )
        self._ensure_csv(
            self.indexes_path,
            ["db_name", "index_name", "table_name", "column_name", "index_path"],
        )
        self._ensure_csv(self.statistics_path, ["db_name", "table_name", "row_count"])
        self._ensure_csv(self.column_stats_path, ["db_name", "table_name", "column_name", "ndv"])

    def _ensure_database_registered(self) -> None:
        rows = self._read_rows(self.databases_path)
        already_exists = any(row["db_name"] == self.db_name for row in rows)
        if already_exists:
            return

        self._append_row(
            self.databases_path,
            {
                "db_name": self.db_name,
                "db_path": os.path.abspath(os.path.join(self.catalog_dir, "..", self.db_name)),
            },
        )

    def _ensure_csv(self, csv_path: str, fieldnames: List[str]) -> None:
        if os.path.exists(csv_path):
            return

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def _read_rows(self, csv_path: str) -> List[Dict[str, str]]:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def _append_row(self, csv_path: str, row: Dict[str, str]) -> None:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            fieldnames = list(csv.DictReader(f).fieldnames or [])

        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

    def _write_rows(self, csv_path: str, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
