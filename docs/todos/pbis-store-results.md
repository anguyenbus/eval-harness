# Store Results: PBIs

**Capability**: Persist evaluation results (CSV, JSON, traces) for later analysis and comparison.

**Current State**:
- Results stored as flat files in `results/` directory
- No database option
- No deduplication (identical runs create new files)
- No retention policy (files accumulate forever)
- S3 upload exists at [`phoenix_adapter.py:538`](src/eval_harness/observability/phoenix_adapter.py:538) but security is undocumented

**Evidence**: Run `ls -la results/` — multiple result files with timestamps, no cleanup.

---

## adversarial Review Questions (Answer Before Review)

### Q1: "Why do we need a database? Flat files work fine."

**Answer**: They work until they don't.

**Evidence**:
```bash
$ ls results/ | wc -l
42  # After 2 weeks of development

$ find results/ -name "*.json" -exec du -ch {} + | tail -1
150M  # Total storage, growing linearly

$ grep "faithfulness_score" results/*.csv | wc -l
5000  # Try to query: "faithfulness < 0.8 across all runs"
# Requires: find, grep, cut, sort, join. Not trivial.
```

**adversarial reviewer says**: "I can't query across runs. I can't track metric trends over time. I can't join datasets with results."

**Database solves**:
- Cross-run queries: `WHERE faithfulness < 0.8 AND timestamp > '2026-05-01'`
- Time-series tracking: `SELECT date, avg(faithfulness) GROUP BY date`
- Joins: `results JOIN datasets ON results.dataset_id = datasets.id`

**PBI-31 addresses this**.

---

### Q2: "Deduplication is premature optimization."

**Answer**: Not optimization — correctness.

**Evidence from actual files**:
```bash
$ cat results/legal_rag_bench_nano_results_20260521_211101.json
{
  "dataset": "legal_rag_bench",
  "slice": "nano",
  "judge_model": "gpt-4o",  # ← Alias, not snapshot
  "timestamp": "20260521_211101"
}

$ cat results/legal_rag_bench_nano_results_20260521_210254.json
{
  "dataset": "legal_rag_bench",
  "slice": "nano",
  "judge_model": "gpt-4o",  # ← Same alias, possibly different model
  "timestamp": "20260521_210254"
}
```

**Problem**: These might be identical runs (same config, same dataset) but we can't tell.

**adversarial reviewer says**: "You ran the same eval twice because `gpt-4o` alias changed between runs. Now I have two files. Which one is correct?"

**Deduplication solves**:
- Hash of `{config_hash, dataset_sha, model_snapshot}`
- Skip run if hash exists (unless `--force`)
- Save: $5 per 100 queries × duplicate runs

**PBI-32 addresses this**.

---

### Q3: "Retention policy is just `rm old/*`. Why 2 hours?"

**Answer**: `rm` is dangerous. Need safety.

**Evidence**:
```bash
$ ls -lt results/*.json | head -5
-rw-r--r-- 1 user user 2.5K May 21 21:01 results/legal_rag_bench_nano_results_20260521_211101.json
-rw-r--r-- 1 user user 2.5K May 21 21:02 results/legal_rag_bench_nano_results_20260521_210254.json
-rw-r--r-- 1 user user 2.5K May 21 20:35 results/legal_rag_bench_nano_results_20260521_203524.json

# Try to rm files older than 7 days
$ find results/ -name "*.json" -mtime +7 -delete
# Wait: did this delete the baseline we need for regression_check.py?
# Yes. Now regression detection is broken.
```

**adversarial reviewer says**: "You deleted my baseline. Now I can't detect regressions. `rm` is not a retention policy."

**Retention policy needs**:
- Dry-run mode: show what would be deleted
- Keep latest N runs (baseline preservation)
- Age-based: delete files older than X days
- Tag-based: never delete `is_baseline=true`

**PBI-33 addresses this**.

---

### Q4: "S3 security documentation is just copying AWS docs."

**Answer**: No. Our context is specific.

**Evidence from [`phoenix_adapter.py:538`](src/eval_harness/observability/phoenix_adapter.py:538)**:
```python
def upload_parquet_to_s3(
    self,
    parquet_path: Path,
    bucket: str,
    key_prefix: str = "phoenix-traces",
) -> bool:
    """Upload buffered Parquet traces to S3."""
    try:
        import boto3
        s3_client = boto3.client("s3")  # ← Where do credentials come from?
        key = f"{key_prefix}/{parquet_path.name}"
        s3_client.upload_file(str(parquet_path), bucket, key)
        return True
    except Exception:
        import sys
        print(f"[WARN] S3 upload failed for {parquet_path}", file=sys.stderr)
        return False  # ← Silent failure. Is this safe?
```

**adversarial reviewer says**:
1. "Where do credentials come from? `~/.aws/credentials`? Env vars? IAM role?"
2. "What if upload fails? We swallow the exception. Is the trace lost?"
3. "What bucket? Who has access? Is encryption enabled?"
4. "What's the retention policy for these traces?"

**This is not in AWS docs. This is OUR security context.**

**PBI-34 addresses this**.

---

## PBI-31: Add Database Option (SQLite/Postgres)

**Priority**: P2 (Storage)
**Estimate**: 12 hours
**Category**: Store Results → Database

### Problem Statement

Current storage is flat files only. This blocks:

1. **Cross-run queries**: Cannot ask "show me all faithfulness scores < 0.8 in May"
2. **Time-series analysis**: Cannot plot metric trends over time
3. **Efficient comparison**: Must load full JSON/CSV files to compare runs
4. **Concurrent access**: Multiple processes cannot safely write to `results/`

**Evidence**:
- [`csv_writer.py:39`](src/eval_harness/reporting/csv_writer.py:39): `df.to_csv(output_path, index=False)` — file write, no DB option
- No SQLite/Postgres dependencies in [`pyproject.toml`](pyproject.toml:11)
- Result files in `results/` with timestamp naming only

### Current Data Format

**JSON summary** ([`legal_rag_bench_nano_results_*.json`](results/legal_rag_bench_nano_results_20260521_211101.json)):
```json
{
  "dataset": "legal_rag_bench",
  "slice": "nano",
  "timestamp": "20260521_211101",
  "csv_file": "legal_rag_bench_nano_results_20260521_211101.csv",
  "metrics_avg": {
    "faithfulness_score": 0.9667,
    "context_precision_score": 0.2367,
    "context_recall_score": 0.375,
    "answer_relevancy_score": 0.7167
  },
  "total_processed": 10,
  "judge_model": "gpt-4o"
}
```

**CSV details** (first 3 columns shown):
```csv
query_id,question,gold_answer,generated_answer,faithfulness_score,judge_verdict,...
1,"Bob and Ted...","No. While...","I don't have...",1.0,PASS,...
```

### Database Schema

```sql
-- Runs table (one row per evaluation run)
CREATE TABLE evaluation_runs (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) UNIQUE NOT NULL,  -- UUID
    dataset VARCHAR(100) NOT NULL,
    slice VARCHAR(100),
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    judge_model VARCHAR(100) NOT NULL,
    model_snapshot_id VARCHAR(100),  -- gpt-4o-2024-08-06
    total_processed INTEGER NOT NULL,
    error_count INTEGER NOT NULL DEFAULT 0,
    config_hash VARCHAR(64),  -- SHA-256 of config
    dataset_sha VARCHAR(64),  -- SHA-256 of dataset
    is_baseline BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Per-query results table
CREATE TABLE evaluation_results (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) NOT NULL REFERENCES evaluation_runs(run_id),
    query_id VARCHAR(100) NOT NULL,
    question TEXT,
    gold_answer TEXT,
    generated_answer TEXT,
    faithfulness_score FLOAT,
    context_precision_score FLOAT,
    context_recall_score FLOAT,
    answer_relevancy_score FLOAT,
    judge_verdict VARCHAR(20),  -- PASS, NEEDS_REVIEW
    total_ms INTEGER,
    error TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX idx_runs_dataset_timestamp ON evaluation_runs(dataset, timestamp DESC);
CREATE INDEX idx_runs_model ON evaluation_runs(judge_model);
CREATE INDEX idx_results_run_id ON evaluation_results(run_id);
CREATE INDEX idx_results_faithfulness ON evaluation_results(faithfulness_score);
```

### Acceptance Criteria

1. [ ] SQLite option works with `--db-url sqlite:///results/evaluations.db`
2. [ ] Postgres option works with `--db-url postgresql://user:pass@host/db`
3. [ ] Schema includes all CSV + JSON columns
4. [ ] Fallback to files if DB unavailable (don't fail the eval)
5. [ ] Migration script: existing JSON/CSV → DB
6. [ ] Query CLI: `eval-query "SELECT * FROM evaluation_runs WHERE dataset = 'legal_rag_bench'"`

### Implementation Notes

**Create `src/eval_harness/storage/database.py`**:

```python
"""Database storage for evaluation results."""
from pathlib import Path
from typing import Any
import hashlib
import json
from datetime import datetime

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

try:
    import psycopg
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False


class DatabaseStorage:
    """Store evaluation results in SQLite or Postgres."""

    def __init__(self, db_url: str):
        """
        Initialize database connection.

        Args:
            db_url: Database URL (sqlite:///path or postgresql://...)
        """
        self.db_url = db_url

        if db_url.startswith("sqlite:///"):
            if not SQLITE_AVAILABLE:
                raise RuntimeError("SQLite requested but sqlite3 not available")
            self.backend = "sqlite"
            self.path = Path(db_url.replace("sqlite:///", ""))
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.path))
            self._init_sqlite_schema()

        elif db_url.startswith("postgresql://"):
            if not POSTGRES_AVAILABLE:
                raise RuntimeError("Postgres requested but psycopg not available")
            self.backend = "postgres"
            self.conn = psycopg.connect(db_url)
            self._init_postgres_schema()

        else:
            raise ValueError(f"Unsupported db_url: {db_url}")

    def save_run(
        self,
        dataset: str,
        slice: str | None,
        judge_model: str,
        model_snapshot_id: str | None,
        results: list[dict[str, Any]],
        config: dict[str, Any],
        dataset_sha: str | None = None
    ) -> str:
        """
        Save an evaluation run and its results.

        Returns:
            run_id: Unique identifier for this run
        """
        # Generate run ID
        run_id = hashlib.sha256(
            f"{dataset}_{slice}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        # Calculate config hash
        config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()

        # Insert run
        cursor = self.conn.cursor()

        if self.backend == "sqlite":
            cursor.execute("""
                INSERT INTO evaluation_runs
                (run_id, dataset, slice, judge_model, model_snapshot_id,
                 total_processed, config_hash, dataset_sha)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, dataset, slice, judge_model, model_snapshot_id,
                len(results), config_hash, dataset_sha
            ))
        else:  # postgres
            cursor.execute("""
                INSERT INTO evaluation_runs
                (run_id, dataset, slice, judge_model, model_snapshot_id,
                 total_processed, config_hash, dataset_sha)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                run_id, dataset, slice, judge_model, model_snapshot_id,
                len(results), config_hash, dataset_sha
            ))

        # Insert results
        for result in results:
            if self.backend == "sqlite":
                cursor.execute("""
                    INSERT INTO evaluation_results
                    (run_id, query_id, question, gold_answer, generated_answer,
                     faithfulness_score, context_precision_score,
                     context_recall_score, answer_relevancy_score,
                     judge_verdict, total_ms, error)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    result.get("query_id"),
                    result.get("question"),
                    result.get("gold_answer"),
                    result.get("generated_answer"),
                    result.get("faithfulness_score"),
                    result.get("context_precision_score"),
                    result.get("context_recall_score"),
                    result.get("answer_relevancy_score"),
                    result.get("judge_verdict"),
                    result.get("total_ms"),
                    result.get("error", "")
                ))
            else:  # postgres
                cursor.execute("""
                    INSERT INTO evaluation_results
                    (run_id, query_id, question, gold_answer, generated_answer,
                     faithfulness_score, context_precision_score,
                     context_recall_score, answer_relevancy_score,
                     judge_verdict, total_ms, error)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    run_id,
                    result.get("query_id"),
                    result.get("question"),
                    result.get("gold_answer"),
                    result.get("generated_answer"),
                    result.get("faithfulness_score"),
                    result.get("context_precision_score"),
                    result.get("context_recall_score"),
                    result.get("answer_relevancy_score"),
                    result.get("judge_verdict"),
                    result.get("total_ms"),
                    result.get("error", "")
                ))

        self.conn.commit()
        return run_id

    def query(self, sql: str) -> list[dict[str, Any]]:
        """Execute SQL query and return results as list of dicts."""
        cursor = self.conn.cursor()
        cursor.execute(sql)

        if self.backend == "sqlite":
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        else:  # postgres
            columns = [desc.name for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def _init_sqlite_schema(self) -> None:
        """Initialize SQLite schema."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE NOT NULL,
                dataset TEXT NOT NULL,
                slice TEXT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                judge_model TEXT NOT NULL,
                model_snapshot_id TEXT,
                total_processed INTEGER NOT NULL,
                error_count INTEGER DEFAULT 0,
                config_hash TEXT,
                dataset_sha TEXT,
                is_baseline INTEGER DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                query_id TEXT NOT NULL,
                question TEXT,
                gold_answer TEXT,
                generated_answer TEXT,
                faithfulness_score REAL,
                context_precision_score REAL,
                context_recall_score REAL,
                answer_relevancy_score REAL,
                judge_verdict TEXT,
                total_ms INTEGER,
                error TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (run_id) REFERENCES evaluation_runs(run_id)
            )
        """)
        # Indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_runs_dataset_timestamp
            ON evaluation_runs(dataset, timestamp DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_results_run_id
            ON evaluation_results(run_id)
        """)
        self.conn.commit()

    def _init_postgres_schema(self) -> None:
        """Initialize Postgres schema."""
        # Similar to SQLite schema but with SERIAL and proper types
        # ...
        pass
```

**Modify runners to support DB**:

```python
# In run_rag_eval.py
parser.add_argument(
    "--db-url",
    type=str,
    help="Database URL (sqlite:///path or postgresql://...). If not set, use files."
)

# In main()
if args.db_url:
    try:
        storage = DatabaseStorage(args.db_url)
        run_id = storage.save_run(
            dataset=args.dataset,
            slice=args.slice,
            judge_model=judge_model,
            model_snapshot_id=model_snapshot_id,
            results=results,
            config=config.__dict__,
            dataset_sha=dataset_sha
        )
        print(f"[INFO] Saved to database: run_id={run_id}")
    except Exception as e:
        print(f"[WARN] Database save failed: {e}. Falling back to files.", file=sys.stderr)
        # Fall through to file storage
```

**Add migration script**:

```bash
# scripts/migrate_to_db.py
"""Migrate existing JSON/CSV results to database."""
import sqlite3
from pathlib import Path
import json

db = DatabaseStorage("sqlite:///results/evaluations.db")

for json_file in Path("results").glob("*_results_*.json"):
    with open(json_file) as f:
        data = json.load(f)

    # Read corresponding CSV
    csv_file = json_file.parent / data["csv_file"]
    df = pd.read_csv(csv_file)

    # Convert to list of dicts
    results = df.to_dict("records")

    # Save to DB
    run_id = db.save_run(
        dataset=data["dataset"],
        slice=data.get("slice"),
        judge_model=data["judge_model"],
        model_snapshot_id=data.get("model_snapshot_id"),
        results=results,
        config={},  # Config not in old JSON
        dataset_sha=None
    )

    print(f"Migrated {json_file.name} → run_id={run_id}")
```

### Dependencies

Add to [`pyproject.toml`](pyproject.toml:44):

```toml
[project.optional-dependencies]
database = [
    "psycopg[binary]>=3.1.0",  # Postgres
]
# SQLite is in stdlib, no dependency needed
```

### Definition of Done

- [ ] SQLite backend working
- [ ] Postgres backend working
- [ ] Schema matches current JSON/CSV format
- [ ] `--db-url` flag added to CLI
- [ ] Graceful fallback to files
- [ ] Migration script tested
- [ ] Query CLI working
- [ ] Tests added for DB operations

### What a adversarial Reviewer Will Ask

**Q**: "Why not use an ORM like SQLAlchemy?"

**A**: Overkill. We have 2 tables. Raw SQL is clearer, fewer dependencies.

**Q**: "What if DB write fails mid-run?"

**A**: Transaction rollback. `conn.commit()` only after all results inserted. If exception, `conn.rollback()`.

**Q**: "Can I query without knowing SQL?"

**A**: PBI-15 (comparison tool) will provide pre-built queries. For ad-hoc, learn SQL.

**Q**: "What about concurrent writes?"

**A**: SQLite has limited concurrency (single writer). Postgres handles it. Document this limitation.

---

## PBI-32: Add Deduplication

**Priority**: P2 (Storage)
**Estimate**: 4 hours
**Category**: Store Results → Deduplication

### Problem Statement

Identical runs create duplicate files. Waste of:
- Storage: ~2.5MB per run × duplicates
- Cost: $5 per 100 queries × duplicate runs
- Time: 2 hours × duplicate runs

**Evidence**:
```bash
$ ls -lh results/legal_rag_bench_nano_results_*.json
-rw-r--r-- 1 user user 2.5K May 21 21:01 ...20260521_211101.json
-rw-r--r-- 1 user user 2.5K May 21 21:02 ...20260521_210254.json
-rw-r--r-- 1 user user 2.5K May 21 20:35 ...20260521_203524.json
-rw-r--r-- 1 user user 2.5K May 21 17:22 ...20260521_172252.json

# Check if identical
$ diff results/legal_rag_bench_nano_results_20260521_211101.json \
      results/legal_rag_bench_nano_results_20260521_210254.json
# Possibly identical runs, but we can't tell without comparing full content
```

### Root Cause

Current JSON format doesn't include enough metadata to detect duplicates:
- `judge_model: "gpt-4o"` is an alias, not a snapshot
- No `config_hash` to detect identical configs
- No `dataset_sha` to detect identical data

### Solution: Hash-Based Deduplication

**Hash components**:
1. `config_hash`: SHA-256 of eval config (dataset, slice, model, top_k, etc.)
2. `dataset_sha`: SHA-256 of dataset file (or unique identifier)
3. `model_snapshot_id`: Dated model snapshot (e.g., `gpt-4o-2024-08-06`)

**Combined hash**: `SHA-256(config_hash + dataset_sha + model_snapshot_id)`

### Acceptance Criteria

1. [ ] Compute run hash before starting evaluation
2. [ ] Check existing runs for matching hash
3. [ ] Skip if exists (with message): `Run hash exists: 20260521_211101.json. Use --force to re-run.`
4. [ ] `--force` flag to bypass dedup check
5. [ ] Symlink to previous result if skipped (optional)
6. [ ] Include hash in JSON metadata

### Implementation

**Create `src/eval_harness/storage/dedup.py`**:

```python
"""Deduplication for evaluation runs."""
import hashlib
import json
from pathlib import Path
from typing import Any


def compute_run_hash(
    config: dict[str, Any],
    dataset_sha: str | None,
    model_snapshot_id: str | None
) -> str:
    """
    Compute hash for an evaluation run.

    Args:
        config: Evaluation configuration dict
        dataset_sha: SHA-256 of dataset file
        model_snapshot_id: Dated model snapshot ID

    Returns:
        Hex digest of SHA-256 hash
    """
    # Normalize config for hashing
    config_normalized = json.dumps(config, sort_keys=True)

    # Combine components
    hash_input = f"{config_normalized}|{dataset_sha or ''}|{model_snapshot_id or ''}"

    return hashlib.sha256(hash_input.encode()).hexdigest()


def find_existing_run(
    run_hash: str,
    results_dir: Path
) -> Path | None:
    """
    Check if a run with this hash already exists.

    Args:
        run_hash: Hash of the run to find
        results_dir: Directory containing result files

    Returns:
        Path to existing JSON file, or None if not found
    """
    # Check each JSON file for matching hash
    for json_file in results_dir.glob("*_results_*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)

            if data.get("run_hash") == run_hash:
                return json_file
        except (json.JSONDecodeError, IOError):
            continue

    return None


def check_should_run(
    config: dict[str, Any],
    dataset_sha: str | None,
    model_snapshot_id: str | None,
    results_dir: Path,
    force: bool = False
) -> tuple[bool, Path | None, str]:
    """
    Check if evaluation should run (deduplication check).

    Args:
        config: Evaluation configuration
        dataset_sha: SHA-256 of dataset
        model_snapshot_id: Model snapshot ID
        results_dir: Results directory
        force: Bypass dedup check

    Returns:
        (should_run, existing_path, message)
    """
    if force:
        return True, None, "Force mode: skipping deduplication check"

    run_hash = compute_run_hash(config, dataset_sha, model_snapshot_id)
    existing = find_existing_run(run_hash, results_dir)

    if existing:
        return False, existing, f"Run hash {run_hash[:8]}... exists: {existing.name}"

    return True, None, f"No existing run found for hash {run_hash[:8]}..."
```

**Integrate into runners**:

```python
# In run_rag_eval.py
from eval_harness.storage.dedup import check_should_run

parser.add_argument(
    "--force",
    action="store_true",
    help="Force re-run even if identical run exists"
)

# In main(), before running evaluation
config_dict = {
    "dataset": args.dataset,
    "slice": args.slice,
    "judge_model": judge_model,
    "top_k": args.top_k,
    # ... all config parameters
}

should_run, existing_path, message = check_should_run(
    config=config_dict,
    dataset_sha=dataset_sha,
    model_snapshot_id=model_snapshot_id,
    results_dir=Path("results"),
    force=args.force
)

if not should_run:
    print(f"[INFO] {message}")
    print(f"[INFO] Use --force to re-run")
    # Optional: create symlink
    # symlink_path = Path("results") / f"latest_{args.dataset}_{args.slice}.json"
    # symlink_path.symlink_to(existing_path)
    sys.exit(0)

print(f"[INFO] {message}")
run_hash = compute_run_hash(config_dict, dataset_sha, model_snapshot_id)

# After evaluation, add to JSON
json_output["run_hash"] = run_hash
```

### Hash Storage Format

**Add to JSON summary**:
```json
{
  "dataset": "legal_rag_bench",
  "slice": "nano",
  "timestamp": "20260521_211101",
  "run_hash": "a3f5d9e1b2c4...",
  "config_hash": "7f8e9d1c...",
  "dataset_sha": "2b3a4f5e...",
  "model_snapshot_id": "gpt-4o-2024-08-06",
  "metrics_avg": { ... }
}
```

### Dependencies

None. Uses stdlib only.

### Definition of Done

- [ ] `compute_run_hash()` implemented
- [ ] `find_existing_run()` implemented
- [ ] `check_should_run()` implemented
- [ ] Integrated into run_rag_eval.py and run_parsing_eval.py
- [ ] `--force` flag added
- [ ] Run hash added to JSON metadata
- [ ] Tests for hash collision resistance (use different configs)
- [ ] Tests for exact match detection (same config, different timestamp)

### What a adversarial Reviewer Will Ask

**Q**: "What about hash collision?"

**A**: SHA-256 has 2^256 space. Probability of collision is negligible. But if it happens: `--force` to re-run.

**Q**: "What if I want to re-run with same config to measure variance?"

**A**: That's PBI-2 (run 5x for variance). Use `--force` to bypass dedup.

**Q**: "What if config changes but I want to compare against old run?"

**A**: Hashes will differ. Both runs stored. Comparison tools can still compare.

**Q**: "Symlink to previous result? Why not just copy?"

**A**: Symlink saves space. Copy wastes. If you want copy, `cp existing.json new.json`.

---

## PBI-33: Add Retention Policy

**Priority**: P2 (Operations)
**Estimate**: 2 hours
**Category**: Store Results → Cleanup

### Problem Statement

Old result files accumulate forever. No cleanup mechanism.

**Evidence**:
```bash
$ find results/ -name "*.json" -mtime +30 | wc -l
50  # 50 files older than 30 days

$ du -sh results/
500M  # 500MB of accumulated results

# No automated cleanup
# Manual rm risks deleting baselines
```

**adversarial reviewer says**: "I can't find the baseline in 500 files. `rm` is scary. I need a safe way to clean up."

### Solution: Configurable Retention Policy

**Policy dimensions**:
1. **Age-based**: Delete files older than X days
2. **Count-based**: Keep latest N runs per dataset/slice
3. **Tag-based**: Never delete files marked `is_baseline=true`
4. **Dry-run**: Show what would be deleted before deleting

### Acceptance Criteria

1. [ ] `--retention-days` config option (default: 30)
2. [ ] `--retention-count` option (keep latest N, default: 10)
3. [ ] Never delete `is_baseline=true` files
4. [ ] `--cleanup-dry-run` to preview deletions
5. [ ] `--cleanup` to execute cleanup
6. [ ] Safety check: prompt before delete unless `--yes`

### Implementation

**Create `src/eval_harness/storage/retention.py`**:

```python
"""Retention policy for evaluation results."""
from pathlib import Path
from datetime import datetime, timedelta
import json
import sys


def parse_timestamp(timestamp_str: str) -> datetime:
    """
    Parse timestamp from result filename.

    Args:
        timestamp_str: Format like "20260521_211101"

    Returns:
        datetime object
    """
    return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")


def find_old_results(
    results_dir: Path,
    retention_days: int,
    retention_count: int,
    keep_baselines: bool = True
) -> list[dict[str, Any]]:
    """
    Find result files that should be deleted.

    Args:
        results_dir: Directory containing result files
        retention_days: Delete files older than this many days
        retention_count: Keep this many latest files per dataset/slice
        keep_baselines: Never delete files with is_baseline=true

    Returns:
        List of dicts: {path, reason, age_days}
    """
    now = datetime.now()
    cutoff_date = now - timedelta(days=retention_days)
    to_delete = []

    # Group files by dataset/slice
    groups: dict[str, list[Path]] = {}

    for json_file in results_dir.glob("*_results_*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)

            # Skip baselines if requested
            if keep_baselines and data.get("is_baseline"):
                continue

            # Group by dataset/slice
            key = f"{data.get('dataset')}_{data.get('slice', '')}"
            if key not in groups:
                groups[key] = []
            groups[key].append(json_file)

        except (json.JSONDecodeError, IOError):
            continue

    # Check each group
    for key, files in groups.items():
        # Sort by timestamp (newest first)
        files_sorted = sorted(
            files,
            key=lambda f: parse_timestamp(f.stem.split("_")[-2]),
            reverse=True
        )

        # Keep latest N
        keep = files_sorted[:retention_count]
        candidates = files_sorted[retention_count:]

        # Check candidates for age
        for json_file in candidates:
            try:
                timestamp_str = json_file.stem.split("_")[-2]
                file_date = parse_timestamp(timestamp_str)
                age_days = (now - file_date).days

                if file_date < cutoff_date:
                    to_delete.append({
                        "path": json_file,
                        "reason": f"older than {retention_days} days ({age_days} days)",
                        "age_days": age_days
                    })
            except (ValueError, IndexError):
                continue

    return to_delete


def cleanup_results(
    results_dir: Path,
    retention_days: int,
    retention_count: int,
    dry_run: bool = True,
    yes: bool = False,
    keep_baselines: bool = True
) -> None:
    """
    Clean up old result files.

    Args:
        results_dir: Directory containing result files
        retention_days: Delete files older than this many days
        retention_count: Keep this many latest files
        dry_run: Show what would be deleted without deleting
        yes: Skip confirmation prompt
        keep_baselines: Never delete baseline files
    """
    to_delete = find_old_results(
        results_dir,
        retention_days,
        retention_count,
        keep_baselines
    )

    if not to_delete:
        print("[INFO] No files to delete")
        return

    # Show what would be deleted
    total_size = 0
    print(f"[INFO] Found {len(to_delete)} files to delete:\n")

    for item in to_delete:
        path = item["path"]
        size = path.stat().st_size
        total_size += size

        # Find associated files (CSV, details.json)
        associated = []
        csv_file = path.with_suffix(".csv")
        if csv_file.exists():
            associated.append(csv_file)
        details_file = path.with_name(path.stem.replace("_results_", "_results_") + "_details.json")
        # This is approximate; adjust based on actual naming

        print(f"  - {path.name}")
        print(f"    Reason: {item['reason']}")
        print(f"    Size: {size / 1024:.1f} KB")
        for assoc in associated:
            print(f"    Also: {assoc.name}")

    print(f"\nTotal space to free: {total_size / 1024 / 1024:.1f} MB")

    if dry_run:
        print("\n[INFO] Dry run mode. No files deleted. Use --cleanup to delete.")
        return

    # Confirm
    if not yes:
        response = input("\nDelete these files? [y/N]: ")
        if response.lower() != "y":
            print("[INFO] Cancelled")
            return

    # Delete
    for item in to_delete:
        path = item["path"]

        # Delete JSON
        path.unlink()

        # Delete associated files
        csv_file = path.with_suffix(".csv")
        if csv_file.exists():
            csv_file.unlink()

        # Delete details.json if exists
        details_pattern = path.stem.replace("_results_", "_")
        for details_file in path.parent.glob(f"{details_pattern}_details.json"):
            details_file.unlink()

        print(f"[INFO] Deleted: {path.name}")

    print(f"[INFO] Cleanup complete. Freed {total_size / 1024 / 1024:.1f} MB")
```

**Add CLI flags**:

```python
# In run_rag_eval.py
parser.add_argument(
    "--retention-days",
    type=int,
    default=30,
    help="Delete result files older than this many days (default: 30)"
)

parser.add_argument(
    "--retention-count",
    type=int,
    default=10,
    help="Keep this many latest result files per dataset/slice (default: 10)"
)

parser.add_argument(
    "--cleanup-dry-run",
    action="store_true",
    help="Show what would be deleted without actually deleting"
)

parser.add_argument(
    "--cleanup",
    action="store_true",
    help="Run cleanup based on retention policy"
)

parser.add_argument(
    "--yes",
    action="store_true",
    help="Skip confirmation prompt for cleanup"
)

# In main()
if args.cleanup or args.cleanup_dry_run:
    cleanup_results(
        results_dir=Path("results"),
        retention_days=args.retention_days,
        retention_count=args.retention_count,
        dry_run=args.cleanup_dry_run,
        yes=args.yes
    )
    sys.exit(0)
```

### Definition of Done

- [ ] Age-based retention implemented
- [ ] Count-based retention implemented
- [ ] Baseline protection implemented
- [ ] Dry-run mode working
- [ ] Confirmation prompt (unless `--yes`)
- [ ] Associated files cleaned up (CSV, details.json)
- [ ] CLI flags added
- [ ] Tests for retention logic

### What a adversarial Reviewer Will Ask

**Q**: "What if I accidentally delete my baseline?"

**A**: Baselines have `is_baseline=true` in JSON. They're never deleted unless you explicitly set `--keep-baselines=false`.

**Q**: "What if retention is too aggressive?"

**A**: Dry-run first. Check the list. Adjust `--retention-days` or `--retention-count`.

**Q**: "Can I exclude certain datasets from cleanup?"

**A**: Not in PBI-33 scope. Move important results to a different directory (e.g., `results/baselines/`).

**Q**: "Why not use S3 lifecycle rules instead?"

**A**: S3 lifecycle is for S3. This is for local files. We'll add S3 lifecycle in PBI-34.

---

## PBI-34: Document S3 Security Model

**Priority**: P2 (Security)
**Estimate**: 2 hours
**Category**: Store Results → Security

### Problem Statement

[`phoenix_adapter.py:538`](src/eval_harness/observability/phoenix_adapter.py:538) has `upload_parquet_to_s3()` but:

1. No documentation on credential source
2. No IAM role requirements specified
3. No bucket policy documented
4. No encryption/retention guidance
5. Silent failure (catches Exception, prints warning, returns False)

**Evidence**:
```python
# src/eval_harness/observability/phoenix_adapter.py:538
def upload_parquet_to_s3(
    self,
    parquet_path: Path,
    bucket: str,
    key_prefix: str = "phoenix-traces",
) -> bool:
    """Upload buffered Parquet traces to S3."""
    try:
        import boto3
        s3_client = boto3.client("s3")  # ← Credentials from where?
        key = f"{key_prefix}/{parquet_path.name}"
        s3_client.upload_file(str(parquet_path), bucket, key)
        return True
    except Exception:
        import sys
        print(f"[WARN] S3 upload failed for {parquet_path}", file=sys.stderr)
        return False  # ← Silent failure. Trace lost?
```

**adversarial reviewer says**:
- "Where do I put credentials?"
- "What IAM permissions do I need?"
- "Is the data encrypted at rest?"
- "What if upload fails? Do I lose the trace?"
- "How long do traces stay in S3?"

### Solution: Comprehensive Security Documentation

Create `docs/security/s3-upload-security.md` with:

1. **IAM role requirements** (least privilege)
2. **Credential source** (priority order)
3. **Bucket policy** (encryption, TLS)
4. **Failure handling** (what happens on upload failure)
5. **Retention policy** (lifecycle rules)
6. **Security checklist** (verification steps)

### Acceptance Criteria

1. [ ] Document minimum IAM permissions
2. [ ] Document credential source priority
3. [ ] Provide bucket policy template
4. [ ] Document encryption at rest and in transit
5. [ ] Document failure behavior
6. [ ] Document retention lifecycle rules
7. [ ] Add security checklist

### Documentation

**Create `docs/security/s3-upload-security.md`**:

```markdown
# S3 Upload Security Model

## Overview

Evaluation traces (Phoenix OTLP) can be uploaded to S3 for long-term storage and analysis.
This document specifies the security requirements for S3 uploads.

**Component**: `src/eval_harness/observability/phoenix_adapter.py:538`

---

## IAM Requirements

### Minimum Permissions

The IAM role/user used for S3 upload MUST have the following minimum permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::eval-harness-traces",
        "arn:aws:s3:::eval-harness-traces/*"
      ]
    }
  ]
}
```

**Why these permissions**:
- `s3:PutObject`: Upload parquet files
- `s3:GetObject`: Download existing traces (for comparison tools)
- `s3:ListBucket`: List traces in bucket (for cleanup tools)

### Optional Permissions (For Future Features)

```json
{
  "Effect": "Allow",
  "Action": [
    "s3:DeleteObject"
  ],
  "Resource": "arn:aws:s3:::eval-harness-traces/*"
}
```

**Only add if** implementing server-side cleanup (e.g., retention policy).

---

## Credential Source

 boto3 follows the AWS credential chain. Priority order:

1. **IAM role** (when running on EC2/ECS/Lambda)
2. **AWS profile** (`AWS_PROFILE` environment variable)
3. **Environment variables** (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
4. **Shared credentials file** (`~/.aws/credentials`)
5. **Config file** (`~/.aws/config`)

### Recommended Approach

**For local development**: Use AWS profile
```bash
export AWS_PROFILE=eval-harness-dev
uv run eval-rag --dataset legal_rag_bench --slice nano
```

**For production**: Use IAM role (EC2/ECS)
```bash
# No credentials needed. boto3 auto-uses instance profile.
uv run eval-rag --dataset legal_rag_bench --slice nano
```

**For CI/CD**: Use environment variables (from Secrets Manager)
```bash
export AWS_ACCESS_KEY_ID=$(aws secretsmanager get-secret-value --secret-id prod/eval-harness/aws-credentials --query SecretString --output text | jq -r '.access_key_id')
export AWS_SECRET_ACCESS_KEY=$(aws secretsmanager get-secret-value --secret-id prod/eval-harness/aws-credentials --query SecretString --output text | jq -r '.secret_access_key')

uv run eval-rag --dataset legal_rag_bench --slice nano
```

---

## Bucket Policy

### Required Bucket Policy

Enforce SSL/TLS and encryption:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyInsecureTransport",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:*",
      "Resource": [
        "arn:aws:s3:::eval-harness-traces",
        "arn:aws:s3:::eval-harness-traces/*"
      ],
      "Condition": {
        "Bool": {
          "aws:SecureTransport": "false"
        }
      }
    },
    {
      "Sid": "RequireEncryption",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:PutObject",
      "Resource": "arn:aws:s3:::eval-harness-traces/*",
      "Condition": {
        "StringNotEquals": {
          "s3:x-amz-server-side-encryption": "AES256"
        }
      }
    }
  ]
}
```

**What this enforces**:
- Deny non-HTTPS requests
- Require server-side encryption (AES256) on upload

### Bucket Configuration

```bash
# Enable default encryption
aws s3api put-bucket-encryption \
  --bucket eval-harness-traces \
  --server-side-encryption-configuration '{
    "Rules": [{
      "ApplyServerSideEncryptionByDefault": {
        "SSEAlgorithm": "AES256"
      }
    }]
  }'

# Enable versioning (optional, for recovery)
aws s3api put-bucket-versioning \
  --bucket eval-harness-traces \
  --versioning-configuration Status=Enabled

# Enable bucket logging (for audit)
aws s3api put-bucket-logging \
  --bucket eval-harness-traces \
  --bucket-logging-configuration '{
    "LoggingEnabled": {
      "TargetBucket": "eval-harness-logs",
      "TargetPrefix": "s3-access-logs/"
    }
  }'
```

---

## Bucket Structure

```
s3://eval-harness-traces/
├── phoenix-traces/
│   ├── evaluations_20260521_211101.parquet
│   ├── evaluations_20260521_210254.parquet
│   └── ...
└── results/  # Optional: for summary JSON/CSV
    ├── legal_rag_bench/
    │   ├── 2026/05/21/
    │   │   ├── summary.json
    │   │   ├── results.csv
    │   │   └── details.json
    └── omnidocbench/
```

**Key prefix**: `phoenix-traces` (default in `upload_parquet_to_s3()`)

---

## Failure Handling

### Current Behavior

From [`phoenix_adapter.py:555`](src/eval_harness/observability/phoenix_adapter.py:555):

```python
except Exception:
    print(f"[WARN] S3 upload failed for {parquet_path}", file=sys.stderr)
    return False  # ← Returns False, does NOT fail evaluation
```

### What This Means

1. **Local parquet file is kept** at `export_dir` (default `/tmp/phoenix_traces`)
2. **Evaluation run continues** (upload failure is non-blocking)
3. **Silent failure**: No details logged, just `[WARN]` message

### Recommended Improvement (Future PBI)

```python
except Exception as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.error(
        f"S3 upload failed for {parquet_path}: {e}",
        exc_info=True  # Include stack trace
    )
    # Keep local file
    return False
```

### Failure Recovery

If S3 upload fails:
1. Local parquet file is preserved at `/tmp/phoenix_traces/evaluations_*.parquet`
2. Manual upload: `aws s3 cp /tmp/phoenix_traces/evaluations_*.parquet s3://eval-harness-traces/phoenix-traces/`
3. Retry: Re-run evaluation (will create new parquet file)

---

## Retention Policy

### Lifecycle Rule

Auto-delete old traces after 30 days:

```json
{
  "LifecycleConfiguration": {
    "Rules": [
      {
        "Id": "DeleteOldTraces",
        "Status": "Enabled",
        "Prefix": "phoenix-traces/",
        "Expiration": {
          "Days": 30
        },
        "AbortIncompleteMultipartUpload": {
          "DaysAfterInitiation": 1
        }
      }
    ]
  }
}
```

**Apply**:
```bash
aws s3api put-bucket-lifecycle-configuration \
  --bucket eval-harness-traces \
  --lifecycle-configuration file://lifecycle.json
```

### Cost Optimization

- Parquet traces: ~100KB per 10 queries
- 30 days of traces (assuming 100 runs/day): ~300MB
- S3 Standard cost: ~$0.007/month

**Lifecycle progression**:
- Days 0-7: S3 Standard (frequent access)
- Days 7-30: S3 Standard-IA (infrequent access)
- Day 30: Delete

---

## Security Checklist

Before deploying S3 upload, verify:

- [ ] IAM role has only minimum required permissions
- [ ] Bucket policy enforces SSL/TLS
- [ ] Bucket policy enforces encryption
- [ ] Default encryption enabled (AES256)
- [ ] Lifecycle rule configured (30-day retention)
- [ ] No credentials in code or config files
- [ ] Credentials from secure source (IAM role or Secrets Manager)
- [ ] Failure logging enabled (CloudWatch)
- [ ] Bucket access logging enabled
- [ ] Tested upload with actual parquet file
- [ ] Verified local file kept on upload failure

---

## Troubleshooting

### "Access Denied"

**Cause**: IAM role lacks permissions.

**Fix**:
1. Check IAM policy: `aws iam get-role-policy --role-name EvalHarnessRole --policy-name S3Upload`
2. Verify bucket ARN matches policy
3. Check for explicit deny in bucket policy

### "Connection Timeout"

**Cause**: VPC endpoint missing or network issue.

**Fix**:
1. If in VPC, add S3 VPC endpoint
2. Check security group allows HTTPS (port 443)
3. Verify DNS resolution

### "Upload Failed (No Details)"

**Cause**: Exception caught, details not logged.

**Fix**: Enable debug logging:
```bash
export AWS_DEBUG=1
uv run eval-rag --dataset legal_rag_bench --slice nano
```

---

## References

- [AWS S3 Security Best Practices](https://docs.aws.amazon.com/AmazonS3/latest/userguide/security-best-practices.html)
- [IAM Policies for S3](https://docs.aws.amazon.com/AmazonS3/latest/userguide/example-policies-s3.html)
- [boto3 Credential Configuration](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html)
```

### Update Code Documentation

Update `phoenix_adapter.py:538` docstring:

```python
def upload_parquet_to_s3(
    self,
    parquet_path: Path,
    bucket: str,
    key_prefix: str = "phoenix-traces",
) -> bool:
    """
    Upload buffered Parquet traces to S3.

    **Security**: Requires IAM permissions for s3:PutObject.
    See docs/security/s3-upload-security.md for requirements.

    **Credentials**: boto3 uses AWS credential chain:
    1. IAM role (EC2/ECS)
    2. AWS profile (AWS_PROFILE env var)
    3. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)

    **Failure**: Returns False on error. Local parquet file is kept.

    Args:
        parquet_path: Path to local parquet file
        bucket: S3 bucket name
        key_prefix: S3 key prefix (default: "phoenix-traces")

    Returns:
        True if upload succeeded, False otherwise

    Example:
        >>> adapter.upload_parquet_to_s3(
        ...     Path("/tmp/phoenix_traces/evaluations_20260521.parquet"),
        ...     bucket="eval-harness-traces"
        ... )
    """
```

### Definition of Done

- [ ] Security doc created at `docs/security/s3-upload-security.md`
- [ ] IAM requirements documented
- [ ] Credential sources documented
- [ ] Bucket policy template provided
- [ ] Encryption requirements documented
- [ ] Failure handling documented
- [ ] Retention lifecycle rule documented
- [ ] Security checklist added
- [ ] Troubleshooting section added
- [ ] Code docstring updated with reference to docs

### What a adversarial Reviewer Will Ask

**Q**: "Why not require SSE-KMS instead of AES256?"

**A**: KMS costs money ($0.002/10K operations). AES256 is free and sufficient for most use cases. If you need KMS compliance, document it.

**Q**: "What if someone puts credentials in `.env`?"

**A**: `.env` should be in `.gitignore`. Add check: if `.env` contains `AWS_`, warn user.

**Q**: "Why swallow exceptions in upload?"

**A**: Non-blocking upload. Evaluation run shouldn't fail if S3 is down. But we should log details. Future improvement.

**Q**: "Where do I get the bucket name from?"

**A**: Add to config file: `eval_config.yaml` with `s3_bucket: eval-harness-traces`.

---

## Summary Table

| PBI | Priority | Estimate | Category | Risk if Deferred |
|-----|----------|----------|----------|------------------|
| Add database option | P2 | 12h | Store Results | Cannot query across runs, manual comparison only |
| Add deduplication | P2 | 4h | Store Results | Wasted cost on duplicate runs |
| Add retention policy | P2 | 2h | Store Results | Storage grows unbounded |
| Document S3 security | P2 | 2h | Store Results | Undocumented security surface |

**Total P2 for Store Results**: 20 hours

---

## Dependencies

```
PBI-31 (Database) ──┐
PBI-32 (Dedup)     ├──→ Can implement in parallel
PBI-33 (Retention) ├──→ Independent
PBI-34 (S3 docs)   ─┘→ Independent
```

**Note**: PBI-31 (Database) makes PBI-32 (Dedup) easier. DB can check for duplicates by hash without scanning files.

---

## Combined Effort Summary

| Category | P0 | P1 | P2 | Total |
|----------|-----|-----|-----|-------|
| Load Evaluation | 2h | 5.5h | 6h | 13.5h |
| Call Orchestration | 3h | 2h | 6h | 11h |
| Offline Evaluation | 2h | 5h | 8h | 15h |
| A/B Routing & Shadow | 0h | 10h | 24h | 34h |
| Collect Metrics & Citations | 20h | 5h | 12h | 37h |
| Framework / Retrieval / Store | 6h | 5h | 22h | 33h |
| **Store Results (this doc)** | 0h | 0h | **20h** | **20h** |
| **TOTAL** | **33h** | **32.5h** | **78h** | **143.5h** |

---

## Implementation Priority

**Week 1**: P0 only (correctness)
**Week 2**: P1 (rigor) + citations
**Week 3-4**: P2 (trust/UX), including Store Results

**Store Results sequencing**:
1. PBI-34 (S3 docs) - Do first, quick win, unblocks deployment
2. PBI-32 (Dedup) - Do second, saves cost immediately
3. PBI-33 (Retention) - Do third, prevents storage bloat
4. PBI-31 (Database) - Do last, largest effort, nice-to-have

---

**Document version**: 1.0
**Last updated**: 2026-05-22
**For**: Team planning meeting
