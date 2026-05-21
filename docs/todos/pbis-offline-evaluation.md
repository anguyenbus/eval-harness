# Offline Evaluation: PBIs

**Capability**: Offline Evaluation — running on pre-collected datasets without real-time traffic.

**Current State**: Static benchmarks with deterministic + LLM-judge metrics. Missing: model version pinning, reproducibility metadata.

---

## PBI-10: Pin Model Snapshots

**Priority**: P0 (Reproducibility)
**Estimate**: 1 hour
**Category**: Offline Evaluation → Model Versioning

### Problem

Judge model uses `gpt-4o` alias. OpenAI updates aliases without notice. Today's `gpt-4o` != next month's `gpt-4o`. Your scores drift and you don't know why.

From [`deepeval_config.py:29`](src/eval_harness/metrics/deepeval_config.py:29):

```python
DEFAULT_OPENAI_MODEL: Final[str] = "gpt-4o"  # ← Floating alias
```

### Acceptance Criteria

1. [ ] Pin judge model to dated snapshot: `gpt-4o-2024-08-06`
2. [ ] Add fallback logic if snapshot is deprecated
3. [ ] Document model snapshot date in JSON summary
4. [ ] Add `--model-snapshot` CLI flag for override
5. [ ] Verify snapshot is accessible before starting eval

### Implementation Notes

**Update `deepeval_config.py`**:

```python
# Before
DEFAULT_OPENAI_MODEL: Final[str] = "gpt-4o"

# After
DEFAULT_OPENAI_MODEL: Final[str] = "gpt-4o-2024-08-06"
MODEL_ALIASES: Final[dict[str, str]] = {
    "gpt-4o": "gpt-4o-2024-08-06",  # Pin alias to snapshot
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18"
}

def resolve_model(model: str) -> str:
    """Resolve model alias to pinned snapshot."""
    return MODEL_ALIASES.get(model, model)
```

**Pre-flight check**:

```python
def verify_model_accessible(model: str, api_key: str) -> bool:
    """Verify model is accessible before starting eval."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        return True
    except Exception as e:
        raise RuntimeError(
            f"Model {model} is not accessible. Error: {e}\n"
            f"Check https://platform.openai.com/docs/models for available snapshots."
        )
```

**JSON summary update**:

```json
{
  "judge_model": {
    "name": "gpt-4o-2024-08-06",
    "alias": "gpt-4o",
    "snapshot_date": "2024-08-06",
    "api_provider": "openai"
  }
}
```

**Snapshot lifecycle** (document this):

| Model | Snapshot | Deprecation | Replacement |
|-------|----------|-------------|-------------|
| gpt-4o | 2024-08-06 | ~6 months | New snapshot TBD |
| gpt-4o-mini | 2024-07-18 | ~6 months | New snapshot TBD |

### Definition of Done

- [ ] Model pinned to dated snapshot
- [ ] Pre-flight verification implemented
- [ ] JSON includes snapshot metadata
- [ ] `--model-snapshot` flag working
- [ ] Aliases documented with fallback logic

---

## PBI-11: Resolve Config Contradiction

**Priority**: P0 (Correctness)
**Estimate**: 1 hour
**Category**: Offline Evaluation → Configuration

### Problem

Two sources say different things:

1. [`eval_config.yaml:26`](eval_config.yaml:26) says `judge_model: gpt-4o`
2. [`deepeval_config.py:29`](src/eval_harness/metrics/deepeval_config.py:29) defaults to `gpt-4o-mini`

Which one wins? Currently: code default wins, config is ignored. This is confusing and error-prone.

### Acceptance Criteria

1. [ ] Config YAML takes precedence over code defaults
2. [ ] Document precedence order clearly
3. [ ] Add warning when code default is used
4. [ ] Validate model name at config load time
5. [ ] Fail fast on invalid model names

### Implementation Notes

**Precedence order** (highest to lowest):

1. CLI flag `--judge-model`
2. Config YAML `datasets.legal_rag_bench.deepeval.judge_model`
3. Environment variable `DEEPEVAL_JUDGE_MODEL`
4. Code default (`deepeval_config.py`)

**Update `get_deepeval_config()`**:

```python
def get_deepeval_config(config: dict, cli_judge_model: str | None = None):
    # 1. CLI flag
    if cli_judge_model:
        return cli_judge_model

    # 2. YAML config
    yaml_model = config.get("datasets", {}).get(
        "legal_rag_bench", {}
    ).get("deepeval", {}).get("judge_model")

    if yaml_model:
        return yaml_model

    # 3. Environment variable
    env_model = os.environ.get("DEEPEVAL_JUDGE_MODEL")
    if env_model:
        return env_model

    # 4. Code default (with warning)
    print(
        "[WARN] Using code default model. "
        "Set judge_model in eval_config.yaml or use --judge-model flag.",
        file=sys.stderr
    )
    return DEFAULT_OPENAI_MODEL
```

**Validation**:

```python
VALID_MODELS = {
    "openai": [
        "gpt-4o-2024-08-06",
        "gpt-4o-mini-2024-07-18",
        "gpt-4-turbo-2024-04-09"
    ]
}

def validate_model(model: str, provider: str = "openai"):
    valid = VALID_MODELS.get(provider, [])
    resolved = resolve_model(model)

    # Check if it's a known alias
    if resolved not in valid and model not in MODEL_ALIASES:
        raise ValueError(
            f"Unknown model: {model}\n"
            f"Valid models for {provider}: {valid}\n"
            f"Use --judge-model to specify."
        )
```

### Definition of Done

- [ ] Precedence order documented
- [ ] Config takes precedence over code
- [ ] Warning logged when using code default
- [ ] Model validation at startup
- [ ] Tests for precedence logic

---

## PBI-12: Capture Dataset SHA

**Priority**: P1 (Reproducibility)
**Estimate**: 2 hours
**Category**: Offline Evaluation → Metadata

### Problem

HuggingFace datasets change. `isaacus/legal-rag-bench` today != next month. No record of which dataset version produced your results.

### Acceptance Criteria

1. [ ] Capture HuggingFace dataset commit SHA
2. [ ] Record dataset SHA in JSON summary
3. [ ] Add `--dataset-version` flag for manual override
4. [ ] Verify SHA format (40 hex chars)
5. [ ] Document dataset in `results/datasets/` directory

### Implementation Notes

**HuggingFace SHA capture**:

```python
from huggingface_hub import dataset_info

def get_dataset_sha(repo_id: str, token: str | None = None) -> str:
    """
    Get the current commit SHA of a HuggingFace dataset.

    Args:
        repo_id: Dataset repo ID (e.g., "isaacus/legal-rag-bench")
        token: HuggingFace auth token

    Returns:
        Commit SHA (40 hex characters)
    """
    try:
        info = dataset_info(repo_id, token=token)
        sha = info.sha
        if not sha or len(sha) != 40:
            raise ValueError(f"Invalid SHA: {sha}")
        return sha
    except Exception as e:
        print(f"[WARN] Could not fetch dataset SHA: {e}", file=sys.stderr)
        return "unknown"

# In load_legal_rag_bench()
dataset_sha = get_dataset_sha(DATASET_NAME, token)
yield (..., ..., ..., {"dataset_sha": dataset_sha})
```

**JSON summary format**:

```json
{
  "dataset": {
    "name": "isaacus/legal-rag-bench",
    "split": "test",
    "sha": "a1b2c3d4e5f6...40chars",
    "slice": "nano",
    "num_questions": 10
  }
}
```

**Manual override**:

```bash
uv run eval-rag --slice nano --dataset-sha a1b2c3d4e5f6...
```

### Definition of Done

- [ ] Dataset SHA captured for HF datasets
- [ ] SHA recorded in JSON summary
- [ ] Manual override flag working
- [ ] SHA validation implemented
- [ ] Local datasets handled gracefully

---

## PBI-13: Capture Environment Metadata

**Priority**: P1 (Reproducibility)
**Estimate**: 3 hours
**Category**: Offline Evaluation → Metadata

### Problem

Results not reproducible across environments. Python version, OS, dependencies all affect behavior. None recorded.

### Acceptance Criteria

1. [ ] Capture Python version
2. [ ] Capture OS name and version
3. [ ] Capture uv.lock hash or git commit SHA
4. [ ] Capture DeepEval version
5. [ ] Capture CUDA/cuDNN versions (if applicable)
6. [ ] All recorded in JSON summary

### Implementation Notes

**Create `src/eval_harness/observability/environment.py`**:

```python
import sys
import platform
import hashlib
from pathlib import Path
from typing import Dict

def get_environment_metadata() -> Dict[str, str]:
    """Capture all relevant environment metadata."""
    metadata = {}

    # Python
    metadata["python_version"] = sys.version
    metadata["python_implementation"] = sys.implementation.name

    # OS
    metadata["os_system"] = platform.system()
    metadata["os_release"] = platform.release()
    metadata["os_version"] = platform.version()
    metadata["machine"] = platform.machine()

    # Package versions
    metadata["deepeval_version"] = get_package_version("deepeval")
    metadata["torch_version"] = get_package_version("torch")
    metadata["transformers_version"] = get_package_version("transformers")

    # Dependency lock
    metadata["lockfile_hash"] = get_lockfile_hash()

    # CUDA (if available)
    cuda_version = get_cuda_version()
    if cuda_version:
        metadata["cuda_version"] = cuda_version

    return metadata

def get_package_version(package_name: str) -> str:
    """Get package version or 'unknown'."""
    try:
        import importlib.metadata as importlib_metadata
        return importlib_metadata.version(package_name)
    except Exception:
        return "unknown"

def get_lockfile_hash() -> str:
    """
    Get hash of uv.lock file for dependency tracking.

    Returns:
        SHA256 hash of uv.lock or 'unknown' if not found
    """
    lock_path = Path("uv.lock")
    if not lock_path.exists():
        return "unknown"

    content = lock_path.read_bytes()
    return hashlib.sha256(content).hexdigest()[:16]

def get_cuda_version() -> str | None:
    """Get CUDA version if torch is using CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.version.cuda
    except Exception:
        pass
    return None
```

**JSON summary format**:

```json
{
  "environment": {
    "python_version": "3.13.0",
    "os_system": "Linux",
    "os_release": "6.14.0-37-generic",
    "deepeval_version": "2.0.0",
    "torch_version": "2.5.0",
    "lockfile_hash": "a1b2c3d4e5f6",
    "cuda_version": null
  }
}
```

### Definition of Done

- [ ] All metadata categories captured
- [ ] JSON summary includes environment section
- [ ] Graceful handling when info unavailable
- [ ] Tested across different OS

---

## PBI-14: Add Streaming Mode

**Priority**: P2 (Performance)
**Estimate**: 8 hours
**Category**: Offline Evaluation → Streaming

### Problem

Results only available after full completion. For long-running evals (full slice = ~2 hours), can't see early trends.

### Acceptance Criteria

1. [ ] `--stream-interval` flag (e.g., every 10 queries)
2. [ ] Write intermediate JSON summary every N queries
3. [ ] Push to webhook URL if provided
4. [ ] Append to existing file, don't overwrite
5. [ ] Mark final summary clearly vs intermediate

### Implementation Notes

```python
parser.add_argument(
    "--stream-interval",
    type=int,
    default=0,
    help="Write intermediate summary every N queries (0 = disabled)"
)

parser.add_argument(
    "--stream-webhook",
    type=str,
    help="Webhook URL to push intermediate results"
)

# In main loop
for idx, query in enumerate(dataset):
    result = process_query(query)

    if args.stream_interval and (idx + 1) % args.stream_interval == 0:
        intermediate_summary = compute_intermediate_summary(results_so_far)
        intermediate_summary["intermediate"] = True
        intermediate_summary["queries_processed"] = idx + 1

        summary_path = output_path.with_suffix(f".intermediate_{idx + 1}.json")
        with open(summary_path, "w") as f:
            json.dump(intermediate_summary, f, indent=2)

        if args.stream_webhook:
            push_to_webhook(args.stream_webhook, intermediate_summary)
```

### Definition of Done

- [ ] Streaming interval working
- [ ] Intermediate files named clearly
- [ ] Webhook push functional
- [ ] Final summary marked as final
- [ ] Tested with long-running eval

---

## Dependencies

```
PBI-10 (Pin model) ──┐
PBI-11 (Resolve config) ├──→ Enable accurate reproducibility (PBI-12, PBI-13)
                      │
PBI-14 (Streaming) ────┘ (Independent)
```

## Summary Table

| PBI | Priority | Estimate | Dependencies |
|-----|----------|----------|--------------|
| Pin model snapshots | P0 | 1h | None |
| Resolve config contradiction | P0 | 1h | None |
| Capture dataset SHA | P1 | 2h | None |
| Capture environment metadata | P1 | 3h | None |
| Add streaming mode | P2 | 8h | None |

**Total P0-P1**: 7 hours
**Total all**: 15 hours
