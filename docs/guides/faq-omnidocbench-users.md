# FAQ: OmniBenchmark Users → Eval-Harness

## For Teams Using OmniBenchmark JSON Schema

### Q: Do I need to change my existing OmniBenchmark data?

**A:** No. Your OmniBenchmark JSON stays as-is. eval-harness reads it directly via `load_omnidocbench()`.

No migration, no conversion, no re-annotation.

### Q: This adds another schema to maintain. Isn't that more work?

**A:** One-time ~50 line adapter. Then you're done.

- **Before**: Your parser → OmniBenchmark → custom metrics
- **After**: Your parser → OmniBenchmark → adapter (50 lines) → standard metrics

The adapter is written once and reused forever.

### Q: Why not just use OmniBenchmark schema directly?

**A:** OmniBenchmark is specific to that benchmark. eval-harness schema is universal.

Same adapter works for:
- OmniBenchmark
- DP-Bench  
- PubLayNet
- Your proprietary datasets

### Q: What if my parser doesn't output everything eval-harness wants?

**A:** Most fields are optional.

Minimum required:
```python
{
    "elements": [
        {"type": "paragraph", "text": "...", "page_index": 0}
    ]
}
```

You get text-based metrics (NID, BLEU) immediately. Add bbox later for layout metrics.

### Q: Does this tie us to Docling?

**A:** No. eval-harness is parser-agnostic.

- Docling is just one of many parser options
- You can use Tesseract, PaddleOCR, Azure Document Intelligence, etc.
- The schema works with any parser

### Q: Our team has custom metrics. Do we lose them?

**A:** No. eval-harness adds standard metrics; you keep your custom ones.

Run both in parallel if needed:
```python
# Your metrics
my_score = my_custom_metric(prediction, ground_truth)

# Standard metrics (new)
std_scores = eval_harness.evaluate(prediction, ground_truth)
```

### Q: How much engineering time to integrate?

**A:** ~30-40 minutes for most teams.

- 5 min: Read this guide
- 10 min: Copy adapter template
- 15 min: Implement type mapping
- 5 min: Test with single document
- 5 min: Run full evaluation

### Q: What if our schema is different from OmniBenchmark?

**A:** The adapter handles any differences.

OmniBenchmark reference is just an example. The key is mapping your parser's output to eval-harness format, which is straightforward type conversion.

### Q: Do we need to retrain our models?

**A:** No.

This is purely an evaluation format change. Your models, training pipeline, and inference stay exactly the same.

### Q: Can we compare against our previous results?

**A:** Yes. eval-harness outputs CSV with all metrics.

Run your old evaluator and eval-harness in parallel. Compare results to validate consistency.

### Q: What about our existing evaluation pipeline?

**A:** Keep it. eval-harness is additive.

Replace or augment based on your needs:
- **Replace**: Switch to eval-harness for standard metrics
- **Augment**: Add eval-harness metrics alongside your custom ones
- **Phase**: Gradually migrate validation to eval-harness

### Q: Who maintains this schema?

**A:** It's open source. You can extend it too.

The schema is in `contracts/parser_output.schema.json`. If you need new fields, propose them via PR or fork.

### Q: What if eval-harness schema changes?

**A:** Versioned with `schema_version`.

Your adapter continues working until you choose to upgrade. Breaking changes require explicit schema version bump.

### Q: Our data is proprietary. Can we use eval-harness?

**A:** Yes. eval-harness runs locally.

No data leaves your environment. Everything runs on your machines.

### Q: Do we need to share our data?

**A:** No. 

eval-harness is evaluation software, not a benchmark provider. Use your own datasets, keep them private.

### Q: How does this compare to existing OmniBenchmark evaluations?

**A:** Compatible format, more metrics.

OmniBenchmark provides specific metrics for their format. eval-harness:
- Supports OmniBenchmark format
- Adds standard metrics (NID, TEDS, MHS, BLEU, METEOR)
- Works with other benchmarks too
- Provides CSV export for analysis

### Q: Can we extend the metrics?

**A:** Yes. Two ways:

1. **Custom metrics in your code**: Keep running alongside eval-harness
2. **Contribute to eval-harness**: Add new metrics to the framework (PRs welcome)

### Q: What's the ROI?

**A:** Standard metrics for minimal integration cost.

| Benefit | Value |
|---------|-------|
| Standard metrics | Compare to other teams/papers |
| Multiple benchmarks | OmniBenchmark + DP-Bench + more |
| Less maintenance | Community-supported metrics |
| CSV export | Direct to analysis notebooks |
| Baseline comparisons | ChromaDB stub provided |

### Q: What if we need help integrating?

**A:** Examples and documentation provided.

- `examples/omnidocbench_adapter.py` — working adapter
- `docs/guides/schema-alignment-guide.md` — step-by-step
- `contracts/parser_output.schema.json` — full spec

### Q: Can we use this for production evaluation?

**A:** Yes.

eval-harness is designed for production use:
- Incremental CSV writes (monitor long-running evals)
- Error handling and recovery
- Summary statistics (JSON + CSV)
- Version tracking

### Q: How do we get started?

**A:** Three steps:

1. **Install**: `uv sync`
2. **Copy adapter**: `cp examples/omnidocbench_adapter.py my_adapter.py`
3. **Run**: `eval-parsing --dataset omnidocbench --parser my_adapter --limit 10`

Then scale up from there.
