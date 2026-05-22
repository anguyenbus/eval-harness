# Ragas vs DeepEval: Comprehensive Technical Report

**Prepared:** May 22, 2026
**Scope:** Repository health, release cadence, open issues, and technical risks for both LLM evaluation frameworks, with the level of detail needed to defend or attack a tooling choice in an engineering review.

---

## 1. Executive summary

| Dimension | Ragas | DeepEval | Edge |
|---|---|---|---|
| Stars | ~14,000 | ~15,300 | DeepEval |
| Forks | 1,400 | 1,400 | Tie |
| Open issues | ~307 | ~209 | DeepEval (better triage) |
| Open PRs | 99 | 53 | DeepEval (less backlog) |
| Total commits (main) | ~1,147 | High (most recent May 12, 2026) | DeepEval (more recent activity) |
| Latest release | v0.4.3 (Jan 13, 2026) | v3.7.7 (Feb 2026) | DeepEval (faster cadence) |
| License | Apache-2.0 | Apache-2.0 | Tie |
| Maintainer org | explodinggradients → "vibrantlabsai" (recently rebranded) | confident-ai | DeepEval (more stable identity) |
| Primary focus | RAG pipelines | General LLM apps + agents + red teaming | DeepEval (broader) |

**Both projects are mature and actively maintained.** DeepEval has the larger community and faster release cadence; Ragas is the older, more academically cited project (EACL 2024) and remains the de facto default for RAG-specific evaluation. **Neither is risk-free.** DeepEval shipped a serious OpenTelemetry hijack in v3.7.7 that has since been removed from `main`, but the framework still defaults to PostHog + Sentry telemetry that must be explicitly disabled. Ragas carries a history of breaking changes, async-loop patching that silently breaks downstream apps, and instability in its testset generation module.

> **Verification note (May 22, 2026):** The OpenTelemetry TracerProvider hijack originally described in DeepEval issue `#2497` (v3.7.7) is **no longer present in the current `main` branch.** This report has been updated to reflect the verified current state of `deepeval/telemetry.py`. See §3.2.1 for the full breakdown and mitigation steps.

---

## 2. Repository activity & velocity

### 2.1 Ragas

- Commits to `main`: ~1,147 total.
- 89 releases since project inception; v0.4.x line introduced in late 2025 as "the most significant change since v0.2," with breaking changes acknowledged in their own migration guide.
- Recent PRs include async fixes (`#2369` detect uvloop and skip nest_asyncio), metric migrations to the new collections API (`#2398`, `#2401`, `#2365`, `#2366`), removal of error suppressors so failures actually surface (`#2362`), and documentation calling out that "Context Relevance implementation differs from the paper" (`#2378`) — an unusually honest admission for a published framework.
- Recently added `aspect_critic` for coherence/harmfulness/maliciousness/correctness (`#2375`), suggesting they're moving toward DeepEval's territory of general-purpose LLM evaluation rather than RAG-only.
- Repository was recently renamed from `explodinggradients/ragas` to `vibrantlabsai/ragas`. Old URLs still redirect, but it indicates a corporate restructure (the company behind it appears to have rebranded). For long-term tooling commitments, this is worth noting.

### 2.2 DeepEval

- Commit activity through May 12, 2026 — actively being worked on day-by-day.
- Releases roughly monthly, with v3.0 (mid-2025) introducing component-level evaluation, simulation, and observability; v3.7.x in early 2026 added security/telemetry features (which then created the issues described in §3.2.1).
- 50+ pre-built metrics out of the box, including benchmark harnesses (MMLU, HellaSwag, DROP, BIG-Bench Hard, TruthfulQA, HumanEval, GSM8K). Red teaming was spun out into a sibling project (DeepTeam) — a sign of healthy modularization rather than a project losing scope.
- Tightly coupled to the Confident AI commercial platform. The open-source library is the funnel; this commercial pressure is what funds the cadence but also drives the telemetry decisions discussed below.

**Velocity verdict:** DeepEval ships faster and triages issues more aggressively (209 open vs 307). Ragas ships less often but each release is bigger, which means upgrades hurt more.

---

## 3. Critical open issues — detailed analysis

### 3.1 Ragas — the bug pile

#### 3.1.1 `nest_asyncio.apply()` called at import time
- **Issues:** `#1819` (originally filed Jan 2025), `#1064`, related to `#2351`, partial fix landing via `#2369`.
- **What it does:** Ragas patches Python's event loop **globally as a side effect of `import ragas`**. The line `nest_asyncio.apply()` runs in `ragas/executor.py` at module load.
- **Why it's bad:**
  - Any host application using `uvloop`, `asyncio.run`, or its own event-loop management can break or behave unpredictably.
  - In already-async contexts (Jupyter notebooks, FastAPI handlers, anyio-based services), users hit `RuntimeError: This event loop is already running` and `coroutine '...' was never awaited` warnings.
  - The standard Python community position is that **libraries should not patch the global event loop** — that's an application-level decision.
- **Fix status:** A detection-and-skip mechanism for uvloop has been merged (`#2369`). Whether it fully resolves the issue depends on how complete the detection is; the cleaner fix would be removing the import-time patching entirely.

#### 3.1.2 Pervasive NaN scores from JSON parse failures
- **Issues:** `#580`, `#1403`, `#1444`, plus a documented note in Ragas's own `docs/index.md`.
- **What it does:** Many Ragas metrics work by sending prompts to a judge LLM and parsing the response as JSON via Pydantic. If the model returns malformed JSON or no statements at all (e.g., when scoring "I don't know" answers), the score silently becomes NaN.
- **Why it's bad:**
  - Small/local models (Llama-class, Bedrock-hosted models, anything not GPT-4-class) hit this constantly.
  - The Tweag engineering team (independent third-party analysis, published Feb 2025) measured Ragas faithfulness scores varying from **0% under Llama 3 to over 80% under Claude 3 Sonnet on the same data** — confirming that judge-model choice dominates the metric.
  - Even Ragas's own quickstart example reportedly produces NaN faithfulness scores for some users.
- **Fix status:** Acknowledged but inherent to the LLM-as-a-judge architecture. The new collections API may stabilize parsing, but the fundamental sensitivity to judge model remains.

#### 3.1.3 Testset Generation module — future officially uncertain
- **Issue:** `#2231` opened by maintainer `jjmachan` in Aug 2025.
- **What it is:** The maintainers themselves opened a poll asking whether to **keep, spin out, or deprecate** the testset generation module — one of Ragas's headline features.
- **Why it matters:** If your team relies on `TestsetGenerator` for synthetic data, you're depending on a feature whose own maintainers haven't decided to keep. Plus existing testset-generation issues like `#1546` (`ValueError: Node has no summary_embedding` when following the documented flow) remain unresolved.

#### 3.1.4 Cross-version upgrade pain
- **Issues:** `#2351` (0.2.15 → 0.3.6 broke with `LM returned 1 generations instead of requested 3`), v0.2 → v0.3 → v0.4 each introduced breaking changes.
- **Why it matters:** Ragas's own migration docs describe v0.4 as the biggest change since v0.2. Three major architectural rewrites in under two years is a stability signal. A team running Ragas in CI needs a pinned version and a deliberate upgrade plan.

#### 3.1.5 Score interpretability and inconsistency
- **Issue:** `#452` ("Ragas scores look inconsistent") and arxiv:2407.12873 (academic paper specifically critiquing Ragas metrics in the telecom domain).
- **What the paper found:** Two of Ragas's headline metrics (Answer Correctness and Answer Relevancy) were judged unsuitable as standalone reliability metrics by domain experts. Only Faithfulness and Factual Correctness held up under SME review. The paper notes Ragas is "a black box; hence, interpretability of the scores is difficult as the scores conflicted with human scores by SMEs."
- **Why it matters:** This isn't just a bug — it's peer-reviewed evidence that some of the marketed metrics may not measure what they claim to measure.

### 3.2 DeepEval — the bug pile

#### 3.2.1 OpenTelemetry hijack (Issue #2497) — verified as fixed on `main`, with caveats

**Original claim (v3.7.7, Feb 2026):** Issue `#2497` reported that DeepEval v3.7.7's `telemetry.py` at module import time:
1. Hijacked the global OpenTelemetry `TracerProvider`, routing all host application spans to DeepEval's New Relic account
2. Initialized Sentry with 100% CPU profiling and trace sampling
3. Clobbered `sys.excepthook` so uncaught exceptions went to DeepEval's Sentry
4. Made a blocking HTTP call to `api.ipify.org`
5. Initialized PostHog analytics

**Verified current state of `deepeval/telemetry.py` on `main` (read May 22, 2026, 518 lines):**

| Behavior | v3.7.7 | Current `main` |
|---|---|---|
| Global `trace.set_tracer_provider(TracerProvider())` | ✅ Yes | ❌ **Removed entirely** |
| OTLP exporter to `otlp.nr-data.net` | ✅ Yes | ❌ **Removed entirely** |
| Hardcoded New Relic API key | ✅ Yes | ❌ **Removed entirely** |
| Sentry `profiles_sample_rate` | `1.0` (100%) | `0.0` |
| Sentry `traces_sample_rate` | `1.0` (100%) | `0.0` |
| `sys.excepthook` behavior | Clobbered | **Chained** — previous excepthook is preserved and called after Sentry capture |
| `api.ipify.org` fetch | Blocking on import | **Async** — runs on daemon thread |
| Sentry initialization | At import, unconditional | At import, gated by `telemetry_opt_out()` |
| PostHog initialization | At import | At import, gated by `telemetry_opt_out()` |
| Excepthook installation | At import, unconditional | Triple-gated: `ERROR_REPORTING` + `not blocked_by_firewall()` + `not telemetry_opt_out()` |

The TracerProvider hijack is **gone from the codebase**. So is the New Relic exfiltration path. The Sentry SDK is still initialized when telemetry is on, but with both sample rates at 0.0 it does not actively profile or trace the host application — it only captures exceptions via the (now-chained) excepthook.

**What still happens when telemetry is on (the default):**

- **PostHog client is constructed** at import time with hardcoded project key `phc_IXvGRcscJJoIb049PtjIZ65JnXQguOUZ5B5MncunFdB` → `us.i.posthog.com`.
- **Sentry SDK is initialized** (DSN hardcoded) but with sampling at 0%. Uncaught exceptions still get reported via `sentry_sdk.capture_exception()` from the chained excepthook.
- **Public IP fetched** from `api.ipify.org` on a background daemon thread.
- **Every PostHog event includes**: deepeval version, login status, Jupyter/non-Jupyter flag, anonymous UUID, public IP, installed integrations, per-feature usage. Events fire on evaluations, synthesizer runs, guardrail runs, benchmarks, dataset pulls, traces sent, logins.

**How to disable all third-party telemetry — verified:**

```bash
export DEEPEVAL_TELEMETRY_OPT_OUT=YES
```

I traced the opt-out path through every telemetry-emitting code path in `telemetry.py`. The pattern is consistent:

```python
def telemetry_opt_out():
    return get_settings().DEEPEVAL_TELEMETRY_OPT_OUT

# Module-level guards
if not telemetry_opt_out():
    sentry_sdk.init(...)        # only initialized if opt-in
    posthog = Posthog(...)      # only constructed if opt-in

# Every context manager
@contextmanager
def capture_evaluation_run(type):
    if telemetry_opt_out():
        yield                   # no telemetry, just yield through
    else:
        # ... posthog.capture(...)
        yield
```

The excepthook installation is additionally gated on `ERROR_REPORTING` and `blocked_by_firewall()`. With opt-out enabled, neither Sentry nor PostHog is initialized at all — there's no object to capture from. The previous opt-out regression (issue `#1613`, where setting the env var caused `NameError: name 'posthog' is not defined`) was fixed by PR `#1976` (merged Aug 2025) and PR `#1596` (merged May 2025).

**Recommended hardening (defense in depth):**

1. **Set `DEEPEVAL_TELEMETRY_OPT_OUT=YES` in your container base image, CI, and dev environments.** Don't rely on per-developer config.
2. **Pin the DeepEval version.** The v3.7.7 episode shows the maintainer is willing to ship aggressive telemetry changes. Pin to a known-good version and review the `telemetry.py` diff on every upgrade.
3. **Egress-filter the destinations** at the network layer: `api.ipify.org`, `*.ingest.sentry.io`, `us.i.posthog.com`, `otlp.nr-data.net`, `*.nr-data.net`. Even with opt-out set, network-layer blocking is a useful redundant control.
4. **Audit `deepeval/telemetry.py` on every version bump.** The file is ~500 lines and a 5-minute read.
5. **Isolate from sensitive workloads.** Run DeepEval in a separate process or container from any service handling PII or regulated data.

**Severity verdict:** The original v3.7.7 issue was genuinely critical. The current `main` is in a defensible state if telemetry is opted out and version-pinned. The remaining default-on PostHog/Sentry behavior is in line with industry norms for open-source libraries backed by commercial platforms (LangChain, LlamaIndex, etc. do similar). It is **not** in line with what a security-conscious enterprise wants out of the box, which is why opt-out + egress filtering + pinning are non-negotiable.

#### 3.2.2 HallucinationMetric: non-deterministic and semantically wrong
- **Issue #1146** (open, Nov 2024): Same inputs → score of 0.0 in one run, 1.0 in the next. Author reports the metric also only assigns 0.0 or 1.0 rather than a continuous range. Maintainer asked for a reproducer in 2024 and the thread went quiet.
- **Issue #730** (open, April 2024): The metric prompt tells GPT-4 to "FORGIVE cases where the actual output is lacking in detail" but GPT-4 ignores that instruction. So DeepEval's HallucinationMetric actually penalizes **missing facts** (treating them as contradictions), meaning it's effectively a summarization-completeness metric, not a hallucination metric.
- **Why it matters:** A metric named `HallucinationMetric` that doesn't reliably measure hallucination and produces different scores on repeated runs is, to put it mildly, a fitness-for-purpose problem. If your team is gating CI on hallucination thresholds, this matters a lot.

#### 3.2.3 G-Eval doesn't work cleanly with newer OpenAI models
- **Issue #2280** (open, Nov 2025, GPT-5 family): Intermittent `ValueError: Evaluation LLM outputted an invalid JSON`, `AttributeError: log_probs unsupported`, and `json.decoder.JSONDecodeError: Invalid \escape` when using `gpt-5-mini` as judge. Same code works with gpt-5 and gpt-5-nano.
- **Issue #1358** (Feb 2025, o3-mini): G-Eval hardcodes `temperature=0.7` and `logprobs=True`. o3-mini requires `temperature=1.0` and does not support `logprobs`, so any G-Eval custom metric immediately fails against o3-mini.
- **Why it matters:** G-Eval is one of DeepEval's flagship metrics. The model-family-specific brittleness suggests the abstraction over judge LLMs leaks.

#### 3.2.4 HallucinationMetric README example just doesn't work
- **Issue #1750** (July 2025, open): A user copied the example directly from the README and got a runtime failure.
- **Why it matters:** Smaller bug, but symptomatic — examples that don't work are usually a sign the README has drifted behind the code.

#### 3.2.5 ToolCorrectnessMetric server errors
- **Issue #1647** (May 2025): Closed, but worth noting — agent evaluation via ToolCorrectnessMetric hit internal server errors. The closure suggests it's been addressed.

---

## 4. Roadmap & fixes in flight

### 4.1 Ragas — what's coming
From the v0.4.x release notes and merged-but-unreleased PRs:
- `#2369` — detect uvloop and skip nest_asyncio (partial fix for the global-loop-patching issue)
- `#2362` — remove error suppressors so genuine errors surface instead of producing silent NaN
- `#2365`, `#2366`, `#2398`, `#2401`, `#2410` — migrate metrics (answer_correctness, context_entity_recall, context_precision, factual_correctness) and remove redundancy (AnswerSimilarity → SemanticSimilarity)
- `#2375` — new `aspect_critic` metric covering coherence, harmfulness, maliciousness, correctness
- `#2378` — documentation explicitly noting the Context Relevance implementation differs from the original paper design
- `#2381` — sidebar fixes plus documenting `ContextUtilization` as deprecated
- v1.0 roadmap teased in `#2231`

### 4.2 DeepEval — what's coming
- `#2497` (OpenTelemetry hijack) — the hijack code has been **removed from `main`** since the issue was filed against v3.7.7. The current telemetry layer is opt-out-respecting and uses PostHog + a zero-sampling Sentry SDK. See §3.2.1 for verification. The next release should formally close the loop on this; expect it in the next minor version.
- `#2223` — multi-turn extension for Task Completion and Tool Correctness metrics, currently marked `awaiting release`.
- `#2407`, `#2401`, `#2396`, `#2385`, `#2352`, `#2351` — recent issues opened Jan 2026, in various stages of investigation.
- DeepTeam (red-teaming sibling project) actively developed — DeepEval is becoming a platform, not just a library.

---

## 5. Independent / academic critique

Both frameworks have been studied independently and the findings are useful evidence:

- **Tweag analysis (Feb 2025):** Ragas faithfulness scores on the same data ranged from 0% (Llama 3 as judge) to over 80% (Claude 3 Sonnet as judge). The conclusion: judge-model selection dominates the metric output, and JSON-parse failures from smaller models silently produce missing or strange scores.
- **arXiv 2407.12873 (telecom-domain RAG study):** Among Ragas metrics, only Faithfulness and Factual Correctness aligned with subject-matter-expert judgments. Answer Correctness and Answer Relevancy did not.
- **Patronus AI commentary:** Notes that Ragas occasionally fails to extract statements from RAG responses, producing wrong computations. They recommend Lynx for hallucination detection over Ragas in long-context cases.
- **HaluBench benchmark:** Ragas Faithfulness underperformed Lynx on long-context hallucination detection.

For DeepEval, the independent critique is more sparse (it's younger as a project), but issue `#1853` — Aikido security scanner flagging DeepEval as malware due to its telemetry behavior — was an external signal that the telemetry approach was outside normal library conventions. The current `main` removes the most egregious behaviors flagged in that scan; the default-on PostHog/Sentry posture remains.

---

## 6. Decision framework

### When Ragas is the right choice
- The use case is specifically RAG (retrieval + generation), not general LLM apps or agents.
- The team uses GPT-4-class judge models and is okay with the cost.
- Synthetic test data generation is not a core dependency (given the uncertain future of that module).
- The team can pin to a specific version and absorb breaking changes on its own schedule.
- Academic/published methodology is important (the EACL 2024 paper is a credibility asset).

### When DeepEval is the right choice
- The use case spans general LLM apps, agents, and conversational systems — not just RAG.
- The team values release velocity and broader metric coverage.
- The team can set `DEEPEVAL_TELEMETRY_OPT_OUT=YES` in container base images and egress-filter the telemetry destinations (`api.ipify.org`, `*.ingest.sentry.io`, `us.i.posthog.com`).
- The team is comfortable pinning versions and reviewing `deepeval/telemetry.py` diffs on each upgrade.
- CI integration in a Pytest-style pattern is a hard requirement.
- Red-teaming/safety evaluation matters (DeepTeam sibling project).

### When neither is right
- Strict compliance environment (SOC 2, HIPAA, PCI) where you can't deploy network-level egress filtering AND can't trust env-var-based opt-out as a sole control → consider alternatives like Patronus, Arize Phoenix, Langfuse evals, or hand-rolled evaluation against your own ground truth.
- Long-context hallucination is the primary concern → consider Lynx, which the HaluBench results suggest outperforms Ragas Faithfulness.
- Determinism and reproducibility are paramount → consider deterministic statistical metrics (BLEU, ROUGE, exact match, semantic similarity via fixed embeddings) plus a fixed-seed judge LLM, rather than either framework's LLM-as-a-judge defaults.

---

## 7. Talking points for the upcoming review

1. "DeepEval v3.7.7 shipped a serious OpenTelemetry hijack (`#2497`) that exfiltrated host application traces to a third party. That code has been removed from `main`, but the framework still defaults to PostHog + Sentry telemetry. Our plan: pin to a post-fix version, set `DEEPEVAL_TELEMETRY_OPT_OUT=YES` in CI and base images, egress-filter the destinations, and re-audit `telemetry.py` on each upgrade. Is the team aligned on enforcing all four controls?"
2. "Ragas is in its third major architectural rewrite in under two years (v0.2 → v0.3 → v0.4), and its testset-generation module's future is being publicly debated by the maintainers (`#2231`). What does our upgrade and pinning strategy look like?"
3. "Independent academic work (arxiv:2407.12873) found that two of Ragas's headline metrics didn't align with subject-matter-expert judgment. Are we using Answer Correctness or Answer Relevancy as gates, and if so, how do we know the scores reflect reality?"
4. "DeepEval's HallucinationMetric has open issues showing it's non-deterministic on the same input (`#1146`) and arguably measures summarization completeness rather than hallucination (`#730`). What's our validation strategy before we trust it in CI?"
5. "Both frameworks are LLM-as-a-judge architectures. Tweag measured Ragas faithfulness scores ranging from 0% to over 80% on identical data depending on which judge model was used. Have we standardized on a judge model, and what's the cost implication of that lock-in?"

---

## 8. Sources

All claims in this report are sourced from public GitHub issues, release notes, official documentation, and third-party engineering or academic publications:

- GitHub: `vibrantlabsai/ragas` (formerly `explodinggradients/ragas`) — repository stats, releases, issues #452, #580, #1064, #1212, #1403, #1444, #1546, #1819, #2231, #2351, #2362, #2365, #2366, #2369, #2375, #2378, #2381, #2398, #2401, #2402, #2407, #2410
- GitHub: `confident-ai/deepeval` — repository stats, releases, issues #730, #757, #1146, #1340, #1358, #1613, #1647, #1750, #1853, #2092, #2223, #2280, #2351, #2352, #2385, #2396, #2401, #2407, #2497; PRs #1596, #1976
- **Primary source verification:** `deepeval/telemetry.py` on `main` branch, read directly May 22, 2026 (518 lines)
- Ragas v0.3 → v0.4 official migration guide (docs.ragas.io)
- DeepEval v3.0 release notes
- arXiv:2309.15217 (the original Ragas paper, EACL 2024)
- arXiv:2407.12873 (independent telecom-domain Ragas evaluation)
- Tweag engineering blog: "Evaluating the evaluators: know your RAG metrics" (Feb 2025)
- Patronus AI: "RAG Evaluation Metrics" commentary
- HaluBench benchmark results (Lynx vs Ragas Faithfulness)