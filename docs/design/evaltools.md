# Case Assistant — Evaluation Framework

**Project:** AI Case Assistant (RAG + Document Parsing + Prompt Evaluation)
**Approved tools:** DeepEval, Ragas, Promptfoo, Garak
**Prepared:** May 22, 2026

---

## 1. Overview

This document defines the evaluation framework for the Case Assistant project. It covers:

- The 26 metrics that matter, organized by layer (doc parsing, RAG, input safety, output safety, end-to-end)
- Targets / thresholds for each metric
- Which approved tool owns which metric
- Identified gaps and recommended additions
- A tiered evaluation strategy (pre-deployment, CI/CD, production)

The framework assumes a five-layer architecture:

```
User input → [Input Safety] → [Doc Parsing] → [RAG] → LLM → [Output Safety] → Response
                                                              ↑
                                                  [End-to-end accuracy gate]
```

---

## 2. The 26 Metrics

| # | Layer | Metric | What it measures | Target |
|---|---|---|---|---|
| 1 | Doc parsing | TEDS (Tree Edit Distance Similarity) | Table structure accuracy | ≥ 0.90 |
| 2 | Doc parsing | NID (Normalized Indel Distance) | Text extraction accuracy | ≥ 0.95 |
| 3 | Doc parsing | Field extraction F1 | Correctness of key fields (names, dates, amounts) | ≥ 0.95 |
| 4 | Doc parsing | Layout preservation | Reading order and hierarchy preserved | ≥ 0.90 |
| 5 | RAG | Faithfulness | Generated answer grounded in retrieved context | ≥ 0.85 |
| 6 | RAG | Context Precision | Retrieved chunks are relevant | ≥ 0.80 |
| 7 | RAG | Context Recall | Retrieved chunks cover the answer | ≥ 0.85 |
| 8 | RAG | Answer Relevancy | Response addresses the question asked | ≥ 0.85 |
| 9 | RAG | Answer Correctness | Response matches ground-truth answer | ≥ 0.80 |
| 10 | Input safety | Prompt injection success rate | % of injection attacks that change behavior | ≤ 5% |
| 11 | Input safety | Jailbreak success rate | % of DAN / encoding / role-play attacks succeeding | ≤ 5% |
| 12 | Input safety | PII detection precision | Of inputs flagged as PII, % actually PII | ≥ 0.85 |
| 13 | Input safety | PII detection recall | Of actual PII in inputs, % caught | ≥ 0.95 |
| 14 | Input safety | PII detection F1 | Harmonic mean of precision and recall | ≥ 0.90 |
| 15 | Output safety | PII leakage rate | % of responses containing PII that shouldn't be there | ≤ 1% |
| 16 | Output safety | Toxicity score (avg) | Harmful / offensive content (mean) | ≤ 0.10 |
| 17 | Output safety | Toxicity score (max) | Worst single response | ≤ 0.30 |
| 18 | Output safety | Closed-domain hallucination | Refuses when context doesn't support answer | ≥ 90% correct refusals |
| 19 | Output safety | Guardrail bypass rate | % of bad inputs that get past guardrails | ≤ 5% |
| 20 | End-to-end | Case resolution accuracy | Output matches SME ground truth | ≥ 0.85 |
| 21 | End-to-end | Citation existence rate | Cited sources actually exist in retrieved docs | **100%** |
| 22 | End-to-end | Citation support rate | Cited source actually supports the claim | ≥ 0.95 |
| 23 | End-to-end | False refusal rate | Refuses legitimate questions | ≤ 5% |
| 24 | End-to-end | Harmful compliance rate | Answers things it shouldn't | ≤ 2% |
| 25 | End-to-end | Latency (p95) | 95th percentile response time | ≤ 5s |
| 26 | End-to-end | Cost per case | Total LLM + retrieval cost per case | Track over time |

---

## 3. Tool Coverage Matrix

| # | Metric | DeepEval | Ragas | Promptfoo | Garak | Status |
|---|---|:---:|:---:|:---:|:---:|:---:|
| 1 | TEDS | ❌ | ❌ | ❌ | ❌ | ❌ Gap |
| 2 | NID | ❌ | ❌ | ❌ | ❌ | ❌ Gap |
| 3 | Field extraction F1 | ❌ | ❌ | ❌ | ❌ | ❌ Gap |
| 4 | Layout preservation | ❌ | ❌ | ❌ | ❌ | ❌ Gap |
| 5 | Faithfulness | ✅ | ✅ | ⚠️ | ❌ | ✅ |
| 6 | Context Precision | ⚠️ | ✅ | ❌ | ❌ | ✅ |
| 7 | Context Recall | ⚠️ | ✅ | ❌ | ❌ | ✅ |
| 8 | Answer Relevancy | ✅ | ✅ | ⚠️ | ❌ | ✅ |
| 9 | Answer Correctness | ✅ | ✅ | ⚠️ | ❌ | ✅ |
| 10 | Prompt injection success | ⚠️ (DeepTeam) | ❌ | ✅ | ✅ | ✅ |
| 11 | Jailbreak success | ⚠️ (DeepTeam) | ❌ | ✅ | ✅ | ✅ |
| 12 | PII detection precision | ❌ | ❌ | ⚠️ | ❌ | ⚠️ Partial |
| 13 | PII detection recall | ❌ | ❌ | ⚠️ | ❌ | ⚠️ Partial |
| 14 | PII detection F1 | ❌ | ❌ | ⚠️ | ❌ | ⚠️ Partial |
| 15 | PII leakage rate | ⚠️ | ❌ | ✅ | ⚠️ (`leakreplay`) | ✅ |
| 16 | Toxicity (avg) | ✅ | ⚠️ | ✅ | ✅ (`realtoxicityprompts`) | ✅ |
| 17 | Toxicity (max) | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| 18 | Closed-domain hallucination | ✅ (G-Eval*) | ✅ | ⚠️ | ⚠️ (`snowball`) | ✅ |
| 19 | Guardrail bypass rate | ❌ | ❌ | ✅ | ✅ | ✅ |
| 20 | Case resolution accuracy | ✅ (G-Eval) | ⚠️ | ✅ (llm-rubric) | ❌ | ✅ |
| 21 | Citation existence | ❌ | ❌ | ⚠️ | ❌ | ⚠️ Partial |
| 22 | Citation support | ⚠️ (G-Eval) | ⚠️ | ⚠️ | ❌ | ⚠️ Partial |
| 23 | False refusal rate | ✅ (G-Eval) | ❌ | ✅ | ❌ | ✅ |
| 24 | Harmful compliance rate | ⚠️ (DeepTeam) | ❌ | ✅ | ✅ | ✅ |
| 25 | Latency (p95) | ❌ | ❌ | ⚠️ | ⚠️ | ❌ Gap |
| 26 | Cost per case | ❌ | ❌ | ⚠️ | ⚠️ | ❌ Gap |

**Legend:** ✅ Native first-class · ⚠️ Possible but partial / requires work · ❌ Not supported

\* DeepEval's `HallucinationMetric` has known issues (#1146 non-determinism, #730 semantic mismatch). Use G-Eval with a custom rubric instead.

---

## 4. Coverage Summary

| Status | Count | % of total |
|---|---|---|
| ✅ Fully covered by approved tools | 16 | 62% |
| ⚠️ Partial — needs custom work | 4 | 15% |
| ❌ Not covered — needs additional tool | 6 | 23% |

**Headline:** the four approved tools cover roughly 77% of the framework. The remaining 23% has well-defined solutions and is not a blocker.

---

## 5. Gaps and Recommended Additions

| Gap | Metrics | Why approved tools can't cover | Recommended addition |
|---|---|---|---|
| Doc parsing | 1, 2, 3, 4 | These tools evaluate LLM behavior; doc parsing is upstream of the LLM and uses different metric families (edit distance, structural similarity) | Custom pytest harness using `apted` (TEDS), `Levenshtein` / `rapidfuzz` (NID), `seqeval` (field F1). Reference benchmarks: PubTabNet, DocLayNet evaluation scripts |
| Input PII detection | 12, 13, 14 | None has a native PII classifier; they can check for PII in text but don't detect it the way a dedicated tool does | **Microsoft Presidio** — open-source, dedicated PII detection. Run as preprocessing; evaluate with labeled test set in pytest |
| Citation existence | 21 | Application-specific — requires comparing model output against retrieval log | ~30-line Python checker as a DeepEval custom metric or pytest assertion |
| Production observability | 25, 26 | These are pre-deployment evaluators; they don't monitor live traffic | **Langfuse** (self-hostable), **Arize Phoenix** (self-hostable), or **Braintrust** (SaaS) |

---

## 6. Tool Roles in the Stack

| Tool | Primary role | Secondary role | Don't use it for |
|---|---|---|---|
| **DeepEval** | Pytest CI/CD quality gate; G-Eval custom rubrics; end-to-end case accuracy | Backup for RAG metrics | Doc parsing, production monitoring, native PII detection |
| **Ragas** | RAG metrics — faithfulness, context precision/recall, answer relevancy, answer correctness | Closed-domain hallucination | Anything outside the retrieval-generation loop |
| **Promptfoo** | Red-teaming, prompt/model comparison, application-specific attacks, guardrail bypass | Quality eval via llm-rubric assertions | RAG-specific metrics, doc parsing, native PII detection |
| **Garak** | Model-level security baseline; prompt injection / jailbreak / toxicity / encoding bypass scans; fine-tune safety regression | Compliance / audit evidence | Quality eval, RAG metrics, workflow testing |
| Presidio *(add)* | Input/output PII detection | — | Quality or safety eval beyond PII |
| Langfuse / Phoenix *(add)* | Production observability (latency, cost, metric drift) | Live traffic sampling for offline re-scoring | Pre-deployment eval (use the others) |

---

## 7. Prioritization

Build evaluation in this order. Each tier assumes the prior tier exists.

| Priority | Metrics | Why |
|---|---|---|
| **P0 — Liability** | 21 Citation existence, 15 PII leakage | A single failure becomes a legal liability, not just a UX problem |
| **P1 — Trust** | 5 Faithfulness, 18 Hallucination, 22 Citation support | Trust killers; hard to recover from publicly |
| **P2 — Security baseline** | 10 Prompt injection, 11 Jailbreak, 19 Guardrail bypass | Required for any external-facing deployment |
| **P3 — Bottom line** | 20 Case resolution accuracy | The metric leadership will ask about first |
| **P4 — Diagnostic** | All remaining | Used for debugging the metrics above |

---

## 8. Three-Tier Evaluation Strategy

### Tier 1 — Pre-deployment (run once, before launch)

| Activity | Tool | Output |
|---|---|---|
| Full security scan on base + fine-tuned model | Garak (all probes) | Baseline vulnerability report |
| Application-specific red-team | Promptfoo (`redteam run`) | Application attack surface report |
| Full RAG eval on labeled test set | Ragas | Faithfulness, context precision/recall, answer relevancy baselines |
| Doc parsing eval on labeled test set | Custom pytest | TEDS, NID, field F1 baselines |
| End-to-end case eval with SME ground truth | DeepEval G-Eval | Case resolution accuracy baseline |
| PII detection accuracy on labeled inputs | Presidio + pytest | Precision, recall, F1 |

### Tier 2 — CI/CD (run on every change)

| Activity | Tool | Gate |
|---|---|---|
| RAG metrics on subset (~50 cases) | Ragas via pytest / DeepEval | Block deploy if Faithfulness < 0.85 |
| Prompt injection regression (10–20 attacks) | Garak / Promptfoo | Block deploy if any pass |
| PII leakage regression (20 cases) | Presidio + DeepEval | Block deploy if > 1% leak rate |
| Doc parsing regression | Custom pytest | Block deploy if TEDS < 0.90 |
| Citation existence check | Custom DeepEval metric | Block deploy if < 100% |

### Tier 3 — Production (continuous)

| Activity | Tool | Trigger |
|---|---|---|
| Sample 1–5% of live traffic | Langfuse / Phoenix | Continuous |
| Re-score sampled traffic async | DeepEval / Ragas + Presidio | Daily |
| Alert on metric drift | Langfuse / Phoenix | Alert when Faithfulness < 0.80 or PII leakage > 1% |
| Latency / cost tracking (25, 26) | Langfuse / Phoenix | Continuous |

---

## 9. Final Bill of Materials

| Component | Status | Purpose |
|---|---|---|
| DeepEval | Approved | CI/CD pytest gate, G-Eval custom metrics, end-to-end accuracy |
| Ragas | Approved | RAG metrics (faithfulness, context precision/recall, relevancy) |
| Promptfoo | Approved | Red-teaming, prompt comparison, guardrail bypass testing |
| Garak | Approved | Model security baseline, audit evidence |
| **Microsoft Presidio** | **To add** | PII detection (input and output) |
| **Custom pytest harness** | **To build** | Doc parsing metrics (TEDS, NID, F1, layout) + citation existence checker |
| **Langfuse or Phoenix** | **To add** | Production observability (latency, cost, drift) |

With these additions, the framework gives full coverage of the 26 metrics across the case assistant's five evaluation layers.

---

## 10. Key Risks and Caveats

| Risk | Mitigation |
|---|---|
| DeepEval ships default-on PostHog + Sentry telemetry | Set `DEEPEVAL_TELEMETRY_OPT_OUT=YES` in base image and CI; egress-filter `api.ipify.org`, `*.ingest.sentry.io`, `us.i.posthog.com`; pin DeepEval version and review `telemetry.py` on each upgrade |
| Ragas in active rewrite (v0.2 → v0.3 → v0.4); testset generation module future debated by maintainers (#2231) | Pin version; avoid depending on testset generation for production tests |
| DeepEval HallucinationMetric non-deterministic (#1146) and semantically wrong (#730) | Use G-Eval with explicit rubric instead of HallucinationMetric |
| Promptfoo acquired by OpenAI (March 2026) | Acceptable for now; monitor roadmap for vendor-bias in attack probes |
| Garak false positives from empty outputs (#1114, open) | Read hit log alongside failure rates; don't rely on raw failure percentages |
| Garak `--probes all` is expensive | Scope to `dan,encoding,promptinject,leakreplay,realtoxicityprompts` for routine runs; full sweep only for major releases |
| LLM-as-judge metric cost ($0.05–$0.15 per evaluated trace on GPT-4o-class judges) | Use cheaper judge models for high-volume runs; reserve GPT-4o-class for ground-truth labelling |