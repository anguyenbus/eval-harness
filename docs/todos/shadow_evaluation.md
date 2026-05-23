# Shadow Evaluation — Team Discussion

**Purpose:** A shared starting point for the team's design discussion. This is not a proposal. It is a structured way to ask the right questions in the right order, so the team's decision is deliberate.

**Tone:** Adversarial in standards (we hold ideas to a high bar), gentle in language (we are colleagues working out a hard problem). We are seeking truths, not problems.

**Reading time:** ~20 minutes. The diagrams help.

---

## Part 1 — Why this conversation is worth having

The case assistant is a legal product. Its job is to ingest case documents, answer chat questions about them, and (eventually) determine whether a case's documents answer the standard RFI questions for that case type. A regression in any of those capabilities can cause real harm to real cases.

The eval-harness exists to catch regressions before they reach production. Today it can run public benchmarks (OmniDocBench, DP-Bench, LegalBench-RAG) and produce per-run metrics. What it cannot do:

- Evaluate against real production traffic patterns (legal documents from actual cases, real questions lawyers ask).
- Detect that "candidate parser B is better than production parser A" with statistical confidence.
- Block a PR from merging if it regresses eval quality.
- Tell us whether a change that helps chat hurts ingestion (or vice versa).

Shadow evaluation is one way to close that gap. There are others. The team's job in this discussion is to decide whether shadow evaluation is the right tool, what shape it should take, and what we're willing to commit to building.

Worth saying explicitly: there are good reasons not to do this. It is real work, has real cost, and the gap it closes might be smaller than the gap of "we don't have lawyer-confirmed RFI data yet." This document tries to surface those trade-offs rather than hide them.

---

## Part 2 — What "shadow evaluation" actually means

The term gets used loosely. Before we discuss what to build, let's agree on what we're talking about.

### 2.1 The strict definition

In service operations, shadow evaluation means: **a copy of real production traffic is sent to a candidate system, the candidate processes it in parallel with production, the candidate's output is discarded (not shown to users), and the two outputs are compared offline**.

The key properties:

1. **Real production inputs.** Not synthetic, not curated.
2. **Discarded outputs.** Users never see the candidate's response.
3. **Side-by-side comparison.** Production and candidate are compared on the same input.
4. **No write effects.** The candidate doesn't write to production stores, doesn't send notifications, doesn't increment usage counters.

### 2.2 What people often mean instead

In practice, teams use "shadow" to mean any of these:

- **Live mirroring** — the strict definition above.
- **Replay** — capture production traffic to durable storage, then re-run it against candidates later. Same epistemic move, different timing.
- **A/B testing** — split traffic between production and candidate, users see both. This is *not* shadow because users see candidate outputs.
- **Offline variant comparison** — run a fixed dataset through two configurations. This is what the current PBIs called "offline shadow." It is not shadow either; it is regression testing.

For this discussion, "shadow evaluation" means **live mirroring or replay**. Both have the same epistemic properties; they differ on timing and infrastructure.

### 2.3 Which one our system actually needs

The system has three layers where a change might happen — parsing, retrieval, generation — and two modes that consume those layers — RFI sufficiency check and chat.

```svg
<svg viewBox="0 0 720 380" xmlns="http://www.w3.org/2000/svg" role="img" font-family="system-ui, -apple-system, sans-serif">
  <title>System layers and modes</title>
  <defs>
    <marker id="ar1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M2 1L8 5L2 9" fill="none" stroke="#555" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker>
  </defs>
  <rect x="40" y="30" width="640" height="100" rx="10" fill="#f3e8ff" stroke="#9333ea" stroke-width="1"/>
  <text x="56" y="56" font-size="14" font-weight="600" fill="#581c87">Shared ingestion (one pipeline)</text>
  <rect x="60" y="70" width="180" height="44" rx="6" fill="#fff" stroke="#9333ea"/>
  <text x="150" y="97" font-size="13" text-anchor="middle" fill="#333">Parse</text>
  <rect x="270" y="70" width="180" height="44" rx="6" fill="#fff" stroke="#9333ea"/>
  <text x="360" y="97" font-size="13" text-anchor="middle" fill="#333">Chunk + embed</text>
  <rect x="480" y="70" width="180" height="44" rx="6" fill="#fff" stroke="#9333ea"/>
  <text x="570" y="97" font-size="13" text-anchor="middle" fill="#333">Index (per-case)</text>
  <line x1="240" y1="92" x2="266" y2="92" stroke="#555" stroke-width="1.5" marker-end="url(#ar1)"/>
  <line x1="450" y1="92" x2="476" y2="92" stroke="#555" stroke-width="1.5" marker-end="url(#ar1)"/>

  <rect x="40" y="170" width="310" height="180" rx="10" fill="#ccfbf1" stroke="#0d9488"/>
  <text x="56" y="196" font-size="14" font-weight="600" fill="#134e4a">Mode A — RFI sufficiency</text>
  <text x="56" y="220" font-size="12" fill="#134e4a">Batch, deterministic, per case</text>
  <text x="56" y="246" font-size="12" fill="#134e4a">10 RFIs → answered / not answered</text>
  <text x="56" y="272" font-size="12" fill="#134e4a">Ground truth exists (historically)</text>
  <text x="56" y="298" font-size="12" fill="#134e4a">Not user-visible (backend signal)</text>
  <text x="56" y="324" font-size="12" font-style="italic" fill="#134e4a">Today: blocked, no labeled data yet</text>

  <rect x="370" y="170" width="310" height="180" rx="10" fill="#fee2e2" stroke="#dc2626"/>
  <text x="386" y="196" font-size="14" font-weight="600" fill="#7f1d1d">Mode B — Chat</text>
  <text x="386" y="220" font-size="12" fill="#7f1d1d">Interactive, ad hoc, per query</text>
  <text x="386" y="246" font-size="12" fill="#7f1d1d">Open-ended Q&amp;A with citations</text>
  <text x="386" y="272" font-size="12" fill="#7f1d1d">No ground truth (novel queries)</text>
  <text x="386" y="298" font-size="12" fill="#7f1d1d">User-visible response</text>
  <text x="386" y="324" font-size="12" font-style="italic" fill="#7f1d1d">Today: active workload</text>

  <line x1="195" y1="134" x2="195" y2="166" stroke="#555" stroke-width="1.5" stroke-dasharray="3 3" marker-end="url(#ar1)"/>
  <line x1="525" y1="134" x2="525" y2="166" stroke="#555" stroke-width="1.5" stroke-dasharray="3 3" marker-end="url(#ar1)"/>
</svg>
```

A change to parsing can affect both modes. A change to the chat prompt only affects chat. A change to the RFI prompt only affects RFI. The shadow eval design has to respect this.

---

## Part 3 — What we have and what we don't

Honest inventory.

### 3.1 What exists today

- **Eval-harness on EKS.** Containerized, deployed, runs `eval-parsing` and `eval-rag` as batch Jobs.
- **Public benchmarks integrated.** OmniDocBench, DP-Bench, LegalBench-RAG datasets in `data/`, loaders in `datasets/`.
- **Adapter pattern.** `ParserAdapter` and `RagAdapter` are the boundary between eval-harness and any system under evaluation.
- **Metrics layer.** Parsing metrics (NID, TEDS, MHS, BLEU, METEOR) and RAGAS-based RAG metrics.
- **JSON Schema contracts.** A `contracts/` directory at the repo root already exists for cross-system contracts.
- **Arize Phoenix tracing.** Production emits OpenTelemetry traces.
- **Bedrock Claude as judge.** No OpenAI dependency.
- **Working RAG stub.** A ChromaDB-based RAG pipeline in `src/eval_harness/stubs/rag/`.

### 3.2 What does not exist

- **RFI evaluation corpus.** No labeled set of historical cases with lawyer-confirmed sufficiency verdicts.
- **Replay-complete payload capture.** Phoenix traces are observability metadata, not replay-ready payloads. Document references, retrieved chunks, generated answers, judge verdicts are not durably captured at the adapter boundary in a form replay code can re-execute.
- **Paired statistical comparison.** Metrics layer produces per-run summaries, not paired comparisons between two runs with significance tests.
- **CI gates.** Nothing currently blocks a PR that regresses eval quality from merging.
- **Live shadow infrastructure.** No Istio VirtualService mirror configuration, no shadow namespace.

### 3.3 The architectural ambiguity we should resolve

Looking at the repo layout, `src/eval_harness/stubs/rag/` contains a working ChromaDB RAG pipeline — chunker, embedder, generator, ingestion, query. It is labeled "REFERENCE IMPLEMENTATIONS (demo only)."

The team should be explicit about which of these is true:

- **(a)** The stub is purely a demo. Real production case-assistant is a separate codebase, the stub exists only to let people running eval-harness without their own RAG system have something to evaluate against.
- **(b)** The stub is now also the production code, because nobody wrote a replacement.
- **(c)** Production has a separate codebase but the stub is structured similarly.

This affects everything downstream. If we're in (b), then "production case-assistant" and "eval-harness" are the same code, which changes how shadow eval is built. We need to know which world we live in before deciding what to build.

---

## Part 4 — The decisions the team needs to make

Listed in priority order. Each decision is framed as a question, with the options, the trade-offs, and what the answer affects.

### Decision 1 — Are we doing shadow evaluation at all?

This is the first question, and it should be on the table.

The arguments for:

- **Real production distribution.** Public benchmarks are not legal cases. Our cases have document mixes (scanned PDFs, redactions, tables, handwritten exhibits) that benchmarks don't reflect.
- **Catches regressions before users see them.** A bad parser change can degrade chat quality in ways no offline benchmark would catch.
- **Builds a corpus over time.** Captured production traffic accumulates. Over a year, we'd have a large, real, growing eval set without manual curation.

The arguments against:

- **It is not the highest-leverage thing we could build.** The single most valuable artifact we don't have is the RFI labeled corpus. Even 50 lawyer-confirmed cases would give us a hard ground-truth eval that shadow eval cannot provide. Time spent on shadow eval is time not spent collecting labels.
- **It requires production-side changes** (the payload sink). Those changes have real reliability cost — the sink runs in the user request path, must validate before writing to a 7-year compliance store, must not affect production performance.
- **The signal is noisy.** Reference-free metrics from an LLM judge have real variance. Without ground truth to anchor against, judge bias can mislead us systematically.

**The honest framing for the team:** if we had a choice between "build shadow eval over 2 months" and "spend 2 months getting lawyers to label 100 historical cases," the labeled cases are probably more valuable for a legal product. Shadow eval is the right answer when the labeling path is blocked or much slower than the engineering path. **Is that the situation we're in?** If yes, proceed. If no, reconsider priorities.

### Decision 2 — Replay or live mirror, and in what order?

If we're doing shadow eval, the next question is which mechanism.

```svg
<svg viewBox="0 0 720 320" xmlns="http://www.w3.org/2000/svg" role="img" font-family="system-ui, -apple-system, sans-serif">
  <title>Replay vs live mirror</title>
  <defs>
    <marker id="ar2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M2 1L8 5L2 9" fill="none" stroke="#555" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker>
  </defs>

  <rect x="40" y="30" width="320" height="240" rx="10" fill="#ecfdf5" stroke="#10b981"/>
  <text x="56" y="56" font-size="14" font-weight="600" fill="#064e3b">Option A — Replay</text>
  <rect x="60" y="76" width="280" height="36" rx="5" fill="#fff" stroke="#10b981"/>
  <text x="200" y="100" font-size="12" text-anchor="middle" fill="#064e3b">Production writes payloads to S3</text>
  <rect x="60" y="124" width="280" height="36" rx="5" fill="#fff" stroke="#10b981"/>
  <text x="200" y="148" font-size="12" text-anchor="middle" fill="#064e3b">Eval Job reads S3, runs candidate</text>
  <rect x="60" y="172" width="280" height="36" rx="5" fill="#fff" stroke="#10b981"/>
  <text x="200" y="196" font-size="12" text-anchor="middle" fill="#064e3b">Compare candidate vs baseline</text>
  <line x1="200" y1="112" x2="200" y2="122" stroke="#555" marker-end="url(#ar2)"/>
  <line x1="200" y1="160" x2="200" y2="170" stroke="#555" marker-end="url(#ar2)"/>
  <text x="56" y="234" font-size="11" fill="#064e3b">Offline, deterministic, lower risk.</text>
  <text x="56" y="252" font-size="11" fill="#064e3b">Reproducible across runs.</text>

  <rect x="380" y="30" width="300" height="240" rx="10" fill="#eff6ff" stroke="#3b82f6"/>
  <text x="396" y="56" font-size="14" font-weight="600" fill="#1e3a8a">Option B — Live mirror</text>
  <rect x="400" y="76" width="260" height="36" rx="5" fill="#fff" stroke="#3b82f6"/>
  <text x="530" y="100" font-size="12" text-anchor="middle" fill="#1e3a8a">Istio mirrors live requests</text>
  <rect x="400" y="124" width="260" height="36" rx="5" fill="#fff" stroke="#3b82f6"/>
  <text x="530" y="148" font-size="12" text-anchor="middle" fill="#1e3a8a">Shadow pods process in parallel</text>
  <rect x="400" y="172" width="260" height="36" rx="5" fill="#fff" stroke="#3b82f6"/>
  <text x="530" y="196" font-size="12" text-anchor="middle" fill="#1e3a8a">Compare logs offline</text>
  <line x1="530" y1="112" x2="530" y2="122" stroke="#555" marker-end="url(#ar2)"/>
  <line x1="530" y1="160" x2="530" y2="170" stroke="#555" marker-end="url(#ar2)"/>
  <text x="396" y="234" font-size="11" fill="#1e3a8a">Real-time, current distribution.</text>
  <text x="396" y="252" font-size="11" fill="#1e3a8a">Production-side risk to manage.</text>
</svg>
```

The trade-offs:

| Property | Replay | Live mirror |
|---|---|---|
| Production-side risk | Low (writes to S3 only) | Low if mesh-level, higher if app-level |
| Reproducibility | Yes (same payloads, same result) | No (each run is a different traffic snapshot) |
| Catches current-moment distribution shift | No (corpus may be stale) | Yes |
| Required for any candidate that doesn't exist yet | Yes | No |
| Cost per evaluation | Bedrock judge calls per replay | Bedrock judge calls per mirrored request |
| Infrastructure complexity | S3 schema, replay runner | Istio VirtualService, shadow namespace, isolation |
| Engineering time to first signal | Weeks | Days (Istio is already in production) |

**A useful framing:** these aren't competitors, they're complements. Replay is the foundation; live mirror is a thin addition once replay exists.

**The team should decide:**

- Build replay first, add live mirror later? (Foundation-first.)
- Build live mirror first, add replay later? (Fastest signal, but doesn't accumulate a corpus.)
- Build both in parallel? (Faster but more concurrent work.)
- Build only one? (Cheaper, but lose half the capability.)

The previous design defaulted to "replay first, live mirror later." This document does not assume that conclusion is correct. The team should debate it.

### Decision 3 — Where does the payload sink live?

The payload sink is the production-side component that captures replay-complete payloads and writes them to S3. It's the most operationally sensitive piece of the system because it runs in the user request path.

Three options:

**Option A: Sink lives in the case-assistant production codebase.**
The case-assistant emits payloads as part of its normal request handling. eval-harness has no production-side code.

- For: clean separation of concerns; the sink follows the case-assistant's reliability discipline; eval-harness stays a pure batch-eval tool.
- Against: requires case-assistant team to do the work; cross-repo coordination on schema changes.

**Option B: Sink lives in eval-harness, deployed to the production namespace.**
eval-harness contains the sink code, and the sink is deployed alongside the case-assistant as a sidecar or daemon.

- For: schema and sink in one place; eval-harness team owns the full evaluation pipeline.
- Against: eval-harness suddenly has code in the user request path; CI/deploy story gets more complex; blast radius of an eval-harness PR includes production.

**Option C: Sink lives in a third repo dedicated to it.**
Small dedicated repo for the sink, with its own deploy lifecycle. Both case-assistant and eval-harness depend on its schema.

- For: rigorous separation; clearest ownership boundaries.
- Against: three repos to coordinate; probably over-engineered for our scale.

The schema definition is a separate question. The `contracts/` directory already exists in eval-harness and is the natural home for a `replay-payload.json` schema. Whichever option we pick for the sink, the schema should live in eval-harness `contracts/` and the sink should consume it as a contract.

**What the team should resolve:** which option, and who owns the work. The answer depends partly on Decision 1 (do we have a working production case-assistant repo with engineers to take this on?) and partly on the unresolved stubs question (3.3 above).

### Decision 4 — What is the 7-year retention design?

The data we capture is regulated case content. 7-year retention is a regulatory requirement.

This is not an engineering decision in isolation. It needs compliance and security review. The decisions to make:

**Storage location:**
- Same bucket as the existing case-records compliance store? (No duplicate storage of regulated content.)
- New bucket dedicated to replay payloads? (Cleaner architecture, but justifies why we have two copies.)
- New bucket holding additional data the compliance store doesn't capture? (Most likely the honest answer.)

**Storage tier strategy:**
- S3 Standard for hot window (recent traffic, frequent replay)
- Standard-IA for warm window
- Glacier (Instant Retrieval, Flexible, or Deep Archive) for cold window

The cost difference between Instant Retrieval and Deep Archive is roughly 4x. The retrieval latency difference is "milliseconds" vs "hours." How often do we replay year-2 traffic? The answer determines the tier.

**Object Lock mode:**
- Compliance mode (no one, including root, can delete before retention expires)
- Governance mode (specific IAM roles can delete with audit trail)

For regulatory retention, Compliance mode is the safer default. It has consequences: a buggy sink that writes malformed payloads pollutes the bucket for 7 years. This means schema validation before write is non-negotiable, and a dead-letter bucket for invalid payloads is required.

**Encryption:**
- SSE-KMS with customer-managed key (standard for regulated content)
- Key rotation policy and recovery procedures must be documented (key loss = 7 years of unreadable data)

**Account isolation:**
- Compliance bucket in a separate AWS account from eval-harness (read-only cross-account access)
- Or same account with strict IAM (simpler, weaker isolation)

**Sampling rate:**
- 100% capture (maximum fidelity, maximum cost)
- N% sampled (lower cost, sampling design complexity, possible loss of rare failure modes)

```svg
<svg viewBox="0 0 720 280" xmlns="http://www.w3.org/2000/svg" role="img" font-family="system-ui, -apple-system, sans-serif">
  <title>7-year storage tier strategy</title>
  <rect x="40" y="40" width="200" height="140" rx="10" fill="#fef3c7" stroke="#f59e0b"/>
  <text x="140" y="68" font-size="13" font-weight="600" text-anchor="middle" fill="#78350f">Hot tier</text>
  <text x="140" y="92" font-size="11" text-anchor="middle" fill="#78350f">Days 0–30</text>
  <text x="140" y="112" font-size="11" text-anchor="middle" fill="#78350f">S3 Standard</text>
  <text x="140" y="138" font-size="10" text-anchor="middle" fill="#78350f">Recent replay,</text>
  <text x="140" y="152" font-size="10" text-anchor="middle" fill="#78350f">PR-time eval</text>
  <text x="140" y="172" font-size="10" font-style="italic" text-anchor="middle" fill="#78350f">~$0.023/GB/mo</text>

  <rect x="260" y="40" width="200" height="140" rx="10" fill="#fed7aa" stroke="#ea580c"/>
  <text x="360" y="68" font-size="13" font-weight="600" text-anchor="middle" fill="#7c2d12">Warm tier</text>
  <text x="360" y="92" font-size="11" text-anchor="middle" fill="#7c2d12">Days 30–180</text>
  <text x="360" y="112" font-size="11" text-anchor="middle" fill="#7c2d12">S3 Standard-IA</text>
  <text x="360" y="138" font-size="10" text-anchor="middle" fill="#7c2d12">Occasional historical</text>
  <text x="360" y="152" font-size="10" text-anchor="middle" fill="#7c2d12">replay</text>
  <text x="360" y="172" font-size="10" font-style="italic" text-anchor="middle" fill="#7c2d12">~$0.0125/GB/mo</text>

  <rect x="480" y="40" width="200" height="140" rx="10" fill="#fecaca" stroke="#dc2626"/>
  <text x="580" y="68" font-size="13" font-weight="600" text-anchor="middle" fill="#7f1d1d">Cold tier</text>
  <text x="580" y="92" font-size="11" text-anchor="middle" fill="#7f1d1d">Day 180 → 7 years</text>
  <text x="580" y="112" font-size="11" text-anchor="middle" fill="#7f1d1d">Glacier (which?)</text>
  <text x="580" y="138" font-size="10" text-anchor="middle" fill="#7f1d1d">Bulk of corpus,</text>
  <text x="580" y="152" font-size="10" text-anchor="middle" fill="#7f1d1d">rarely accessed</text>
  <text x="580" y="172" font-size="10" font-style="italic" text-anchor="middle" fill="#7f1d1d">$1–4/TB/mo</text>

  <text x="360" y="220" font-size="12" text-anchor="middle" fill="#333" font-style="italic">Trade-off: cold-tier choice depends on how often we replay year-2 traffic.</text>
  <text x="360" y="240" font-size="12" text-anchor="middle" fill="#333" font-style="italic">Deep Archive saves ~75% on cold storage but adds hours of restore time.</text>
</svg>
```

**What the team should resolve:** all of the above, ideally with compliance and security represented in the meeting.

### Decision 5 — What is the statistical comparison rule?

Even with perfect data capture, the comparison layer is where most teams get fooled by noise.

The decisions:

**Default statistical test:**
- Paired t-test (assumes normality, which judge scores violate)
- Wilcoxon signed-rank (non-parametric, handles non-normal data)
- McNemar's test (for paired binary decisions, applicable when RFI corpus exists)

**Effect size threshold:**
- Statistical significance alone is not enough — a p<0.01 result with effect size 0.005 isn't worth shipping. What's our minimum effect size?
- Cohen's d of 0.2 is a common "small effect" threshold; 0.5 is "medium." Which do we use?

**Variance calibration:**
- LLM judges are noisy. Same input, two runs, different scores. We need to run the same candidate multiple times to estimate noise floor.
- How many repeated runs? 3? 5? 10?
- This adds cost (every repeat is another Bedrock spend) but is the only honest way to set thresholds.

**Composite rule across surfaces:**
We have three surfaces (public benchmark, production replay, differential diff). A candidate change can pass one and fail another.
- Composite rule: one pass/fail summarizing all three. More permissive.
- Independent gates: each surface must pass. More conservative.
- Weighted: each surface contributes to a score with thresholds. Most flexible, hardest to defend.

**What the team should resolve:** the default test, the effect size threshold, the variance calibration approach, and the composite rule shape.

### Decision 6 — What does the CI integration look like?

Once we can run shadow eval, the question is what to do with it.

- **Block PRs on regression?** Strong, opinionated, may cause friction.
- **Comment on PRs with comparison, but don't block?** Informational, weaker enforcement.
- **Run nightly against main, alert on regression?** Decouples eval from PR cycle.
- **All of the above?** Most coverage, most setup work.

**What the team should resolve:** what enforcement model fits our culture and our merge velocity.

---

## Part 5 — The diagrams of what we'd build

Assuming we work through the decisions above and end up roughly where the previous design landed (replay-first, sink in case-assistant codebase, Istio mirror as Phase 4), here's the picture.

### 5.1 The EKS topology

```svg
<svg viewBox="0 0 720 540" xmlns="http://www.w3.org/2000/svg" role="img" font-family="system-ui, -apple-system, sans-serif">
  <title>EKS topology — three namespaces, shared cluster</title>
  <defs>
    <marker id="ar3" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M2 1L8 5L2 9" fill="none" stroke="#555" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker>
  </defs>

  <rect x="20" y="20" width="680" height="380" rx="14" fill="#f9fafb" stroke="#6b7280" stroke-width="1"/>
  <text x="40" y="46" font-size="14" font-weight="600" fill="#111827">EKS cluster</text>
  <text x="40" y="64" font-size="11" fill="#6b7280">One cluster, three namespaces</text>

  <rect x="40" y="84" width="200" height="296" rx="10" fill="#dbeafe" stroke="#2563eb"/>
  <text x="140" y="110" font-size="13" font-weight="600" text-anchor="middle" fill="#1e3a8a">case-assistant-prod</text>
  <rect x="56" y="130" width="168" height="40" rx="5" fill="#fff" stroke="#2563eb"/>
  <text x="140" y="154" font-size="11" text-anchor="middle" fill="#1e3a8a">Chat pods</text>
  <rect x="56" y="180" width="168" height="40" rx="5" fill="#fff" stroke="#2563eb"/>
  <text x="140" y="204" font-size="11" text-anchor="middle" fill="#1e3a8a">Ingestion workers</text>
  <rect x="56" y="230" width="168" height="40" rx="5" fill="#fff" stroke="#2563eb"/>
  <text x="140" y="254" font-size="11" text-anchor="middle" fill="#1e3a8a">Vector index</text>
  <rect x="56" y="280" width="168" height="40" rx="5" fill="#fff" stroke="#2563eb" stroke-dasharray="3 3"/>
  <text x="140" y="304" font-size="11" text-anchor="middle" fill="#1e3a8a">Payload sink (new)</text>
  <text x="140" y="346" font-size="10" text-anchor="middle" fill="#1e3a8a">Owner: case-assistant team</text>
  <text x="140" y="362" font-size="10" text-anchor="middle" fill="#1e3a8a">Existing app + sink</text>

  <rect x="260" y="84" width="220" height="296" rx="10" fill="#ccfbf1" stroke="#0d9488"/>
  <text x="370" y="110" font-size="13" font-weight="600" text-anchor="middle" fill="#134e4a">eval-harness</text>
  <rect x="276" y="130" width="188" height="40" rx="5" fill="#fff" stroke="#0d9488"/>
  <text x="370" y="154" font-size="11" text-anchor="middle" fill="#134e4a">CronJob — nightly benchmarks</text>
  <rect x="276" y="180" width="188" height="40" rx="5" fill="#fff" stroke="#0d9488"/>
  <text x="370" y="204" font-size="11" text-anchor="middle" fill="#134e4a">Job — PR-triggered eval</text>
  <rect x="276" y="230" width="188" height="40" rx="5" fill="#fff" stroke="#0d9488" stroke-dasharray="3 3"/>
  <text x="370" y="254" font-size="11" text-anchor="middle" fill="#134e4a">Job — replay (new)</text>
  <rect x="276" y="280" width="188" height="40" rx="5" fill="#fff" stroke="#0d9488" stroke-dasharray="3 3"/>
  <text x="370" y="304" font-size="11" text-anchor="middle" fill="#134e4a">Job — comparison (new)</text>
  <text x="370" y="346" font-size="10" text-anchor="middle" fill="#134e4a">Owner: eval-harness team</text>
  <text x="370" y="362" font-size="10" text-anchor="middle" fill="#134e4a">Batch, spot nodes OK</text>

  <rect x="500" y="84" width="180" height="296" rx="10" fill="#fef3c7" stroke="#d97706"/>
  <text x="590" y="110" font-size="13" font-weight="600" text-anchor="middle" fill="#78350f">observability</text>
  <rect x="516" y="140" width="148" height="40" rx="5" fill="#fff" stroke="#d97706"/>
  <text x="590" y="164" font-size="11" text-anchor="middle" fill="#78350f">OTel collector</text>
  <rect x="516" y="190" width="148" height="40" rx="5" fill="#fff" stroke="#d97706"/>
  <text x="590" y="214" font-size="11" text-anchor="middle" fill="#78350f">Phoenix UI</text>
  <rect x="516" y="240" width="148" height="40" rx="5" fill="#fff" stroke="#d97706"/>
  <text x="590" y="264" font-size="11" text-anchor="middle" fill="#78350f">Trace storage</text>
  <text x="590" y="346" font-size="10" text-anchor="middle" fill="#78350f">Shared</text>

  <rect x="40" y="420" width="200" height="50" rx="6" fill="#f3e8ff" stroke="#9333ea"/>
  <text x="140" y="442" font-size="11" font-weight="600" text-anchor="middle" fill="#581c87">S3 — compliance store</text>
  <text x="140" y="458" font-size="10" text-anchor="middle" fill="#581c87">7-year, separate account</text>

  <rect x="260" y="420" width="220" height="50" rx="6" fill="#f3e8ff" stroke="#9333ea"/>
  <text x="370" y="442" font-size="11" font-weight="600" text-anchor="middle" fill="#581c87">Bedrock Claude</text>
  <text x="370" y="458" font-size="10" text-anchor="middle" fill="#581c87">Judge calls, IAM auth</text>

  <rect x="500" y="420" width="180" height="50" rx="6" fill="#f3e8ff" stroke="#9333ea"/>
  <text x="590" y="442" font-size="11" font-weight="600" text-anchor="middle" fill="#581c87">ECR</text>
  <text x="590" y="458" font-size="10" text-anchor="middle" fill="#581c87">Candidate images</text>

  <line x1="140" y1="320" x2="140" y2="416" stroke="#555" stroke-dasharray="3 3" marker-end="url(#ar3)"/>
  <line x1="370" y1="320" x2="140" y2="416" stroke="#555" stroke-dasharray="3 3" marker-end="url(#ar3)"/>
  <line x1="370" y1="320" x2="370" y2="416" stroke="#555" marker-end="url(#ar3)"/>
  <line x1="370" y1="320" x2="590" y2="416" stroke="#555" stroke-dasharray="3 3" marker-end="url(#ar3)"/>

  <text x="380" y="500" font-size="11" fill="#555" font-style="italic" text-anchor="middle">Solid = synchronous call. Dashed = async write or telemetry.</text>
  <text x="380" y="520" font-size="11" fill="#555" font-style="italic" text-anchor="middle">Dashed component boxes = not yet built.</text>
</svg>
```

### 5.2 The replay data flow

```svg
<svg viewBox="0 0 720 480" xmlns="http://www.w3.org/2000/svg" role="img" font-family="system-ui, -apple-system, sans-serif">
  <title>Replay data flow</title>
  <defs>
    <marker id="ar4" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M2 1L8 5L2 9" fill="none" stroke="#555" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker>
  </defs>

  <text x="40" y="30" font-size="13" font-weight="600" fill="#111">Production side</text>
  <rect x="40" y="44" width="180" height="56" rx="6" fill="#dbeafe" stroke="#2563eb"/>
  <text x="130" y="68" font-size="12" text-anchor="middle" fill="#1e3a8a">Lawyer request</text>
  <text x="130" y="86" font-size="10" text-anchor="middle" fill="#1e3a8a">Chat or upload</text>

  <rect x="260" y="44" width="180" height="56" rx="6" fill="#dbeafe" stroke="#2563eb"/>
  <text x="350" y="68" font-size="12" text-anchor="middle" fill="#1e3a8a">Handler + sink</text>
  <text x="350" y="86" font-size="10" text-anchor="middle" fill="#1e3a8a">Schema validated</text>

  <rect x="480" y="44" width="180" height="56" rx="6" fill="#fee2e2" stroke="#dc2626"/>
  <text x="570" y="68" font-size="12" text-anchor="middle" fill="#7f1d1d">Dead-letter</text>
  <text x="570" y="86" font-size="10" text-anchor="middle" fill="#7f1d1d">Invalid only, deletable</text>

  <line x1="220" y1="72" x2="256" y2="72" stroke="#555" marker-end="url(#ar4)"/>
  <line x1="440" y1="72" x2="476" y2="72" stroke="#555" marker-end="url(#ar4)"/>
  <text x="455" y="62" font-size="9" fill="#555">if invalid</text>

  <text x="40" y="140" font-size="13" font-weight="600" fill="#111">Compliance archive — separate account</text>
  <rect x="40" y="156" width="620" height="120" rx="8" fill="#f3e8ff" stroke="#9333ea"/>
  <text x="56" y="180" font-size="12" font-weight="600" fill="#581c87">Replay payloads bucket</text>
  <text x="56" y="196" font-size="10" fill="#581c87">SSE-KMS, Object Lock Compliance mode, 7-year retention</text>

  <rect x="56" y="208" width="190" height="50" rx="5" fill="#fff" stroke="#9333ea"/>
  <text x="151" y="228" font-size="11" font-weight="600" text-anchor="middle" fill="#581c87">Hot — Standard</text>
  <text x="151" y="246" font-size="10" text-anchor="middle" fill="#581c87">Days 0–30</text>

  <rect x="256" y="208" width="190" height="50" rx="5" fill="#fff" stroke="#9333ea"/>
  <text x="351" y="228" font-size="11" font-weight="600" text-anchor="middle" fill="#581c87">Warm — Standard-IA</text>
  <text x="351" y="246" font-size="10" text-anchor="middle" fill="#581c87">Days 30–180</text>

  <rect x="456" y="208" width="190" height="50" rx="5" fill="#fff" stroke="#9333ea"/>
  <text x="551" y="228" font-size="11" font-weight="600" text-anchor="middle" fill="#581c87">Cold — Glacier</text>
  <text x="551" y="246" font-size="10" text-anchor="middle" fill="#581c87">Day 180 to 7 years</text>

  <line x1="350" y1="100" x2="350" y2="152" stroke="#555" marker-end="url(#ar4)"/>

  <text x="40" y="316" font-size="13" font-weight="600" fill="#111">Eval side</text>
  <rect x="40" y="332" width="290" height="80" rx="8" fill="#ccfbf1" stroke="#0d9488"/>
  <text x="185" y="358" font-size="12" font-weight="600" text-anchor="middle" fill="#134e4a">Replay Job — candidate</text>
  <text x="185" y="378" font-size="10" text-anchor="middle" fill="#134e4a">Reads hot/warm, runs candidate adapter</text>
  <text x="185" y="394" font-size="10" text-anchor="middle" fill="#134e4a">Builds shadow vector index</text>

  <rect x="350" y="332" width="290" height="80" rx="8" fill="#ccfbf1" stroke="#0d9488"/>
  <text x="495" y="358" font-size="12" font-weight="600" text-anchor="middle" fill="#134e4a">Replay Job — baseline</text>
  <text x="495" y="378" font-size="10" text-anchor="middle" fill="#134e4a">Same payloads, production code</text>
  <text x="495" y="394" font-size="10" text-anchor="middle" fill="#134e4a">Controls for judge drift</text>

  <line x1="185" y1="276" x2="185" y2="328" stroke="#555" stroke-dasharray="3 3" marker-end="url(#ar4)"/>
  <line x1="495" y1="276" x2="495" y2="328" stroke="#555" stroke-dasharray="3 3" marker-end="url(#ar4)"/>
  <text x="200" y="306" font-size="9" fill="#555">cross-account read</text>

  <rect x="40" y="432" width="600" height="36" rx="6" fill="#fef3c7" stroke="#d97706"/>
  <text x="340" y="455" font-size="12" font-weight="600" text-anchor="middle" fill="#78350f">Bedrock Claude scores both runs — Wilcoxon, effect size, composite rule</text>
</svg>
```

### 5.3 The build sequence

```svg
<svg viewBox="0 0 720 500" xmlns="http://www.w3.org/2000/svg" role="img" font-family="system-ui, -apple-system, sans-serif">
  <title>Build sequence</title>
  <defs>
    <marker id="ar5" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M2 1L8 5L2 9" fill="none" stroke="#555" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker>
  </defs>

  <text x="40" y="30" font-size="13" font-weight="600" fill="#111">Phase 1 — Foundation</text>
  <rect x="40" y="44" width="300" height="60" rx="8" fill="#f3e8ff" stroke="#9333ea"/>
  <text x="190" y="68" font-size="12" font-weight="600" text-anchor="middle" fill="#581c87">Payload schema + sink</text>
  <text x="190" y="86" font-size="10" text-anchor="middle" fill="#581c87">Schema in eval-harness contracts/</text>

  <rect x="360" y="44" width="300" height="60" rx="8" fill="#f3e8ff" stroke="#9333ea"/>
  <text x="510" y="68" font-size="12" font-weight="600" text-anchor="middle" fill="#581c87">Compliance bucket + lifecycle</text>
  <text x="510" y="86" font-size="10" text-anchor="middle" fill="#581c87">Separate account, KMS, Object Lock</text>

  <text x="40" y="140" font-size="13" font-weight="600" fill="#111">Phase 2 — Core eval</text>
  <rect x="40" y="154" width="200" height="60" rx="8" fill="#ccfbf1" stroke="#0d9488"/>
  <text x="140" y="178" font-size="12" font-weight="600" text-anchor="middle" fill="#134e4a">Benchmark CI runner</text>
  <text x="140" y="196" font-size="10" text-anchor="middle" fill="#134e4a">Formalize existing</text>

  <rect x="260" y="154" width="200" height="60" rx="8" fill="#ccfbf1" stroke="#0d9488"/>
  <text x="360" y="178" font-size="12" font-weight="600" text-anchor="middle" fill="#134e4a">Replay harness</text>
  <text x="360" y="196" font-size="10" text-anchor="middle" fill="#134e4a">New CLI, new module</text>

  <rect x="480" y="154" width="200" height="60" rx="8" fill="#ccfbf1" stroke="#0d9488"/>
  <text x="580" y="178" font-size="12" font-weight="600" text-anchor="middle" fill="#134e4a">Differential diff</text>
  <text x="580" y="196" font-size="10" text-anchor="middle" fill="#134e4a">Parser, retrieval diff</text>

  <line x1="190" y1="104" x2="190" y2="150" stroke="#555" marker-end="url(#ar5)"/>
  <line x1="510" y1="104" x2="360" y2="150" stroke="#555" marker-end="url(#ar5)"/>

  <text x="40" y="250" font-size="13" font-weight="600" fill="#111">Phase 3 — Operationalize</text>
  <rect x="40" y="264" width="300" height="60" rx="8" fill="#fef3c7" stroke="#d97706"/>
  <text x="190" y="288" font-size="12" font-weight="600" text-anchor="middle" fill="#78350f">Statistical comparison</text>
  <text x="190" y="306" font-size="10" text-anchor="middle" fill="#78350f">Wilcoxon, effect size, variance</text>

  <rect x="360" y="264" width="300" height="60" rx="8" fill="#fef3c7" stroke="#d97706"/>
  <text x="510" y="288" font-size="12" font-weight="600" text-anchor="middle" fill="#78350f">CI integration</text>
  <text x="510" y="306" font-size="10" text-anchor="middle" fill="#78350f">PR gates, comments</text>

  <line x1="190" y1="214" x2="190" y2="260" stroke="#555" marker-end="url(#ar5)"/>
  <line x1="360" y1="214" x2="510" y2="260" stroke="#555" marker-end="url(#ar5)"/>

  <text x="40" y="360" font-size="13" font-weight="600" fill="#111">Phase 4 — Sequenced (Istio in place)</text>
  <rect x="40" y="374" width="300" height="60" rx="8" fill="#dbeafe" stroke="#2563eb"/>
  <text x="190" y="398" font-size="12" font-weight="600" text-anchor="middle" fill="#1e3a8a">Live mirror via Istio</text>
  <text x="190" y="416" font-size="10" text-anchor="middle" fill="#1e3a8a">VirtualService.mirror, shadow ns</text>

  <rect x="360" y="374" width="300" height="60" rx="8" fill="#e5e7eb" stroke="#6b7280"/>
  <text x="510" y="398" font-size="12" font-weight="600" text-anchor="middle" fill="#374151">Distribution shift monitor</text>
  <text x="510" y="416" font-size="10" text-anchor="middle" fill="#374151">Deferred</text>

  <line x1="190" y1="324" x2="190" y2="370" stroke="#555" stroke-dasharray="3 3" marker-end="url(#ar5)"/>

  <rect x="40" y="454" width="620" height="36" rx="6" fill="#f9fafb" stroke="#6b7280"/>
  <text x="350" y="477" font-size="12" font-style="italic" text-anchor="middle" fill="#374151">Parked: RFI sufficiency eval suite — depends on lawyer-labeled data we don't have yet</text>
</svg>
```

---

## Part 6 — The hostile questions we should ask each other

The team should be able to answer each of these before committing to any plan. If we can't answer one, that's the conversation to have first.

**On scope:**

1. Are we sure shadow eval is more valuable right now than collecting RFI labels? What's the evidence?
2. What specific regression would shadow eval catch that our existing benchmarks miss? Can we name one?
3. If shadow eval has a 6-month cost and reveals no regressions, is the corpus alone worth it?

**On architecture:**

4. Is the production case-assistant a separate codebase, or is the eval-harness stub the production code? (This blocks most other decisions.)
5. Who owns the payload sink at runtime? Who is on the pager?
6. If the sink starts dropping payloads, what's the failure mode — silent data loss, or alerts? Who acts?

**On compliance:**

7. Have we talked to legal about a 7-year retention store for case-content payloads? What's the lead time on that review?
8. Is there an existing case-records store we should write to instead of creating a new one?
9. What's our breach-response plan if the replay bucket is compromised?

**On statistics:**

10. Are we sure we can detect a real regression above LLM judge noise? Have we run the same eval twice and measured the variance?
11. What's our acceptable false-positive rate (block a good PR) vs false-negative rate (let a bad PR through)?
12. Who decides when the noise threshold is wrong and needs recalibration?

**On organization:**

13. Which team writes the sink code? Which team operates it? Are those the same people?
14. Do we have CI capacity to add 10-30 minutes to every PR for parser/retriever/chat changes?
15. What's our process if shadow eval blocks a PR that the engineer believes is fine?

**On exit:**

16. Under what circumstances would we stop using shadow eval? What would trigger a decision to abandon it?
17. If we build this and never look at the results, what should happen to the corpus?

---

## Part 7 — What this document is not

To set expectations honestly:

- **Not a proposal to approve.** It is a structured way to disagree productively.
- **Not a complete design.** Several decisions in Part 4 are unresolved on purpose. Resolving them is the team's job.
- **Not a guarantee of success.** Even with a perfect design, the eval system might reveal less than we hope, or be deprecated by a better approach (e.g., we get the RFI labels and they're sufficient).
- **Not the only path.** "Don't do this and invest in RFI labeling instead" is a valid outcome of the discussion.

---

## Part 8 — Suggested meeting structure

If the team is going to discuss this, the order matters. Suggested 90 minutes:

- **10 min** — frame the question (Part 1, Part 2). Confirm everyone agrees on what "shadow evaluation" means.
- **15 min** — Decision 1 (are we doing this at all?). If the answer is no, the meeting ends here and we go work on RFI labels instead.
- **15 min** — Decision 3 (where does the sink live?). This depends on resolving the stubs ambiguity (3.3).
- **15 min** — Decision 4 (7-year retention design). Requires compliance/security to be in the room or to follow up.
- **10 min** — Decision 2 (replay-first or mirror-first or both).
- **10 min** — Decisions 5 and 6 (statistics, CI). Can be punted to a follow-up.
- **15 min** — pick three hostile questions from Part 6 that someone owns answering before the next meeting.

We don't expect to finish the design in one meeting. We expect to finish the *decisions* that block the design.

---

## Appendix — The document's own weaknesses

To match the standard we hold our own designs to. These are real limitations of this document:

- **The cost estimates** (e.g., $0.023/GB/mo for S3 Standard) are list prices and not adjusted for our actual contract or volume.
- **The compliance team's capacity** is an unknown that affects everything in Phase 1. We have not validated their availability.
- **The statistical recommendations** (Wilcoxon, Cohen's d > 0.2) are sensible defaults but not validated against our specific judge model's noise profile. We would need to measure to know.
- **The architectural ambiguity in 3.3** is unresolved and several other parts of this document assume away. If the production code turns out to be in `stubs/`, the design changes.
- **The "diagrams show what we'd build"** in Part 5 quietly assumes the team will end up at the same design the previous document landed on. The decisions in Part 4 could legitimately produce a different shape. The diagrams are illustrative, not prescriptive.
- **The hostile questions in Part 6** are skewed toward questions I think we should ask. A team member may have hostile questions I haven't thought of. The list is a starting point.

The document gets stronger if these are challenged, not weaker.

---

*Document for team discussion. The strongest contribution it can make is to be argued with.*