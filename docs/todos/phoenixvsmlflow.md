# Note: Live Evaluation Capability — Phoenix vs MLflow

**To:** [Manager]
**From:** [Your name]
**Date:** 23 May 2026
**Subject:** Live evaluation capability gap between Phoenix (our current tool) and MLflow, and what I recommend we do about it
**Reading time:** 4 minutes

---

## The bottom line

**Phoenix OSS — the tool we currently use — does not include continuous online evaluation with alerting. MLflow OSS does. This is a verified capability gap, not a marketing comparison.**

My recommendation: **stay on Phoenix and defer the live-evaluation decision to post-launch (after November)**. The gap is real, but we do not need live evaluation before we have real production traffic, and the migration cost during our pre-launch window is not justified by the marginal benefit.

This note exists so you have the facts and citations on hand if the question comes up before then.

---

## What "live evaluation" means

Live evaluation (also called online evaluation or automatic evaluation) is the pattern where LLM judges run automatically and continuously on production traffic as it arrives, with sampling, filtering, and alerting on threshold breaches. It is distinct from:

- **Offline evaluation** (running judges on a curated dataset, on demand) — both Phoenix and MLflow do this well
- **Shadow / replay evaluation** (comparing a candidate to a baseline on captured production traffic) — both tools support this
- **Live monitoring without judges** (latency, cost, errors) — both tools cover this too

The gap is specifically about *continuous LLM-judge scoring of live production traffic with alerts*.

---

## Verified facts (with citations)

### Phoenix OSS: no built-in continuous online evaluation

From the Phoenix official documentation, Evaluation page:

> "For continuous monitoring of application performance — evals on production traffic with alerting and threshold-based triggers — see Arize AX Online Evals."

**Source:** https://arize.com/docs/phoenix/evaluation/llm-evals (verified 23 May 2026)

This is Phoenix's own documentation directing users to Arize AX — the paid commercial tier — for this specific capability. Phoenix OSS provides the *primitives* (trace storage, evaluator SDK, annotation API) but not the *orchestration layer* that runs judges continuously with alerting.

### MLflow OSS: built-in automatic evaluation

From the MLflow official documentation, Automatic Evaluation page:

> "Automatic evaluation runs your LLM judges automatically on traces and multi-turn conversations as they're logged to MLflow, without requiring manual execution of code."

And on the underlying mechanism:

> "LLM judges are periodically executed securely within the MLflow server as new traces and multi-turn conversations are received. Evaluation happens asynchronously and does not block trace logging, so your application's performance is unaffected."

**Source:** https://mlflow.org/docs/latest/genai/eval-monitor/automatic-evaluations/ (verified 23 May 2026)

MLflow OSS ships this as a first-class feature, including configurable sampling rates per judge, filtering by trace metadata, session-level evaluation for multi-turn conversations, and built-in trend visualization in the MLflow UI.

---

## Arguments for switching to MLflow

1. **Continuous online evaluation works out of the box.** Configure a judge in the UI or with three lines of SDK code, set a sample rate, and it runs automatically on production traffic. No custom polling worker to maintain.

2. **Trend dashboards are built in.** MLflow shows quality and performance trends over time in its native UI. With Phoenix, we would build this in Grafana with custom dashboards.

3. **Production-grade scaling story.** MLflow is designed for production tracking workloads at scale. Phoenix OSS is single-node by design; the multi-node scaling story is the paid Arize AX tier.

4. **Unified ecosystem.** If we later add classical ML work (fine-tuning embedding models, training case-type classifiers), MLflow tracks both LLM and traditional ML in one platform. Phoenix only does LLM.

5. **AI Gateway endpoints.** MLflow provides a managed gateway for judge LLM calls, which simplifies security review for judges that call Bedrock from production infrastructure.

---

## Arguments against switching (and for staying on Phoenix)

1. **Phoenix is already deployed and working in our production EKS cluster.** Engineers know the tool. Production spans flow into it. The cost of switching is real engineering time we do not currently have.

2. **Migration cost during pre-launch is unjustified.** I estimate 1-2 weeks to migrate instrumentation, re-learn the API, and validate parity. That is 5-10% of our remaining pre-launch runway, spent on something that does not retire any of our top product risks (real-document validation, RFI labeling, P0 fixes in eval-harness).

3. **We will not need continuous online evaluation before November.** There is no production traffic to evaluate continuously until users arrive. Until then, live evaluation is just expensive batch evaluation.

4. **Phoenix is genuinely stronger for RAG-specific work.** Built-in faithfulness, relevance, hallucination, and citation evaluators tuned for RAG. MLflow's RAG support is competent but less mature.

5. **The live-evaluation gap on Phoenix can be closed by writing custom code post-launch.** A polling worker that fetches new traces, samples them, runs evaluators, and writes scores back via Phoenix's annotation API is approximately 5-7 days of work plus ongoing maintenance. Not free, but bounded.

6. **The framework we build between now and November is mostly tool-portable.** Datasets, evaluators, statistical comparison, CI integration — these are not Phoenix-specific. If we revisit the platform decision post-launch, we are not throwing away most of the work.

---

## My recommendation

**Three options, in order of my preference:**

**Option C (recommended): Defer the decision to post-launch.** Build the offline and replay layers on Phoenix between now and November. Once we have real production traffic and concrete monitoring requirements, revisit live evaluation with actual data. Decide between Options A and B then.

**Option A: Build live evaluation on Phoenix ourselves, post-launch.** Polling worker, sampling controls, Slack alerting, Grafana dashboards. Roughly 5-7 days of work, plus ongoing maintenance. Keeps us on a tool the team knows.

**Option B: Switch to MLflow OSS, post-launch (or pre-launch if the team strongly disagrees).** Get continuous evaluation for free, accept migration cost and loss of Phoenix-specific RAG features.

**I do not recommend switching before November.** The team has 5 months and several higher-priority risks to retire. Migrating tools mid-flight is the kind of work that absorbs engineering attention without retiring any of those risks.

---

## What I would like from you

Two things:

1. **Confirm that deferring the live-evaluation decision to post-launch is acceptable.** This unblocks the team to commit to the Phoenix-native shadow eval design without the platform question lingering.

2. **Surface this to anyone else who might raise it.** If finance, security, or another engineering team has a concern about continuous monitoring requirements before November, it is better to learn that now than mid-build.

Happy to discuss live or write a longer version. The full shadow evaluation design document with this framing already integrated is on Confluence [link / location].

---

## Appendix: source notes

- Both quotes above are from official documentation pages, not marketing materials. I verified both URLs are live and the quoted text is present in the page as of 23 May 2026.
- I checked third-party comparisons (Confident AI, ZenML, etc.) and found general agreement on this specific point, though many third-party sources have their own product biases. The primary sources above are the defensible citation.
- The "5-7 days to build on Phoenix" estimate is my engineering judgment based on the work scope (polling worker, idempotency, sampling, alerting, dashboards). It is not validated against a real implementation. Conservative engineers should budget 7-10 days; optimistic ones might do it in 3-4.