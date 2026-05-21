# Why DeepEval: Technical Comparison and Justification

## Introduction: Evaluating AI Quality 

When we build AI systems that answer questions, we need to know: **Is the AI actually helping users, or is it making things up?** This is the challenge of AI evaluation.

Imagine asking a lawyer a question. A good lawyer gives you an accurate answer, cites relevant laws, and admits when they don't know something. A bad lawyer might sound confident but be wrong, miss important details, or hallucinate cases that don't exist.

Our RAG (Retrieval-Augmented Generation) systems work similarly—they search through documents and then generate answers. We need to test these systems to ensure they're reliable, accurate, and helpful.

**DeepEval is better than RAGAS for three main reasons:**

### 1. Better Conversational Testing

Users ask follow-up questions, request clarifications, and build context over multiple back-and-forth exchanges—just like a real conversation.

**DeepEval** handles these multi-turn conversations naturally. It can track whether the AI remembers what was discussed earlier, stays focused on the user's needs, and maintains consistency throughout the conversation.

**RAGAS**, the alternative we considered, treats conversations as a series of disconnected Q&A pairs. It's like evaluating each sentence in a conversation without understanding the flow.

### 2. Easier Debugging When Things Go Wrong

When an AI gives a wrong answer, we need to know **why**. Was the wrong information retrieved from our database? Did the AI misinterpret what it found? Did it hallucinate something entirely?

**DeepEval** provides three levels of explanation:

1. **Overall explanation**: "The score is low because the answer contradicts the retrieved documents"
2. **Chunk-by-chunk analysis**: "Chunk 1 was relevant, Chunk 2 was irrelevant, Chunk 3 was partially relevant"
3. **Claim verification**: "The AI claimed 'X' but the documents only support 'Y'"

This is like having an expert reviewer mark up exactly where things went wrong. **RAGAS** only gives you a numeric score—you know something failed, but not why or where.

### 3. Future-Ready for Advanced AI Applications

AI is evolving. Soon we'll have systems that can:
- Perform multi-step research
- Use external tools (searching case databases, filing forms)
- Maintain specific roles (paralegal, research assistant, document reviewer)

**DeepEval** already has built-in tests for these advanced capabilities. **RAGAS** is focused only on basic question-answering and would require custom work to handle these newer use cases.

### 4. Flexibility: Use Any AI Model

Different organizations use different AI providers. Some use OpenAI, some use AWS (Bedrock), some use Google, and some run their own local models.

**DeepEval** works with all of them:
- **OpenAI** (GPT-4, GPT-4o, GPT-4o-mini)
- **AWS Bedrock** (Claude, Titan models)
- **Anthropic** (Claude directly)
- **Google** (Gemini)
- **Azure OpenAI**
- **Local models** (Ollama, custom deployments)

**Why this matters:** We're not locked into one vendor. If OpenAI raises prices or Claude becomes better, we can switch by changing one line of configuration. Our tests continue working without modification.

**RAGAS** supports multiple providers too, but DeepEval's integration is more seamless and actively maintained with newer providers added regularly.

**In short:** DeepEval gives us better testing today, works with any AI provider, and is ready for the AI systems of tomorrow. The debugging capabilities alone save hours when improving our systems—we can see exactly what went wrong instead of guessing from a single number.

---

## Technical Deep Dive

The following sections provide technical justification with verified sources for engineering review.

This document explains why **DeepEval** was chosen over RAGAS for the eval-harness RAG evaluation framework. DeepEval offers superior versatility, debugging capabilities, and future-proofing through native multi-turn conversation evaluation, comprehensive prompt evaluation, and production-ready observability.

## Key Advantages Over RAGAS

### 1. Native Multi-Turn Conversation Evaluation

**DeepEval treats multi-turn evaluation as a first-class feature**, not an add-on. This is critical for legal RAG applications where users ask follow-up questions, request clarifications, and build context over multiple interactions.

#### DeepEval Multi-Turn Capabilities

| Metric | Purpose |
|--------|---------|
| **Knowledge Retention** | Evaluates whether chatbot retains factual information throughout conversation |
| **Conversation Completeness** | Measures whether chatbot satisfies user needs throughout conversation |
| **Turn Relevancy** | Evaluates whether chatbot generates consistently relevant responses across turns |
| **Role Adherence** | Evaluates whether chatbot adheres to assigned role throughout conversation |
| **Multi-Turn MCP Use** | Evaluates MCP server usage across conversation turns |
| **Conversational G-Eval** | Custom criteria evaluation for entire conversations using chain-of-thought |

**Source:** [DeepEval Metrics Documentation](https://github.com/confident-ai/deepeval#-metrics-and-features)

#### RAGAS Multi-Turn Support

RAGAS offers only the **AspectCritic metric** for multi-turn conversations, which provides binary outcome evaluation. This is fundamentally limited compared to DeepEval's comprehensive multi-turn metric suite.

**Source:** [RAGAS Multi-Turn Documentation](https://docs.ragas.io/en/stable/howtos/applications/evaluating_multi_turn_conversations/)

#### Comparison Table

| Capability | DeepEval | RAGAS |
|------------|----------|-------|
| Multi-turn metrics | 6+ specialized metrics | 1 (AspectCritic) |
| Conversation-level scoring | Native | Limited |
| Turn-by-turn analysis | Native (ConversationalDAG) | No |
| Context retention evaluation | Yes | No |
| Role adherence tracking | Yes | No |

### 2. Enhanced Debugging Through Reasoning Extraction

DeepEval provides **three layers of explainability** that RAGAS lacks:

#### Layer 1: Overall Reason (L1)
Each metric includes a natural language explanation of why a specific score was given.

```python
# Example from actual evaluation
faithfulness.reason = "The score is 1.00 because there are no contradictions between the actual output and the retrieval context."
```

#### Layer 2: Per-Chunk Verdicts (L2)
ContextualPrecisionMetric provides yes/no judgments for each retrieved chunk with rationale:

```json
[
  {"verdict": "no", "reason": "Chunk discusses jury selection, not excusing jurors"},
  {"verdict": "no", "reason": "Chunk mentions trial judge directions, irrelevant"},
  {"verdict": "yes", "reason": "Chunk directly addresses excusing jurors criteria"}
]
```

This directly identifies **which chunks failed retrieval and why** - critical for debugging RAG pipelines.

#### Layer 3: Claim Analysis (L3)
FaithfulnessMetric extracts claims and verifies them against context:

```python
faithfulness.claims = [
  {"claim": "Judge must excuse Bob", "truth": "Partial truth - depends on impartiality"}
]
faithfulness.truths = [
  {"statement": "Court may excuse if person cannot consider case impartially"}
]
```

**RAGAS Limitation:** RAGAS provides only numeric scores without this granular explainability.

### 3. Prompt Evaluation and Custom Criteria

#### G-Eval: Research-Backed Custom Evaluation

DeepEval's **G-Eval** framework allows evaluation against **any custom criteria** using natural language:

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase

metric = GEval(
    name="LegalAccuracy",
    criteria="Evaluate whether legal advice is accurate and cites relevant case law",
    evaluation_params=[
        "precision",
        "recall",
        "legal_correctness"
    ]
)
```

**Key Features:**
- Chain-of-thought (CoT) reasoning for human-like evaluation
- Supports completely custom evaluation criteria
- Research-backed methodology

**Source:** [G-Eval Documentation](https://deepeval.com/docs/metrics-llm-evals)

**Conversational G-Eval** extends this to entire conversations:
- Evaluates quality of each turn
- Evaluates conversation overall
- JSON-formatted results for programmatic analysis

**RAGAS Limitation:** RAGAS has no equivalent to G-Eval's flexible custom criteria evaluation.

### 4. Production Observability Integration

Our implementation demonstrates DeepEval's superior observability:

**Phoenix Integration (Implemented):**
- Evaluation traces appear in Phoenix UI with full reasoning
- Per-chunk verdicts visible and expandable
- Span hierarchy: `eval_run` → `rag_query` → `retrieval/generation/evaluation`

**Output Files (Implemented):**
- CSV: Scores only (backward compatible)
- `_summary.json`: Aggregated metrics
- `_details.json`: Full reasoning per query with L1/L2/L3 explainability

**RAGAS Limitation:** RAGAS requires third-party integrations for similar observability. DeepEval's native support enables:

1. **Faster debugging** - Root cause analysis directly in evaluation output
2. **Production monitoring** - Same metrics in dev and prod
3. **Trace-backed debugging** - Link evaluation results to traces

### 5. Agentic and Advanced Use Cases

DeepEval provides metrics specifically designed for AI agents:

| Metric | Purpose |
|--------|---------|
| **Task Completion** | Evaluates whether agent accomplished its goal |
| **Tool Correctness** | Checks if right tools called with right arguments |
| **Goal Accuracy** | Measures accuracy of goal achievement |
| **Step Efficiency** | Evaluates whether agent took unnecessary steps |
| **Plan Adherence** | Checks if agent followed expected plan |
| **Plan Quality** | Evaluates quality of agent's plan |

**Source:** [DeepEval Agentic Metrics](https://github.com/confident-ai/deepeval#-metrics-and-features)

**RAGAS Limitation:** RAGAS has no agentic evaluation capabilities. As legal RAG systems evolve toward agent-like workflows (multi-step reasoning, tool use), DeepEval provides ready-to-use evaluation metrics.

### 6. Self-Hosted Deployment

DeepEval can be deployed **without Confident AI platform**:

```python
# Completely self-hosted evaluation
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric

metric = FaithfulnessMetric(
    model="gpt-4o-mini",  # Or Bedrock, or any OpenAI-compatible model
    include_reason=True   # Enable reasoning extraction
)

# Runs locally, no external platform required
test_results = evaluate([test_case], [metric])
```

**Deployment Options:**
- Local execution
- CI/CD pipeline integration
- Kubernetes deployment (async batch evaluation with semaphore control)
- Custom LLM backends (OpenAI, AWS Bedrock, local models)

**RAGAS Comparison:** RAGAS also supports self-hosted evaluation, but lacks:
1. Native reasoning extraction
2. Agentic metrics
3. Comprehensive multi-turn support

## Future-Proofing Considerations

### 1. Legal RAG Evolution

Legal RAG applications are evolving toward:
- **Conversational interfaces** - Multi-turn Q&A about legal documents
- **Agent-like workflows** - Multi-step legal analysis
- **Tool use** - Integration with legal databases, case law search
- **Role adherence** - Maintaining legal assistant persona

DeepEval has native metrics for all these use cases **today**.

### 2. Evaluation Framework Maturity

DeepEval's 2026 roadmap focuses on:
- Reliability improvements
- Enhanced observability
- Easier evaluation across real-world LLM systems

**Source:** [DeepEval 2026 Changelog](https://deepeval.com/changelog/changelog-2026)

### 3. Community and Ecosystem

- **Active Discord community** for support
- **Regular releases** with new metrics
- **Integration partners**: LangChain, LlamaIndex, Phoenix, LangSmith
- **Academic research backing** for metrics

### 6. Multi-Model Provider Support

DeepEval supports evaluation using **any major LLM provider**, giving organizations flexibility to choose or switch vendors without rewriting evaluation code.

#### Supported Providers

| Provider | Models | Integration |
|----------|--------|-------------|
| **OpenAI** | GPT-4, GPT-4o, GPT-4o-mini | Native |
| **AWS Bedrock** | Claude, Titan, Llama | Native (Bedrock Runtime Converse API) |
| **Anthropic** | Claude 3.5 Sonnet, Opus | Native |
| **Google** | Gemini Pro/Ultra | Native |
| **Azure OpenAI** | GPT models | Native |
| **Local** | Ollama, custom | Custom LLM configuration |

**Source:** [DeepEval 2025 Changelog](https://deepeval.com/changelog/changelog-2025), [Amazon Bedrock Integration](https://deepeval.com/integrations/models/amazon-bedrock)

#### Implementation Example

Our code demonstrates this flexibility:

```python
# Uses OpenAI by default
evaluator = DeepEvalEvaluator(
    llm_provider="openai",
    judge_model="gpt-4o-mini"  # Cost-effective option
)

# Can switch to AWS Bedrock with one line change
evaluator = DeepEvalEvaluator(
    llm_provider="bedrock",
    judge_model="anthropic.claude-3-5-sonnet-20241022-v2:0"
)
```

**RAGAS Comparison:** RAGAS also supports multiple providers, but DeepEval's provider support is more comprehensive and actively updated with newer integrations (e.g., Bedrock Runtime Converse API added in 2025).

**Business Impact:**
- No vendor lock-in
- Cost optimization (switch to cheaper models when appropriate)
- Geographic compliance (use local models for data residency)
- Redundancy (switch providers during outages)

## Cost Considerations

Our implementation uses `gpt-4o-mini` as the default judge model:

**Cost Comparison (approximate):**
- gpt-4o: ~$5.00 / 1M input tokens
- gpt-4o-mini: ~$0.15 / 1M input tokens (~33x cheaper)

**Actual Results (Legal RAG Bench nano):**
- 10 queries, 4 metrics each
- Average metric computation time: ~20 seconds
- Estimated cost: <$0.01 per evaluation

The reasoning extraction adds minimal overhead while providing maximum debuggability.

## Migration Notes

### From RAGAS to DeepEval

| RAGAS Metric | DeepEval Equivalent |
|--------------|---------------------|
| `faithfulness` | `FaithfulnessMetric` |
| `context_precision` | `ContextualPrecisionMetric` |
| `context_recall` | `ContextualRecallMetric` |
| `answer_relevancy` | `AnswerRelevancyMetric` |

**Additional DeepEval Metrics (No RAGAS Equivalent):**
- `KnowledgeRetention` (multi-turn)
- `ConversationCompleteness` (multi-turn)
- `GEval` (custom criteria)
- All agentic metrics

### Breaking Changes

1. **Reasoning extraction**: Requires `include_reason=True` (now default)
2. **Verdicts access**: Use `getattr(metric, "verdicts", [])` pattern
3. **Async evaluation**: Requires asyncio event loop management

## Conclusion

DeepEval provides superior capabilities for legal RAG evaluation:

1. **Multi-turn conversation evaluation** - First-class support with 6+ specialized metrics
2. **Enhanced debugging** - Three-layer reasoning extraction (L1: reason, L2: verdicts, L3: claims)
3. **Prompt evaluation** - G-Eval for any custom criteria using natural language
4. **Production observability** - Native Phoenix integration with full reasoning traces
5. **Future-proof** - Agentic metrics, role adherence, conversation-level scoring
6. **Cost-effective** - gpt-4o-mini provides 33x cost savings with quality results

The implementation in eval-harness demonstrates these advantages through:
- Per-chunk verdicts showing **which** retrieval failed and **why**
- Claim analysis identifying hallucinations at the statement level
- Phoenix traces linking evaluation results to full execution context
- _details.json output for offline analysis

## Sources

- [DeepEval Official Documentation](https://deepeval.com/docs)
- [DeepEval vs RAGAS Comparison](https://deepeval.com/blog/deepeval-vs-ragas)
- [Multi-Turn Evaluation Guide](https://deepeval.com/guides/guides-multi-turn-evaluation)
- [G-Eval Documentation](https://deepeval.com/docs/metrics-llm-evals)
- [Conversational G-Eval](https://deepeval.com/docs/metrics-conversational-g-eval)
- [Multi-Turn Test Cases](https://deepeval.com/docs/evaluation-multiturn-test-cases)
- [DeepEval GitHub Repository](https://github.com/confident-ai/deepeval)
- [RAGAS Multi-Turn Documentation](https://docs.ragas.io/en/stable/howtos/applications/evaluating_multi_turn_conversations/)
- [DeepEval 2026 Changelog](https://deepeval.com/changelog/changelog-2026)
- [DeepEval 2025 Changelog - Provider Support](https://deepeval.com/changelog/changelog-2025)
- [Amazon Bedrock Integration](https://deepeval.com/integrations/models/amazon-bedrock)
- [Custom LLM Guide](https://deepeval.com/guides/guides-using-custom-llms)
- [Best AI Evaluation Tools 2026](https://www.confident-ai.com/knowledge-base/compare/best-ai-evaluation-tools-for-prompt-experimentation-2026)
