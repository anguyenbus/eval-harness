# Software Dependencies Inventory

## Core Dependencies

| Software Type | Name | Version | Component | Approval Status | Function |
|---------------|------|---------|-----------|-----------------|----------|
| Library | `pydantic` | 2.13.4 | Data Validation | | Type-safe data validation and settings management |
| Library | `jsonschema` | 4.26.0 | Data Validation | | JSON schema validation |
| Library | `pandas` | 3.0.3 | Data Processing | | Tabular data manipulation and analysis |
| Library | `polars` | 1.41.0 | Data Processing | | High-performance DataFrame library |
| Library | `pyyaml` | 6.0.3 | Config | | YAML configuration file parsing |
| Library | `jinja2` | 3.1.6 | Templating | | Template rendering for reports |
| Library | `pypdf` | 6.11.0 | Document Parsing | | PDF text extraction |
| Library | `docling` | 2.93.0 | Document Parsing | | Advanced document parsing and OCR |
| Library | `rapidfuzz` | 3.14.5 | Text Processing | | Fast fuzzy string matching |
| Library | `apted` | 1.0.3 | Text Processing | | Tree edit distance calculation |
| Library | `beautifulsoup4` | 4.14.3 | HTML Parsing | | HTML/XML parsing |
| Library | `lxml` | 6.1.0 | HTML Parsing | | Fast XML/HTML processing |
| Library | `nltk` | 3.9.4 | NLP | | Natural language processing utilities |
| Library | `sacrebleu` | 2.6.0 | Evaluation | | BLEU score calculation for translation |
| Library | `chromadb` | 1.5.9 | Vector DB | | Vector database for embeddings |
| Library | `huggingface-hub` | 1.15.0 | ML/AI | | HuggingFace model and dataset access |
| Library | `torch` | 2.12.0 | ML/AI | | Deep learning framework |
| Library | `torchmetrics[detection]` | 1.9.0 | ML/AI | | Evaluation metrics for PyTorch |
| Library | `sentence-transformers` | 5.5.0 | ML/AI | | Sentence embeddings for semantic similarity |
| Library | `evaluate` | 0.4.6 | Evaluation | | HuggingFace evaluation metrics |
| Library | `pycocotools` | 2.0.11 | Evaluation | | COCO evaluation metrics for object detection |
| SDK | `openai` | 2.37.0 | LLM API | | OpenAI API client |
| SDK | `anthropic` | 0.102.0 | LLM API | | Anthropic Claude API client |
| Library | `langchain-openai` | 1.2.1 | LLM Framework | | LangChain integration for OpenAI |
| Library | `datasets` | 4.8.5 | ML/AI | | HuggingFace datasets library |
| Library | `deepeval` | 4.0.3 | Evaluation | | LLM evaluation framework |
| Library | `beartype` | 0.22.9 | Type Safety | | Runtime type checking |
| Library | `python-dotenv` | 1.2.2 | Config | | Environment variable management from .env |

## Optional Dependencies

### Phoenix (Observability)

| Software Type | Name | Version | Component | Approval Status | Function |
|---------------|------|---------|-----------|-----------------|----------|
| Library | `arize-phoenix` | 15.11.1 | Observability | | LLM tracing and evaluation |
| Library | `openinference-instrumentation-openai` | 0.1.49 | Observability | | OpenAI tracing instrumentation |

### Replay

| Software Type | Name | Version | Component | Approval Status | Function |
|---------------|------|---------|-----------|-----------------|----------|
| Library | `arize-phoenix` | 15.11.1 | Observability | | LLM tracing and evaluation |
| Library | `openinference-semantic-conventions` | 0.1.29 | Observability | | OpenInference semantic standards |
| Framework | `fastapi` | 0.136.1 | Web Server | | Async web framework for replay API |
| Server | `uvicorn` | 0.47.0 | Web Server | | ASGI server for FastAPI |
| Library | `click` | 8.3.3 | CLI | | Command-line interface creation |
| Library | `faiss-cpu` | N/A | Vector Search | | Efficient similarity search (not installed) |

### Bedrock

| Software Type | Name | Version | Component | Approval Status | Function |
|---------------|------|---------|-----------|-----------------|----------|
| SDK | `boto3` | 1.43.11 | AWS SDK | | AWS SDK for Bedrock access |

## Development Dependencies

| Software Type | Name | Version | Component | Approval Status | Function |
|---------------|------|---------|-----------|-----------------|----------|
| Framework | `pytest` | 9.0.3 | Testing | | Test framework |
| Plugin | `pytest-cov` | 7.1.0 | Testing | | Coverage reporting for pytest |
| Linter | `ruff` | 0.15.13 | Quality | | Fast Python linter and formatter |
| Formatter | `black` | 26.5.0 | Quality | | Code formatter |
| Linter | `mypy` | 2.1.0 | Type Checking | | Static type checker |
| Types | `types-pyyaml` | 6.0.12.20260510 | Type Checking | | Stubs for pyyaml |
| Types | `types-jsonschema` | 4.26.0.20260508 | Type Checking | | Stubs for jsonschema |
