"""Tests for LegalBench-RAG loader."""

from eval_harness.datasets.legalbench_rag import load_legalbench_rag


class TestLegalBenchRAGLoader:
    """Test suite for LegalBench-RAG dataset loader."""

    def test_mini_slice_loads_queries(self, tmp_path):
        """Test that mini slice loads queries."""
        # Create tiny fixture dataset
        corpus_dir = tmp_path / "cuad"
        corpus_dir.mkdir(parents=True)

        import json

        queries_data = {
            "queries": [
                {
                    "query_id": "q001",
                    "query_text": "What is the termination clause?",
                    "gold_spans": [[0, 50]],
                    "gold_answer_text": "The contract can be terminated with 30 days notice.",
                },
                {
                    "query_id": "q002",
                    "query_text": "What is the payment schedule?",
                    "gold_spans": [[100, 150]],
                    "gold_answer_text": "Payments are due monthly.",
                },
            ],
        }

        (corpus_dir / "queries.json").write_text(json.dumps(queries_data))

        results = list(load_legalbench_rag(tmp_path, slice="mini"))

        assert len(results) == 2

    def test_returns_correct_tuple_structure(self, tmp_path):
        """Test that loader returns (query_id, query_text, gold_spans, gold_answer_text)."""
        corpus_dir = tmp_path / "maud"
        corpus_dir.mkdir(parents=True)

        import json

        queries_data = {
            "queries": [
                {
                    "query_id": "q001",
                    "query_text": "Test question?",
                    "gold_spans": [[0, 25], [50, 75]],
                    "gold_answer_text": "Test answer.",
                },
            ],
        }

        (corpus_dir / "queries.json").write_text(json.dumps(queries_data))

        results = list(load_legalbench_rag(tmp_path, slice="mini"))

        assert len(results) == 1
        query_id, query_text, gold_spans, gold_answer_text = results[0]

        assert query_id == "q001"
        assert query_text == "Test question?"
        assert gold_spans == [[0, 25], [50, 75]]
        assert gold_answer_text == "Test answer."

    def test_iterator_pattern_works(self, tmp_path):
        """Test that loader uses iterator pattern."""
        corpus_dir = tmp_path / "contract_nli"
        corpus_dir.mkdir(parents=True)

        import json

        queries_data = {
            "queries": [
                {
                    "query_id": f"q{i:03d}",
                    "query_text": f"Question {i}?",
                    "gold_spans": [[0, 10]],
                    "gold_answer_text": f"Answer {i}.",
                }
                for i in range(10)
            ],
        }

        (corpus_dir / "queries.json").write_text(json.dumps(queries_data))

        result = load_legalbench_rag(tmp_path, slice="mini")
        assert hasattr(result, "__iter__")

        count = sum(1 for _ in result)
        assert count == 10

    def test_configurable_root_path(self, tmp_path):
        """Test that loader accepts configurable root path."""
        data_dir = tmp_path / "data" / "rag" / "legalbench"
        corpus_dir = data_dir / "privacyqa"
        corpus_dir.mkdir(parents=True)

        import json

        queries_data = {"queries": []}
        (corpus_dir / "queries.json").write_text(json.dumps(queries_data))

        results = list(load_legalbench_rag(data_dir, slice="mini"))

        # Should work without error
        assert isinstance(results, list)

    def test_full_slice_loads_from_subcorpora(self, tmp_path):
        """Test that full slice loads queries from sub-corpora directories."""
        # Create subcorpora with queries
        for corpus_name in ["cuad", "maud"]:
            corpus_dir = tmp_path / corpus_name
            corpus_dir.mkdir(parents=True)

            import json

            queries_data = {
                "queries": [
                    {
                        "query_id": f"{corpus_name}_q001",
                        "query_text": f"{corpus_name} question?",
                        "gold_spans": [[0, 25]],
                        "gold_answer_text": f"{corpus_name} answer.",
                    },
                ],
            }

            (corpus_dir / "queries.json").write_text(json.dumps(queries_data))

        results = list(load_legalbench_rag(tmp_path, slice="full"))

        # Full slice should load from all subcorpora when no full/ directory exists
        assert len(results) == 2
