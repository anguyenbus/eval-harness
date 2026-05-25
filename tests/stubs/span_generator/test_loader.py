"""
Tests for span generator loader and runner components.
"""

from unittest.mock import patch

from eval_harness.stubs.span_generator.loader import (
    GeneratorQuestion,
    _generate_case_id,
    _generate_tenant_id_hashed,
    iter_questions,
    sample_questions,
)


class TestGeneratorQuestion:
    """Tests for GeneratorQuestion dataclass."""

    def test_generator_question_fields(self) -> None:
        """Test that GeneratorQuestion contains all required fields."""
        question = GeneratorQuestion(
            id="q1",
            question="What is this?",
            expected_answer="This is a test",
            relevant_passage_id="doc1",
            case_id="synth-case-0001",
            tenant_id_hashed="synth-tenant-1",
        )

        assert question.id == "q1"
        assert question.question == "What is this?"
        assert question.expected_answer == "This is a test"
        assert question.relevant_passage_id == "doc1"
        assert question.case_id == "synth-case-0001"
        assert question.tenant_id_hashed == "synth-tenant-1"


class TestCaseIdGeneration:
    """Tests for synthetic case ID generation."""

    def test_generate_case_id_format(self) -> None:
        """Test that case IDs follow the expected format."""
        case_id = _generate_case_id(0)
        assert case_id == "synth-case-0000"

        case_id = _generate_case_id(1)
        assert case_id == "synth-case-0001"

        case_id = _generate_case_id(9999)
        assert case_id == "synth-case-9999"

    def test_generate_case_id_zero_padded(self) -> None:
        """Test that case IDs are zero-padded to 4 digits."""
        case_id = _generate_case_id(5)
        assert case_id == "synth-case-0005"


class TestTenantIdGeneration:
    """Tests for synthetic tenant ID generation."""

    def test_generate_tenant_id_modulo_10(self) -> None:
        """Test that tenant IDs cycle through 0-9."""
        assert _generate_tenant_id_hashed(0) == "synth-tenant-0"
        assert _generate_tenant_id_hashed(1) == "synth-tenant-1"
        assert _generate_tenant_id_hashed(9) == "synth-tenant-9"
        assert _generate_tenant_id_hashed(10) == "synth-tenant-0"
        assert _generate_tenant_id_hashed(15) == "synth-tenant-5"


class TestQuestionLoader:
    """Tests for question iteration."""

    def test_iter_questions_yields_generator_question(self) -> None:
        """Test that iter_questions yields GeneratorQuestion objects."""
        # Mock the dataset loader
        with patch(
            "eval_harness.stubs.span_generator.loader.load_legal_rag_bench"
        ) as mock_loader:
            mock_loader.return_value = iter(
                [
                    ("q1", "Question 1", "passage1", "Answer 1"),
                    ("q2", "Question 2", "passage2", "Answer 2"),
                ]
            )

            questions = list(iter_questions(limit=2))

            assert len(questions) == 2
            assert isinstance(questions[0], GeneratorQuestion)
            assert questions[0].id == "q1"
            assert questions[0].question == "Question 1"
            assert questions[0].expected_answer == "Answer 1"
            assert questions[0].relevant_passage_id == "passage1"
            assert questions[0].case_id == "synth-case-0000"
            assert questions[0].tenant_id_hashed == "synth-tenant-0"

    def test_iter_questions_respects_limit(self) -> None:
        """Test that iter_questions respects the limit parameter."""
        with patch(
            "eval_harness.stubs.span_generator.loader.load_legal_rag_bench"
        ) as mock_loader:
            # Return 10 questions
            mock_loader.return_value = iter(
                [(f"q{i}", f"Q{i}", f"p{i}", f"A{i}") for i in range(10)]
            )

            questions = list(iter_questions(limit=5))

            assert len(questions) == 5

    def test_iter_questions_increments_case_id(self) -> None:
        """Test that case IDs increment with each question."""
        with patch(
            "eval_harness.stubs.span_generator.loader.load_legal_rag_bench"
        ) as mock_loader:
            mock_loader.return_value = iter(
                [
                    ("q1", "Q1", "p1", "A1"),
                    ("q2", "Q2", "p2", "A2"),
                    ("q3", "Q3", "p3", "A3"),
                ]
            )

            questions = list(iter_questions(limit=3))

            assert questions[0].case_id == "synth-case-0000"
            assert questions[1].case_id == "synth-case-0001"
            assert questions[2].case_id == "synth-case-0002"

    def test_iter_questions_generates_tenant_ids(self) -> None:
        """Test that tenant IDs follow modulo 10 pattern."""
        with patch(
            "eval_harness.stubs.span_generator.loader.load_legal_rag_bench"
        ) as mock_loader:
            mock_loader.return_value = iter(
                [(f"q{i}", f"Q{i}", f"p{i}", f"A{i}") for i in range(12)]
            )

            questions = list(iter_questions(limit=12))

            assert questions[0].tenant_id_hashed == "synth-tenant-0"
            assert questions[9].tenant_id_hashed == "synth-tenant-9"
            assert questions[10].tenant_id_hashed == "synth-tenant-0"  # Cycles
            assert questions[11].tenant_id_hashed == "synth-tenant-1"


class TestSampleQuestions:
    """Test deterministic question sampling."""

    def test_same_seed_produces_identical_sequence(self) -> None:
        """Test that same seed produces identical question sequence."""
        with patch(
            "eval_harness.stubs.span_generator.loader.load_legal_rag_bench"
        ) as mock_loader:
            # Return 20 mock questions
            mock_loader.return_value = iter(
                [(f"q{i}", f"Question {i}", f"p{i}", f"Answer {i}") for i in range(20)]
            )

            questions1 = sample_questions(limit=10, seed=42)
            questions2 = sample_questions(limit=10, seed=42)

            assert len(questions1) == len(questions2)
            for q1, q2 in zip(questions1, questions2, strict=True):
                assert q1.id == q2.id
                assert q1.question == q2.question

    def test_different_seeds_produce_different_sequences(self) -> None:
        """Test that different seeds produce different sequences."""
        with patch(
            "eval_harness.stubs.span_generator.loader.load_legal_rag_bench"
        ) as mock_loader:
            mock_loader.return_value = iter(
                [(f"q{i}", f"Question {i}", f"p{i}", f"Answer {i}") for i in range(20)]
            )

            questions1 = sample_questions(limit=10, seed=42)
            questions2 = sample_questions(limit=10, seed=123)

            # At least some questions should be different
            assert questions1[0].id != questions2[0].id or questions1[1].id != questions2[1].id

    def test_limit_parameter_truncates_correctly(self) -> None:
        """Test that limit parameter truncates correctly."""
        with patch(
            "eval_harness.stubs.span_generator.loader.load_legal_rag_bench"
        ) as mock_loader:
            mock_loader.return_value = iter(
                [(f"q{i}", f"Question {i}", f"p{i}", f"Answer {i}") for i in range(20)]
            )

            questions_5 = sample_questions(limit=5, seed=42)
            questions_10 = sample_questions(limit=10, seed=42)

            assert len(questions_5) == 5
            assert len(questions_10) == 10
            # First 5 should match (same seed)
            for q5, q10 in zip(questions_5, questions_10[:5], strict=True):
                assert q5.id == q10.id

    def test_default_values_match_constants(self) -> None:
        """Test that default values match spec constants."""
        from eval_harness.stubs.span_generator.loader import (
            DEFAULT_DEMO_QUESTIONS,
            DEFAULT_DEMO_SEED,
        )

        assert DEFAULT_DEMO_QUESTIONS == 50
        assert DEFAULT_DEMO_SEED == 42
