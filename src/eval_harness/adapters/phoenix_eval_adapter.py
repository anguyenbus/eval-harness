"""
Phoenix evaluator adapter for RAG evaluation.

This module provides PhoenixEvalAdapter which wraps Phoenix evaluators
with the existing evaluation interface for gradual migration from DeepEval.

PHOENIX NATIVE MIGRATION: Phase 3 - Evaluator Migration
Replaces DeepEval evaluators with Phoenix-native equivalents:
- FaithfulnessEvaluator (replaces DeepEval Faithfulness)
- CorrectnessEvaluator (replaces DeepEval Answer Relevancy)
- DocumentRelevanceEvaluator (replaces DeepEval ContextualPrecision/Recall)
"""

from __future__ import annotations

from typing import Any, Final

import pandas as pd
from beartype import beartype
from beartype.typing import Dict, List

# Constants
DEFAULT_LLM_PROVIDER: Final[str] = "openai"
DEFAULT_JUDGE_MODEL: Final[str] = "gpt-4o-mini"
DEFAULT_TEMPERATURE: Final[float] = 0.0


@beartype
class PhoenixEvalAdapter:
    """
    Phoenix evaluator adapter for RAG metrics.

    Wraps Phoenix evaluators with the existing evaluation interface
    for gradual migration from DeepEval.

    Evaluators:
    - FaithfulnessEvaluator: Detects hallucinations (replaces DeepEval Faithfulness)
    - CorrectnessEvaluator: Measures answer accuracy (replaces DeepEval Answer Relevancy)
    - DocumentRelevanceEvaluator: Evaluates context relevance (replaces DeepEval
      ContextualPrecision/ContextualRecall)

    Attributes:
        _llm: Phoenix LLM wrapper for judge model.
        _llm_provider: LLM provider name (openai, anthropic, etc.).
        _judge_model: Judge model name.
        _temperature: Sampling temperature.
        _faithfulness_evaluator: Phoenix FaithfulnessEvaluator instance.
        _correctness_evaluator: Phoenix CorrectnessEvaluator instance.
        _relevance_evaluator: Phoenix DocumentRelevanceEvaluator instance.

    Example:
        >>> adapter = PhoenixEvalAdapter(judge_model="gpt-4o-mini")
        >>> scores = adapter.compute_metrics(rag_output, reference_answer)
        >>> print(scores["faithfulness"])
        0.85

    """

    __slots__ = (
        "_llm",
        "_llm_provider",
        "_judge_model",
        "_temperature",
        "_faithfulness_evaluator",
        "_correctness_evaluator",
        "_relevance_evaluator",
    )

    def __init__(
        self,
        llm_provider: str = DEFAULT_LLM_PROVIDER,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        """
        Initialize Phoenix evaluator adapter.

        Args:
            llm_provider: LLM provider name (default: "openai").
            judge_model: Judge model name (default: "gpt-4o-mini").
            temperature: Sampling temperature (default: 0.0).

        """
        from phoenix.evals import LLM
        from phoenix.evals.metrics import (
            CorrectnessEvaluator,
            DocumentRelevanceEvaluator,
            FaithfulnessEvaluator,
        )

        self._llm_provider = llm_provider
        self._judge_model = judge_model
        self._temperature = temperature

        # Initialize Phoenix LLM wrapper
        self._llm = LLM(
            provider=llm_provider,
            model=judge_model,
        )

        # Initialize Phoenix evaluators
        self._faithfulness_evaluator = FaithfulnessEvaluator(llm=self._llm)
        self._correctness_evaluator = CorrectnessEvaluator(llm=self._llm)
        self._relevance_evaluator = DocumentRelevanceEvaluator(llm=self._llm)

    @beartype
    def _rag_output_to_dataframe(
        self,
        rag_output: Dict[str, Any],
        reference_answer: str,
    ) -> pd.DataFrame:
        """
        Transform RAG output to Phoenix evaluation DataFrame.

        Creates a DataFrame with the expected columns for Phoenix evaluators.

        Args:
            rag_output: Dictionary conforming to legal_rag_bench schema.
            reference_answer: Reference answer text from dataset.

        Returns:
            DataFrame with columns for evaluation.

        """
        # Extract question and answer
        question = rag_output.get("query", {}).get("text", "")
        actual_answer = rag_output.get("answer", {}).get("text", "")

        # Extract retrieved contexts
        retrieved_chunks = rag_output.get("retrieved_chunks", [])
        retrieval_context = [
            chunk.get("text", "") for chunk in retrieved_chunks if chunk.get("text")
        ]

        # Create DataFrame
        return pd.DataFrame([{
            "input": question,
            "output": actual_answer,
            "reference": reference_answer,
            "retrieved_documents": retrieval_context,
        }])

    @beartype
    def compute_metrics(
        self,
        rag_output: Dict[str, Any],
        reference_answer: str,
    ) -> Dict[str, Any]:
        """
        Compute Phoenix metrics for RAG output.

        Evaluates with all configured Phoenix evaluators:
        - faithfulness: Hallucination detection (0-1)
        - correctness: Answer accuracy (0-1)
        - relevance: Document relevance (0-1)

        Args:
            rag_output: Dictionary conforming to legal_rag_bench schema.
            reference_answer: Reference answer text from dataset.

        Returns:
            Dictionary mapping metric names to scores:
                - faithfulness: float (0-1)
                - correctness: float (0-1)
                - relevance: float (0-1)

        Example:
            >>> scores = adapter.compute_metrics(rag_output, "Reference answer")
            >>> print(f"Faithfulness: {scores['faithfulness']}")

        """
        from phoenix.evals import evaluate_dataframe

        # Transform to DataFrame
        df = self._rag_output_to_dataframe(rag_output, reference_answer)

        # Run Phoenix evaluators
        evaluators = [
            self._faithfulness_evaluator,
            self._correctness_evaluator,
            self._relevance_evaluator,
        ]

        try:
            result_df = evaluate_dataframe(
                dataframe=df,
                evaluators=evaluators,
            )
        except Exception as e:
            import sys

            print(f"[ERROR] Phoenix evaluation failed: {e}", file=sys.stderr)
            # Return default scores on error
            return {
                "faithfulness": 0.0,
                "correctness": 0.0,
                "relevance": 0.0,
            }

        # Extract scores
        scores: Dict[str, Any] = {}

        # Phoenix evaluators return columns with suffix "_score"
        for metric_name in ["faithfulness", "correctness", "relevance"]:
            score_col = f"{metric_name}_score"
            if score_col in result_df.columns:
                scores[metric_name] = float(result_df[score_col].iloc[0])
            else:
                # Default score if column not found
                scores[metric_name] = 0.0

        return scores

    @beartype
    def batch_compute_metrics(
        self,
        rag_outputs: List[Dict[str, Any]],
        reference_answers: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Compute Phoenix metrics for multiple RAG outputs.

        Batch evaluation for efficiency using Phoenix's evaluate_dataframe.

        Args:
            rag_outputs: List of dictionaries conforming to legal_rag_bench schema.
            reference_answers: List of reference answer texts from dataset.

        Returns:
            List of dictionaries mapping metric names to scores,
            maintaining input order.

        Example:
            >>> scores = adapter.batch_compute_metrics(rag_outputs, references)
            >>> print(f"First query faithfulness: {scores[0]['faithfulness']}")

        """
        from phoenix.evals import evaluate_dataframe

        # Transform all outputs to DataFrame
        rows = []
        for rag_output, reference_answer in zip(rag_outputs, reference_answers, strict=True):
            question = rag_output.get("query", {}).get("text", "")
            actual_answer = rag_output.get("answer", {}).get("text", "")

            retrieved_chunks = rag_output.get("retrieved_chunks", [])
            retrieval_context = [
                chunk.get("text", "") for chunk in retrieved_chunks if chunk.get("text")
            ]

            rows.append({
                "input": question,
                "output": actual_answer,
                "reference": reference_answer,
                "retrieved_documents": retrieval_context,
            })

        df = pd.DataFrame(rows)

        # Run Phoenix evaluators
        evaluators = [
            self._faithfulness_evaluator,
            self._correctness_evaluator,
            self._relevance_evaluator,
        ]

        try:
            result_df = evaluate_dataframe(
                dataframe=df,
                evaluators=evaluators,
            )
        except Exception as e:
            import sys

            print(f"[ERROR] Phoenix batch evaluation failed: {e}", file=sys.stderr)
            # Return default scores for all
            return [
                {
                    "faithfulness": 0.0,
                    "correctness": 0.0,
                    "relevance": 0.0,
                }
                for _ in rag_outputs
            ]

        # Extract scores for each row
        scores_list: List[Dict[str, Any]] = []

        for idx in range(len(rag_outputs)):
            row_scores: Dict[str, Any] = {}

            for metric_name in ["faithfulness", "correctness", "relevance"]:
                score_col = f"{metric_name}_score"
                if score_col in result_df.columns:
                    row_scores[metric_name] = float(result_df[score_col].iloc[idx])
                else:
                    row_scores[metric_name] = 0.0

            scores_list.append(row_scores)

        return scores_list

    @beartype
    def compute_metrics_with_reasoning(
        self,
        rag_output: Dict[str, Any],
        reference_answer: str,
    ) -> Dict[str, Any]:
        """
        Compute Phoenix metrics with reasoning extraction.

        Extracts explanations for metric scores.

        Args:
            rag_output: Dictionary conforming to legal_rag_bench schema.
            reference_answer: Reference answer text from dataset.

        Returns:
            Dictionary with:
                - scores: dict of metric name -> float score
                - reasoning: dict of metric name -> dict with explanation

        Example:
            >>> result = adapter.compute_metrics_with_reasoning(rag_output, ref)
            >>> print(f"Faithfulness: {result['scores']['faithfulness']}")
            >>> print(f"Reasoning: {result['reasoning']['faithfulness']}")

        """
        from phoenix.evals import evaluate_dataframe

        # Transform to DataFrame
        df = self._rag_output_to_dataframe(rag_output, reference_answer)

        # Run Phoenix evaluators
        evaluators = [
            self._faithfulness_evaluator,
            self._correctness_evaluator,
            self._relevance_evaluator,
        ]

        try:
            result_df = evaluate_dataframe(
                dataframe=df,
                evaluators=evaluators,
            )
        except Exception as e:
            import sys

            print(f"[ERROR] Phoenix evaluation failed: {e}", file=sys.stderr)
            return {
                "scores": {
                    "faithfulness": 0.0,
                    "correctness": 0.0,
                    "relevance": 0.0,
                },
                "reasoning": {
                    "faithfulness": {"reason": f"ERROR: {e}"},
                    "correctness": {"reason": f"ERROR: {e}"},
                    "relevance": {"reason": f"ERROR: {e}"},
                },
            }

        # Extract scores
        scores: Dict[str, Any] = {}
        reasoning: Dict[str, Any] = {}

        for metric_name in ["faithfulness", "correctness", "relevance"]:
            score_col = f"{metric_name}_score"
            reason_col = f"{metric_name}_reason"  # Phoenix provides reasoning

            if score_col in result_df.columns:
                scores[metric_name] = float(result_df[score_col].iloc[0])
            else:
                scores[metric_name] = 0.0

            if reason_col in result_df.columns:
                reasoning[metric_name] = {
                    "reason": str(result_df[reason_col].iloc[0]),
                }
            else:
                reasoning[metric_name] = {"reason": ""}

        return {"scores": scores, "reasoning": reasoning}
