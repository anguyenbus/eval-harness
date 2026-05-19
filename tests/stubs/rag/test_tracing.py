"""
Tests for Phoenix tracing integration.

These tests verify the PhoenixTracer class and trace context generation.
"""

from eval_harness.stubs.rag.tracing import PhoenixTracer


class TestPhoenixTracer:
    """Test suite for PhoenixTracer."""

    def test_trace_id_generation(self):
        """Test trace_id generation and format."""
        tracer = PhoenixTracer("http://localhost:6006", "test-project")
        trace_ctx = tracer.start_trace("test query")

        assert "trace_id" in trace_ctx
        assert len(trace_ctx["trace_id"]) == 36  # UUID format

    def test_span_id_generation(self):
        """Test span_id generation and format."""
        tracer = PhoenixTracer("http://localhost:6006", "test-project")
        trace_ctx = tracer.start_trace("test query")

        assert "span_id" in trace_ctx
        assert len(trace_ctx["span_id"]) == 36  # UUID format

    def test_openinference_trace_context_population(self):
        """Test OpenInference trace context is populated."""
        tracer = PhoenixTracer("http://localhost:6006", "test-project")
        trace_ctx = tracer.start_trace("test query")

        assert "trace_id" in trace_ctx
        assert "span_id" in trace_ctx
        assert "project_name" in trace_ctx

    def test_phoenix_project_name_from_config(self):
        """Test phoenix.project_name from config."""
        tracer = PhoenixTracer("http://localhost:6006", "eval-harness")
        trace_ctx = tracer.start_trace("test query")

        assert trace_ctx["project_name"] == "eval-harness"

    def test_trace_fields_in_output_schema(self):
        """Test trace fields conform to output schema structure."""
        tracer = PhoenixTracer("http://localhost:6006", "eval-harness")
        trace_ctx = tracer.start_trace("test query")

        # These are the fields expected by the schema
        assert "trace_id" in trace_ctx
        assert "span_id" in trace_ctx
        assert "project_name" in trace_ctx

    def test_tracing_works_without_phoenix_connection(self):
        """Test tracing works without Phoenix connection (graceful degradation)."""
        tracer = PhoenixTracer("http://invalid:9999", "test-project")

        # Should not raise error even with invalid endpoint
        trace_ctx = tracer.start_trace("test query")
        assert "trace_id" in trace_ctx

        # End trace should not raise error
        tracer.end_trace(trace_ctx["trace_id"], {})

    def test_trace_id_stability_across_calls(self):
        """Test trace_id is unique for each call."""
        tracer = PhoenixTracer("http://localhost:6006", "test-project")

        trace_ctx1 = tracer.start_trace("query1")
        trace_ctx2 = tracer.start_trace("query2")

        # Each call should generate unique IDs
        assert trace_ctx1["trace_id"] != trace_ctx2["trace_id"]
        assert trace_ctx1["span_id"] != trace_ctx2["span_id"]

    def test_error_handling_when_phoenix_unavailable(self):
        """Test error handling when Phoenix is unavailable."""
        tracer = PhoenixTracer("http://invalid:9999", "test-project")

        # Should not raise error
        trace_ctx = tracer.start_trace("test query")
        assert "trace_id" in trace_ctx

        # End trace should handle errors gracefully
        tracer.end_trace(trace_ctx["trace_id"], {})
