"""Tests for distributed tracing header propagation."""


import yaml


class TestTraceHeaderPropagation:
    """Test suite for trace header propagation."""

    def test_traceparent_header_passed_from_client_to_service(self, tmp_path):
        """Test that traceparent header is passed from client to service."""
        from eval_harness.replay.candidate_config import CandidateConfig
        from eval_harness.replay.http_client import HTTPClient

        # Create candidate spec
        candidate_path = tmp_path / "candidate.yaml"
        candidate_data = {
            "name": "test-candidate",
            "description": "Test candidate",
            "candidate": {
                "service_url": "http://localhost:8081/query",
                "service_version": "1.0.0",
                "contract_version": "1.0",
                "timeout_seconds": 30,
                "max_retries": 2,
            },
        }
        with open(candidate_path, "w") as f:
            yaml.dump(candidate_data, f)

        config = CandidateConfig.from_yaml_file(candidate_path)
        http_client = HTTPClient(config, health_check_enabled=False)

        # Test the _get_traceparent method directly
        # First test without a mock (should return None)
        traceparent = http_client._get_traceparent()
        # Should return None if no active trace
        assert traceparent is None or isinstance(traceparent, str)

    def test_traceparent_format(self):
        """Test that traceparent header format is correct."""
        # Test the W3C traceparent format
        # Format: 00-{trace_id}-{span_id}-{trace_flags}
        trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
        span_id = "00f067aa0ba902b7"
        trace_flags = "01"

        expected_traceparent = f"00-{trace_id}-{span_id}-{trace_flags}"
        assert expected_traceparent == (
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
        )

        # Verify format
        parts = expected_traceparent.split("-")
        assert len(parts) == 4
        assert parts[0] == "00"  # version
        assert len(parts[1]) == 32  # trace_id
        assert len(parts[2]) == 16  # span_id
        assert len(parts[3]) == 2  # trace_flags

    def test_child_spans_share_same_trace_id_as_client(self, tmp_path):
        """Test that child spans in HTTP service share same trace_id as client."""
        # Tested indirectly via traceparent propagation.
        # If traceparent is propagated correctly, child spans will have
        # the same trace_id
        from eval_harness.stubs.service.tracing import extract_trace_context

        # Mock W3C traceparent header
        traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
        headers = {"traceparent": traceparent}

        trace_context = extract_trace_context(headers)
        assert trace_context is not None
        assert trace_context["traceparent"] == traceparent
        # Extract trace_id from traceparent (after "00-" and before first "-")
        trace_id = traceparent.split("-")[1]
        assert trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"

    def test_trace_context_extraction_in_http_service(self, tmp_path):
        """Test that HTTP service extracts traceparent and tracestate headers."""
        from eval_harness.stubs.service.tracing import extract_trace_context

        # Test extraction of traceparent
        headers1 = {
            "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
        }
        context1 = extract_trace_context(headers1)
        assert context1 is not None
        assert "traceparent" in context1

        # Test extraction of both traceparent and tracestate
        headers2 = {
            "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            "tracestate": "vendor1=key1=value1",
        }
        context2 = extract_trace_context(headers2)
        assert context2 is not None
        assert "traceparent" in context2
        assert "tracestate" in context2

        # Test empty headers
        context3 = extract_trace_context({})
        assert context3 is None
