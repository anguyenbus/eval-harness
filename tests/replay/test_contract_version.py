"""Tests for contract version checking."""


import pytest
import yaml


class TestContractVersionChecking:
    """Test suite for contract version validation."""

    def test_successful_query_with_matching_contract_version(self, tmp_path):
        """Test successful query with matching contract version."""
        from eval_harness.replay.candidate_config import CandidateConfig
        from eval_harness.replay.http_client import _SUPPORTED_CONTRACTS, HTTPClient

        # Verify 1.0 is supported
        assert "1.0" in _SUPPORTED_CONTRACTS

        # Create candidate spec with supported version
        candidate_path = tmp_path / "candidate.yaml"
        candidate_data = {
            "name": "test-candidate",
            "description": "Test candidate",
            "candidate": {
                "service_url": "http://localhost:9999/query",
                "service_version": "1.0.0",
                "contract_version": "1.0",
                "timeout_seconds": 30,
                "max_retries": 0,
            },
        }
        with open(candidate_path, "w") as f:
            yaml.dump(candidate_data, f)

        candidate_config = CandidateConfig.from_yaml_file(candidate_path)

        # Should not raise error for supported version
        HTTPClient(candidate_config, health_check_enabled=False)

    def test_fail_fast_on_major_version_mismatch(self, tmp_path):
        """Test fail-fast on major version mismatch."""
        from eval_harness.replay.candidate_config import CandidateConfig
        from eval_harness.replay.http_client import HTTPClient

        # Create candidate spec with unsupported major version
        candidate_path = tmp_path / "candidate.yaml"
        candidate_data = {
            "name": "test-candidate",
            "description": "Test candidate",
            "candidate": {
                "service_url": "http://localhost:9999/query",
                "service_version": "1.0.0",
                "contract_version": "2.0",  # Major version 2 not supported
                "timeout_seconds": 30,
                "max_retries": 0,
            },
        }
        with open(candidate_path, "w") as f:
            yaml.dump(candidate_data, f)

        candidate_config = CandidateConfig.from_yaml_file(candidate_path)

        # Should raise error for unsupported major version
        with pytest.raises(RuntimeError, match="Unsupported contract version"):
            HTTPClient(candidate_config, health_check_enabled=False)

    def test_warning_logged_for_minor_version_differences(self, tmp_path, caplog):
        """Test warning logged for minor version differences."""
        import logging

        from eval_harness.replay.candidate_config import CandidateConfig
        from eval_harness.replay.http_client import HTTPClient

        # Create candidate spec with minor version ahead of supported
        candidate_path = tmp_path / "candidate.yaml"
        candidate_data = {
            "name": "test-candidate",
            "description": "Test candidate",
            "candidate": {
                "service_url": "http://localhost:9999/query",
                "service_version": "1.0.0",
                "contract_version": "1.2",  # Minor version ahead of 1.0/1.1
                "timeout_seconds": 30,
                "max_retries": 0,
            },
        }
        with open(candidate_path, "w") as f:
            yaml.dump(candidate_data, f)

        candidate_config = CandidateConfig.from_yaml_file(candidate_path)

        # Should log warning but not fail
        with caplog.at_level(logging.WARNING):
            HTTPClient(candidate_config, health_check_enabled=False)

        # Check for warning about minor version
        assert any(
            "contract version" in record.message.lower()
            for record in caplog.records
        )

    def test_supported_contracts_constant(self):
        """Test that _SUPPORTED_CONTRACTS constant is defined."""
        from eval_harness.replay.http_client import _SUPPORTED_CONTRACTS

        # Should be a list/tuple of version strings
        assert isinstance(_SUPPORTED_CONTRACTS, (list, tuple))
        assert len(_SUPPORTED_CONTRACTS) > 0
        assert all(isinstance(v, str) for v in _SUPPORTED_CONTRACTS)
        # Should include 1.0 and 1.1 as specified
        assert "1.0" in _SUPPORTED_CONTRACTS
        assert "1.1" in _SUPPORTED_CONTRACTS
