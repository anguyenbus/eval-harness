"""Tests for observability module optional dependency import pattern."""

from __future__ import annotations

import sys

import pytest


class TestOptionalDependencyImport:
    """Test optional dependency import pattern for Phoenix."""

    def test_module_exists_and_has_all(self):
        """Test that observability module exists and has __all__ defined."""
        import eval_harness.observability as obs_module

        assert hasattr(obs_module, "__all__")
        assert isinstance(obs_module.__all__, list)

    def test_phoenix_adapter_not_available_without_dependency(self):
        """Test that PhoenixAdapter is not available when arize-phoenix is not installed."""
        # Clear the module if it was already imported
        if "eval_harness.observability" in sys.modules:
            del sys.modules["eval_harness.observability"]

        # Remove Phoenix-related modules from sys.modules to simulate not installed
        for key in list(sys.modules.keys()):
            if "arize" in key or "phoenix" in key or "openinference" in key:
                del sys.modules[key]

        # Now import observability module
        import eval_harness.observability as obs_module

        # PhoenixAdapter should NOT be in __all__ when Phoenix is not installed
        assert "PhoenixAdapter" not in obs_module.__all__

        # Trying to import PhoenixAdapter should raise ImportError
        with pytest.raises(ImportError, match="cannot import name"):
            from eval_harness.observability import PhoenixAdapter  # noqa: F401

    @pytest.mark.skipif(sys.version_info < (3, 13), reason="Requires Python 3.13+")
    def test_phoenix_adapter_class_structure(self):
        """Test PhoenixAdapter class structure when Phoenix is available."""
        # This test will only pass if Phoenix is installed
        try:
            from eval_harness.observability import PhoenixAdapter
        except ImportError:
            pytest.skip("Phoenix not installed")

        # Check class has required methods
        assert hasattr(PhoenixAdapter, "start_rag_query_span")
        assert hasattr(PhoenixAdapter, "start_retrieval_span")
        assert hasattr(PhoenixAdapter, "start_generation_span")
        assert hasattr(PhoenixAdapter, "start_evaluation_span")
        assert hasattr(PhoenixAdapter, "export_traces")
        assert hasattr(PhoenixAdapter, "is_connected")

        # Check class uses __slots__
        assert hasattr(PhoenixAdapter, "__slots__")

    def test_module_exports_list_type(self):
        """Test that __all__ is a list of strings."""
        import eval_harness.observability as obs_module

        assert isinstance(obs_module.__all__, list)
        for item in obs_module.__all__:
            assert isinstance(item, str)
