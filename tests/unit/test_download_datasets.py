"""Tests for dataset download scripts."""

from scripts.download_datasets import get_manifest, update_manifest, verify_hash


class TestDatasetDownload:
    """Test suite for dataset download functionality."""

    def test_manifest_creation(self, tmp_path):
        """Test that manifest file is created with correct structure."""
        manifest_path = tmp_path / "MANIFEST.yaml"

        update_manifest(manifest_path, "test_dataset", "v1.0", "abc123")

        assert manifest_path.exists()

        import yaml

        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)

        assert "test_dataset" in manifest
        assert manifest["test_dataset"]["version"] == "v1.0"
        assert manifest["test_dataset"]["sha256"] == "abc123"

    def test_hash_verification(self, tmp_path):
        """Test that file hash verification works correctly."""
        # Create test file with known content
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        # SHA-256 of "hello world" is known
        known_hash = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"

        result = verify_hash(test_file, known_hash)
        assert result is True

    def test_hash_verification_fails_on_mismatch(self, tmp_path):
        """Test that hash verification fails for wrong hash."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("different content")

        result = verify_hash(test_file, "wronghash")
        assert result is False

    def test_get_manifest_returns_none_if_missing(self, tmp_path):
        """Test that get_manifest returns None if manifest doesn't exist."""
        manifest_path = tmp_path / "nonexistent.yaml"

        manifest = get_manifest(manifest_path)
        assert manifest is None

    def test_get_manifest_loads_existing(self, tmp_path):
        """Test that get_manifest loads existing manifest."""
        manifest_path = tmp_path / "MANIFEST.yaml"

        import yaml

        test_data = {"omnidocbench": {"version": "v1.0", "sha256": "abc123"}}
        with open(manifest_path, "w") as f:
            yaml.dump(test_data, f)

        manifest = get_manifest(manifest_path)
        assert manifest == test_data
