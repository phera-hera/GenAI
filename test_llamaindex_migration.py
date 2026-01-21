"""
Test Script for LlamaIndex Migration

This script verifies that all LlamaIndex migration components can be imported
and are correctly configured.

Run this script to validate the migration before enabling the feature flag.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def test_imports():
    """Test that all new components can be imported."""
    print("Testing imports...")

    try:
        # Test metadata extractor
        from medical_agent.ingestion.metadata.llamaindex_extractor import (
            LlamaIndexMedicalMetadataExtractor,
        )
        print("✓ LlamaIndexMedicalMetadataExtractor imported successfully")

        # Test vector store adapter
        from medical_agent.ingestion.storage.llamaindex_vector_store import (
            MedicalPGVectorStore,
        )
        print("✓ MedicalPGVectorStore imported successfully")

        # Test pipeline
        from medical_agent.ingestion.llamaindex_pipeline import (
            LlamaIndexIngestionPipeline,
            LlamaIndexPipelineConfig,
        )
        print("✓ LlamaIndexIngestionPipeline imported successfully")

        # Test API routes with feature flag
        import os
        os.environ["USE_LLAMAINDEX_PIPELINE"] = "true"

        # Reload the module to pick up the env var
        import importlib
        import medical_agent.api.routes.ingestion as ingestion_module
        importlib.reload(ingestion_module)

        print("✓ API routes imported successfully with feature flag enabled")

        return True

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_instantiation():
    """Test that components can be instantiated."""
    print("\nTesting component instantiation...")

    try:
        from medical_agent.ingestion.metadata.llamaindex_extractor import (
            LlamaIndexMedicalMetadataExtractor,
        )
        from medical_agent.ingestion.storage.llamaindex_vector_store import (
            MedicalPGVectorStore,
        )
        from medical_agent.ingestion.llamaindex_pipeline import (
            LlamaIndexIngestionPipeline,
            LlamaIndexPipelineConfig,
        )

        # Test metadata extractor
        extractor = LlamaIndexMedicalMetadataExtractor()
        print("✓ LlamaIndexMedicalMetadataExtractor instantiated")

        # Test vector store
        vector_store = MedicalPGVectorStore()
        print("✓ MedicalPGVectorStore instantiated")

        # Test pipeline config
        config = LlamaIndexPipelineConfig()
        print("✓ LlamaIndexPipelineConfig instantiated")

        # Note: We can't fully instantiate the pipeline without Azure credentials
        # but we can verify the class exists
        print("✓ LlamaIndexIngestionPipeline class available")

        return True

    except Exception as e:
        print(f"✗ Instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration():
    """Test configuration validation."""
    print("\nTesting configuration...")

    try:
        from medical_agent.core.config import settings

        # Check database configuration
        if settings.database_connection_string:
            print("✓ Database connection string configured")
        else:
            print("⚠ Database connection string not configured")

        # Check Azure OpenAI configuration
        if settings.is_azure_openai_configured():
            print("✓ Azure OpenAI configured")
        else:
            print("⚠ Azure OpenAI not configured (required for embeddings)")

        # Check embedding configuration
        if settings.is_azure_openai_embedding_configured():
            print("✓ Azure OpenAI embeddings configured")
        else:
            print("⚠ Azure OpenAI embeddings not configured")

        return True

    except Exception as e:
        print(f"✗ Configuration check failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("LlamaIndex Migration Validation")
    print("=" * 70)

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Instantiation", test_instantiation()))
    results.append(("Configuration", test_configuration()))

    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n✓ All validation checks passed!")
        print("\nNext steps:")
        print("1. Set USE_LLAMAINDEX_PIPELINE=true in your .env file")
        print("2. Restart your application")
        print("3. Test ingestion with a sample PDF")
        print("4. Verify table extraction quality")
        return 0
    else:
        print("\n✗ Some validation checks failed")
        print("Please fix the issues before enabling the feature flag")
        return 1


if __name__ == "__main__":
    sys.exit(main())
