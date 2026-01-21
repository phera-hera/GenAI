# LlamaIndex Ingestion Pipeline Migration

## Overview

This migration implements a LlamaIndex-based ingestion pipeline as an alternative to the custom ingestion pipeline. The main goal is to fix table extraction issues by using Docling's JSON export instead of markdown, while maintaining compatibility with existing storage and retrieval systems.

## What Was Implemented

### 1. Dependencies (`pyproject.toml`)
- Added `llama-index-readers-docling>=0.1.0` for PDF parsing with JSON export

### 2. Medical Metadata Extractor Wrapper (`src/medical_agent/ingestion/metadata/llamaindex_extractor.py`)
- Wraps existing `MedicalMetadataExtractor` to work with LlamaIndex's `BaseExtractor` interface
- Extracts document-level medical metadata once and stamps it onto all nodes
- Maintains all 8 medical metadata fields (ethnicities, diagnoses, symptoms, etc.)
- Caches metadata extraction per document for efficiency

### 3. PGVector Store Adapter (`src/medical_agent/ingestion/storage/llamaindex_vector_store.py`)
- Adapter between LlamaIndex `PGVectorStore` and existing `paper_chunks` schema
- Provides `SearchResult` compatibility for retrieval system
- Maintains async operations
- Preserves `chunk_metadata` JSONB structure

### 4. LlamaIndex Ingestion Pipeline (`src/medical_agent/ingestion/llamaindex_pipeline.py`)
- Complete pipeline using LlamaIndex components:
  - `DoclingReader` with JSON export (fixes table extraction)
  - `SentenceSplitter` for section-aware chunking (1200 chars, no overlap)
  - `LlamaIndexMedicalMetadataExtractor` for medical metadata
  - `AzureOpenAIEmbedding` for embeddings (3072 dimensions)
- Maintains compatibility with existing `Paper` and `PaperChunk` models
- Provides same result format as original pipeline

### 5. Feature Flag (`src/medical_agent/api/routes/ingestion.py`)
- Added `USE_LLAMAINDEX_PIPELINE` environment variable
- Seamless switching between old and new pipelines
- Set in `.env` file for easy rollback

### 6. Testing (`test_llamaindex_migration.py`)
- Validation script to verify all components work correctly
- Tests imports, instantiation, and configuration
- Run before enabling feature flag

## Architecture Comparison

### Current Pipeline
```
PDF bytes → DoclingPDFParser (markdown)
         → DoclingHierarchicalChunker (custom)
         → MedicalMetadataExtractor (8 medical fields)
         → AsyncAzureEmbedder (custom)
         → VectorStore (custom pgvector)
```

### New LlamaIndex Pipeline
```
PDF bytes → DoclingReader (JSON export, built-in LlamaIndex)
         → SentenceSplitter (LlamaIndex built-in)
         → LlamaIndexMedicalMetadataExtractor (wraps existing extractor)
         → AzureOpenAIEmbedding (LlamaIndex built-in)
         → MedicalPGVectorStore (adapter for existing schema)
```

## Key Improvements

1. **Better Table Extraction**: JSON export preserves table structure instead of flattening to markdown
2. **Standard Components**: Uses battle-tested LlamaIndex components
3. **Simpler Maintenance**: Less custom code to maintain
4. **Same Schema**: Works with existing database schema
5. **Same Results**: Compatible with existing retrieval system

## Files Changed

### Created (4 files)
1. `src/medical_agent/ingestion/metadata/llamaindex_extractor.py` - Metadata wrapper
2. `src/medical_agent/ingestion/storage/llamaindex_vector_store.py` - Vector store adapter
3. `src/medical_agent/ingestion/llamaindex_pipeline.py` - Main pipeline
4. `test_llamaindex_migration.py` - Validation script

### Modified (3 files)
1. `pyproject.toml` - Added llama-index-readers-docling dependency
2. `src/medical_agent/api/routes/ingestion.py` - Added feature flag
3. `.env.example` - Documented USE_LLAMAINDEX_PIPELINE flag

### Unchanged
- All existing ingestion components (kept for rollback)
- Database schema (`papers`, `paper_chunks` tables)
- Retrieval system (`MedicalPaperRetriever`)
- All other API endpoints

## How to Use

### Step 1: Install Dependencies

```bash
# Using uv (recommended)
uv pip install llama-index-readers-docling

# Or using pip
pip install llama-index-readers-docling
```

### Step 2: Validate Installation

Run the test script to ensure everything is correctly configured:

```bash
python3 test_llamaindex_migration.py
```

All tests should pass:
- ✓ Imports
- ✓ Instantiation
- ✓ Configuration

### Step 3: Enable Feature Flag

Edit your `.env` file:

```bash
# Enable LlamaIndex pipeline
USE_LLAMAINDEX_PIPELINE=true
```

### Step 4: Restart Application

```bash
# Restart your FastAPI application
# The new pipeline will be used for all ingestion requests
```

### Step 5: Test Ingestion

Test with a sample PDF to verify table extraction:

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "gcp_paths": ["sample_paper.pdf"]
  }'
```

### Step 6: Verify Results

Check that:
1. Ingestion succeeds
2. Tables are properly extracted (check chunks with `chunk_type="table"`)
3. Medical metadata is present in `chunk_metadata.extracted_metadata`
4. Retrieval works correctly

## Rollback

If you encounter issues, you can instantly rollback:

```bash
# Disable LlamaIndex pipeline in .env
USE_LLAMAINDEX_PIPELINE=false

# Restart application
# Old pipeline will be used
```

## Configuration

The LlamaIndex pipeline uses these settings from your `.env`:

```bash
# Database
POSTGRES_* variables (same as before)

# Azure OpenAI (for embeddings)
AZURE_OPENAI_EMBEDDING_API_KEY
AZURE_OPENAI_EMBEDDING_ENDPOINT
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME

# Or falls back to main credentials:
AZURE_OPENAI_API_KEY
AZURE_OPENAI_ENDPOINT

# GCP Storage (same as before)
GCP_BUCKET_NAME
GCP_PROJECT_ID
```

## Differences from Original Pipeline

### Same
- Chunk size: 1200 characters
- Chunk overlap: 0 (section boundaries respected)
- Medical metadata: All 8 fields extracted
- Embeddings: text-embedding-3-large (3072 dimensions)
- Database schema: No changes
- Retrieval: Fully compatible

### Different
- Parser: DoclingReader (JSON) instead of DoclingPDFParser (markdown)
- Chunker: SentenceSplitter instead of DoclingHierarchicalChunker
- Embedding client: LlamaIndex AzureOpenAIEmbedding instead of custom AsyncAzureEmbedder
- Storage: Adapter wrapping LlamaIndex PGVectorStore

### Better
- Table extraction: Structured JSON instead of flattened markdown
- Maintenance: Standard LlamaIndex components
- Testing: More predictable behavior

## Monitoring

After enabling the feature flag, monitor:

1. **Ingestion Success Rate**: Should be similar to before
2. **Table Extraction Quality**: Should be significantly better
3. **Chunk Counts**: May differ slightly due to different chunker
4. **Medical Metadata**: Should be same as before
5. **Retrieval Quality**: Should be same or better
6. **Performance**: May differ slightly

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'llama_index.readers.docling'`:

```bash
# Install the missing package
uv pip install llama-index-readers-docling
```

### Configuration Errors

If embeddings fail, check:

```bash
# Verify Azure OpenAI credentials
echo $AZURE_OPENAI_EMBEDDING_API_KEY
echo $AZURE_OPENAI_EMBEDDING_ENDPOINT

# Or check main credentials as fallback
echo $AZURE_OPENAI_API_KEY
echo $AZURE_OPENAI_ENDPOINT
```

### Storage Errors

If storage fails, verify:

```bash
# Test database connection
python3 -c "from medical_agent.core.config import settings; print(settings.database_connection_string)"
```

### Feature Flag Not Working

If the old pipeline is still being used:

1. Check `.env` file has `USE_LLAMAINDEX_PIPELINE=true`
2. Restart the application
3. Check logs for "Using LlamaIndex ingestion pipeline"

## Next Steps

1. **Monitor Production**: Track metrics after enabling
2. **Compare Results**: Compare table extraction quality
3. **Optimize Performance**: Tune chunk size if needed
4. **Remove Old Code**: After 2-4 weeks of stability, remove old pipeline (optional)

## Support

For issues or questions:
1. Check logs for detailed error messages
2. Run `test_llamaindex_migration.py` to validate setup
3. Review this documentation
4. Check LlamaIndex documentation: https://docs.llamaindex.ai/

## Credits

- LlamaIndex: https://github.com/run-llama/llama_index
- Docling: https://github.com/DS4SD/docling
- Azure OpenAI: https://azure.microsoft.com/en-us/products/ai-services/openai-service
