# Quick Start: Metadata-Based Filtering

## ✅ What We Just Implemented

You can now filter retrieved papers by **medical metadata**:
- ✅ Ethnicities (12 options)
- ✅ Diagnoses (11 hormone-related conditions)
- ✅ Symptoms (discharge, odor, pain, etc.)
- ✅ Menstrual status
- ✅ Birth control types
- ✅ Hormone therapies
- ✅ Fertility treatments

## 🚀 How to Use It

### 1. Basic Usage

```python
from medical_agent.rag.retriever import RetrievalConfig, create_retriever
from llama_index.core.schema import QueryBundle

# Create retriever with filters
config = RetrievalConfig(
    top_k=10,
    filter_diagnoses=["Polycystic ovary syndrome (PCOS)"],
    filter_ethnicities=["African / Black"],
)

retriever = create_retriever(config=config)

# Query - returns ONLY papers that mention PCOS AND African/Black populations
results = await retriever._aretrieve(QueryBundle(query_str="vaginal pH levels"))
```

### 2. In Your Agent

```python
# In your LangGraph agent node
def retriever_node(state: AgentState):
    """Agent decides what metadata to filter by."""

    user_profile = state["health_profile"]
    query = state["query"]

    # Build filters dynamically from user context
    config = RetrievalConfig(
        top_k=10,
        # Filter by user's ethnicity
        filter_ethnicities=[user_profile.get("ethnicity")]
        if user_profile.get("ethnicity") else None,

        # Filter by user's diagnoses
        filter_diagnoses=user_profile.get("diagnoses", []),

        # Filter by user's birth control
        filter_birth_control=[user_profile.get("birth_control")]
        if user_profile.get("birth_control") else None,
    )

    retriever = create_retriever(config=config)
    results = await retriever._aretrieve(QueryBundle(query_str=query))

    return {"retrieved_chunks": results}
```

### 3. Available Filters

```python
config = RetrievalConfig(
    top_k=10,

    # Ethnicity (pick from 12 options)
    filter_ethnicities=["African / Black", "Asian", "Hispanic / Latina"],

    # Diagnoses (11 hormone-related conditions)
    filter_diagnoses=["Polycystic ovary syndrome (PCOS)", "Bacterial vaginosis"],

    # Symptoms
    filter_symptoms=["Vaginal Odor", "Gray", "Itchy"],

    # Menstrual status
    filter_menstrual_status=["Premenstrual", "Ovulation"],

    # Birth control
    filter_birth_control=["IUD", "Pill"],

    # Hormone therapy
    filter_hormone_therapy=["HRT", "Estrogen"],

    # Fertility treatments
    filter_fertility_treatments=["IVF", "Clomiphene"],
)
```

## 📊 What Happens Under the Hood

### SQL Query Generated

When you use:
```python
config = RetrievalConfig(
    filter_diagnoses=["PCOS"],
    filter_ethnicities=["African / Black"],
)
```

PostgreSQL executes:
```sql
SELECT * FROM paper_chunks
WHERE
    -- Semantic similarity
    embedding <=> '[0.1, 0.2, ...]' < 0.5

    -- Metadata filters
    AND chunk_metadata->'extracted_metadata'->'diagnoses' ?| array['Polycystic ovary syndrome (PCOS)']
    AND chunk_metadata->'extracted_metadata'->'ethnicities' ?| array['African / Black']

ORDER BY embedding <=> '[0.1, 0.2, ...]'
LIMIT 10;
```

The `?|` operator checks if **ANY** of the values exist in the JSONB array.

### Filter Logic

- **Multiple categories** = AND logic (all must match)
- **Values within a category** = OR logic (any can match)

```python
# AND across categories
filter_diagnoses=["PCOS"] AND filter_symptoms=["Odor"]
# = Chunks that have PCOS AND Odor

# OR within category
filter_ethnicities=["African / Black", "Asian"]
# = Chunks that have EITHER ethnicity
```

## 🧪 Testing

Run the example script:

```bash
cd /path/to/Medical_Agent
python examples/metadata_filtering_example.py
```

This runs 5 examples:
1. Basic filtering by diagnosis
2. Multi-category filtering
3. Hormone therapy context
4. Personalized user profile
5. Comparison with/without filters

## 📝 Normalized Values

**IMPORTANT:** Filters must use normalized values from the dropdown options.

### Ethnicity Options (12)
```python
["African / Black", "Asian", "Caucasian", "Hispanic / Latina",
 "Middle Eastern", "Mixed", "Native American / Indigenous",
 "North African", "Pacific Islander", "South Asian", "Southeast Asian"]
```

### Diagnosis Options (11)
```python
["Adenomyosis", "Endometriosis", "Bacterial vaginosis",
 "Yeast infection", "Sexually transmitted infection",
 "Polycystic ovary syndrome (PCOS)", "Premature ovarian insufficiency",
 "Thyroid disorder", "Fibroids (uterine myomas)",
 "Ovarian cysts", "Pelvic inflammatory disease"]
```

### Symptom Options (Common)
```python
# Discharge
["Creamy", "Clear", "Yellow", "Green", "Gray", "Pink", "Brown"]

# Other symptoms
["Vaginal Odor", "Itchy", "Burning", "Pelvic Pain",
 "Swelling", "Redness", "Vaginal Dryness",
 "Frequent Urination", "Painful Urination"]
```

### Birth Control Options
```python
["Pill", "IUD", "Implant", "Patch", "Ring",
 "Injection", "Condom", "Diaphragm", "Sterilization"]
```

### Hormone Therapy Options
```python
["HRT", "Testosterone", "Progesterone", "Estrogen", "Thyroid Medication"]
```

See `METADATA_FILTERING_GUIDE.md` for complete list.

## ⚡ Performance Tips

### 1. Use Specific Filters

```python
# ✅ Good - Specific filters
config = RetrievalConfig(
    top_k=10,
    filter_diagnoses=["Bacterial vaginosis"],
    filter_symptoms=["Vaginal Odor", "Gray"],
)

# ❌ Bad - Too broad or no filters
config = RetrievalConfig(top_k=10)  # Returns anything
```

### 2. Combine with Chunk Type Filters

```python
# Get only abstracts and results from papers about PCOS
config = RetrievalConfig(
    top_k=10,
    chunk_types=[ChunkType.ABSTRACT, ChunkType.RESULTS],
    filter_diagnoses=["Polycystic ovary syndrome (PCOS)"],
)
```

### 3. Use Hybrid Search (Default)

```python
# Hybrid search (semantic + BM25) with metadata filters
config = RetrievalConfig(
    top_k=10,
    search_mode=SearchMode.HYBRID,  # Default
    filter_ethnicities=["Hispanic / Latina"],
)
```

## 🔍 Debugging

### Check if Metadata Exists

```python
# Retrieve without filters first
config = RetrievalConfig(top_k=5)
retriever = create_retriever(config=config)
results = await retriever._aretrieve(QueryBundle(query_str="PCOS"))

# Check metadata
for result in results:
    metadata = result.node.metadata.get("chunk_metadata", {})
    extracted = metadata.get("extracted_metadata", {})

    print(f"Diagnoses: {extracted.get('diagnoses', [])}")
    print(f"Ethnicities: {extracted.get('ethnicities', [])}")
    print(f"Symptoms: {extracted.get('symptoms', [])}")
```

### Enable Debug Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("medical_agent.ingestion.storage.vector_store")
logger.setLevel(logging.DEBUG)

# Now you'll see:
# DEBUG: Applied metadata filters: diagnoses=['PCOS'], ethnicities=['African / Black']
```

## 🎯 Next Steps

1. **Ingest papers** with new pipeline (see main README)
2. **Run examples** (`python examples/metadata_filtering_example.py`)
3. **Integrate into your agent** (see `ARCHITECTURE_EXPLAINED.md`)
4. **Test with real queries** from your users

## 📚 Related Documentation

- `METADATA_FILTERING_GUIDE.md` - Complete implementation guide
- `ARCHITECTURE_EXPLAINED.md` - How LlamaIndex fits in
- `examples/metadata_filtering_example.py` - Working examples

## ❓ FAQ

**Q: What if a paper has no metadata for a category?**
A: Papers with empty arrays for a filtered category will be excluded. Only papers with explicit mentions are returned.

**Q: Can I filter by multiple diagnoses?**
A: Yes! `filter_diagnoses=["PCOS", "Endometriosis"]` returns papers mentioning EITHER diagnosis (OR logic).

**Q: Can I filter by ethnicity AND diagnosis?**
A: Yes! Filters across categories use AND logic. The paper must match ALL filters.

**Q: What if I use a non-normalized value?**
A: The filter won't match anything. Always use the exact normalized dropdown values.

**Q: Does this slow down queries?**
A: Minimal impact. PostgreSQL JSONB queries with GIN indexes are very fast.

## 🎉 Benefits

✅ **Personalized retrieval** based on user health context
✅ **Higher precision** - no irrelevant papers
✅ **Better citations** - ethnicity/diagnosis-specific research
✅ **Agent intelligence** - dynamically adjust filters
✅ **Scalable** - works with thousands of papers
