"""
Example: Using Metadata-Based Filtering for Personalized Retrieval

This script demonstrates how to use the new metadata filtering capabilities
to retrieve papers based on medical context (ethnicity, diagnoses, symptoms, etc.)

Run this after ingesting some papers with the new pipeline.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llama_index.core.schema import QueryBundle

from medical_agent.rag.retriever import RetrievalConfig, create_retriever


async def example_1_basic_filtering():
    """Example 1: Basic metadata filtering by diagnosis."""
    print("\n" + "=" * 80)
    print("Example 1: Filter by Diagnosis (PCOS)")
    print("=" * 80)

    # Create retriever with diagnosis filter
    config = RetrievalConfig(
        top_k=5,
        filter_diagnoses=["Polycystic ovary syndrome (PCOS)"],
    )

    retriever = create_retriever(config=config)

    # Query
    query = "What are the effects on vaginal pH?"
    results = await retriever._aretrieve(QueryBundle(query_str=query))

    print(f"\nQuery: {query}")
    print(f"Filter: Diagnosis = PCOS")
    print(f"Results: {len(results)} chunks\n")

    for i, result in enumerate(results, 1):
        metadata = result.node.metadata.get("chunk_metadata", {})
        extracted = metadata.get("extracted_metadata", {})

        print(f"{i}. Score: {result.score:.4f}")
        print(f"   Diagnoses: {extracted.get('diagnoses', [])}")
        print(f"   Content: {result.node.text[:150]}...")
        print()


async def example_2_multi_category_filtering():
    """Example 2: Filter by multiple categories (ethnicity + symptoms)."""
    print("\n" + "=" * 80)
    print("Example 2: Filter by Ethnicity + Symptoms")
    print("=" * 80)

    # Create retriever with multiple filters
    config = RetrievalConfig(
        top_k=5,
        filter_ethnicities=["African / Black", "Hispanic / Latina"],
        filter_symptoms=["Vaginal Odor", "Gray"],
    )

    retriever = create_retriever(config=config)

    # Query
    query = "bacterial vaginosis treatment outcomes"
    results = await retriever._aretrieve(QueryBundle(query_str=query))

    print(f"\nQuery: {query}")
    print(f"Filters: Ethnicity = [African/Black, Hispanic/Latina], Symptoms = [Vaginal Odor, Gray]")
    print(f"Results: {len(results)} chunks\n")

    for i, result in enumerate(results, 1):
        metadata = result.node.metadata.get("chunk_metadata", {})
        extracted = metadata.get("extracted_metadata", {})

        print(f"{i}. Score: {result.score:.4f}")
        print(f"   Ethnicities: {extracted.get('ethnicities', [])}")
        print(f"   Symptoms: {extracted.get('symptoms', [])}")
        print(f"   Paper: {result.node.metadata.get('paper_title', 'Unknown')[:60]}...")
        print()


async def example_3_hormone_context():
    """Example 3: Filter by hormone therapy and birth control."""
    print("\n" + "=" * 80)
    print("Example 3: Filter by Hormone Therapy + Birth Control")
    print("=" * 80)

    # Create retriever for hormone-related queries
    config = RetrievalConfig(
        top_k=5,
        filter_hormone_therapy=["HRT", "Estrogen"],
        filter_birth_control=["IUD"],
        filter_symptoms=["Vaginal Dryness"],
    )

    retriever = create_retriever(config=config)

    # Query
    query = "hormone therapy effects on vaginal health"
    results = await retriever._aretrieve(QueryBundle(query_str=query))

    print(f"\nQuery: {query}")
    print(f"Filters: Hormone Therapy = [HRT, Estrogen], Birth Control = [IUD], Symptoms = [Vaginal Dryness]")
    print(f"Results: {len(results)} chunks\n")

    for i, result in enumerate(results, 1):
        metadata = result.node.metadata.get("chunk_metadata", {})
        extracted = metadata.get("extracted_metadata", {})

        print(f"{i}. Score: {result.score:.4f}")
        print(f"   Hormone Therapy: {extracted.get('hormone_therapy', [])}")
        print(f"   Birth Control: {extracted.get('birth_control', [])}")
        print(f"   Symptoms: {extracted.get('symptoms', [])}")
        print()


async def example_4_personalized_user_context():
    """Example 4: Simulating personalized retrieval based on user profile."""
    print("\n" + "=" * 80)
    print("Example 4: Personalized Retrieval (User Profile)")
    print("=" * 80)

    # Simulate user health profile
    user_profile = {
        "ethnicity": "Asian",
        "diagnoses": ["Polycystic ovary syndrome (PCOS)"],
        "birth_control": "Pill",
        "age": 28,
    }

    # User query
    user_query = "My pH is 5.2. What could this mean?"

    print(f"\nUser Profile:")
    print(f"  Ethnicity: {user_profile['ethnicity']}")
    print(f"  Diagnoses: {user_profile['diagnoses']}")
    print(f"  Birth Control: {user_profile['birth_control']}")
    print(f"  Age: {user_profile['age']}")
    print(f"\nUser Query: {user_query}\n")

    # Build filters from user profile
    config = RetrievalConfig(
        top_k=5,
        filter_ethnicities=[user_profile["ethnicity"]],
        filter_diagnoses=user_profile["diagnoses"],
        filter_birth_control=[user_profile["birth_control"]],
    )

    retriever = create_retriever(config=config)
    results = await retriever._aretrieve(QueryBundle(query_str=user_query))

    print(f"Retrieved {len(results)} personalized results:\n")

    for i, result in enumerate(results, 1):
        metadata = result.node.metadata.get("chunk_metadata", {})
        extracted = metadata.get("extracted_metadata", {})

        print(f"{i}. Score: {result.score:.4f}")
        print(f"   Paper: {result.node.metadata.get('paper_title', 'Unknown')[:60]}...")
        print(f"   Ethnicities: {extracted.get('ethnicities', [])}")
        print(f"   Diagnoses: {extracted.get('diagnoses', [])}")
        print(f"   Birth Control: {extracted.get('birth_control', [])}")
        print(f"   Excerpt: {result.node.text[:120]}...")
        print()


async def example_5_no_filters_comparison():
    """Example 5: Compare results with and without filters."""
    print("\n" + "=" * 80)
    print("Example 5: Comparison - With vs Without Filters")
    print("=" * 80)

    query = "vaginal pH and bacterial vaginosis"

    # Without filters
    print("\n--- WITHOUT FILTERS ---")
    config_no_filter = RetrievalConfig(top_k=3)
    retriever_no_filter = create_retriever(config=config_no_filter)
    results_no_filter = await retriever_no_filter._aretrieve(QueryBundle(query_str=query))

    print(f"Query: {query}")
    print(f"Results: {len(results_no_filter)} chunks\n")
    for i, result in enumerate(results_no_filter, 1):
        print(f"{i}. Score: {result.score:.4f}")
        print(f"   Paper: {result.node.metadata.get('paper_title', 'Unknown')[:60]}...")
        print()

    # With filters
    print("\n--- WITH FILTERS (Diagnosis: Bacterial vaginosis, Ethnicity: African/Black) ---")
    config_with_filter = RetrievalConfig(
        top_k=3,
        filter_diagnoses=["Bacterial vaginosis"],
        filter_ethnicities=["African / Black"],
    )
    retriever_with_filter = create_retriever(config=config_with_filter)
    results_with_filter = await retriever_with_filter._aretrieve(QueryBundle(query_str=query))

    print(f"Query: {query}")
    print(f"Results: {len(results_with_filter)} chunks\n")
    for i, result in enumerate(results_with_filter, 1):
        metadata = result.node.metadata.get("chunk_metadata", {})
        extracted = metadata.get("extracted_metadata", {})

        print(f"{i}. Score: {result.score:.4f}")
        print(f"   Paper: {result.node.metadata.get('paper_title', 'Unknown')[:60]}...")
        print(f"   Diagnoses: {extracted.get('diagnoses', [])}")
        print(f"   Ethnicities: {extracted.get('ethnicities', [])}")
        print()


async def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("METADATA-BASED FILTERING EXAMPLES")
    print("=" * 80)
    print("\nThese examples demonstrate how to use medical metadata filters")
    print("to retrieve papers based on ethnicity, diagnoses, symptoms, etc.")
    print("\nNOTE: You need to have papers ingested with the new pipeline first!")
    print("=" * 80)

    try:
        await example_1_basic_filtering()
        await example_2_multi_category_filtering()
        await example_3_hormone_context()
        await example_4_personalized_user_context()
        await example_5_no_filters_comparison()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Ingested papers using the new pipeline (with metadata extraction)")
        print("2. Set up your database connection properly")
        print("3. Configured Azure OpenAI credentials")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
