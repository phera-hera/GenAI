
⏺ Your RAGAS Scores - Detailed Breakdown

  Overall Assessment: 🟡 Mixed Performance - Critical Issue in Factual Correctness

  ---
  1. Faithfulness: 0.8333 🟢 GOOD

  What This Means:

  - 83% of your LLM's claims can be traced back to the retrieved medical papers
  - Your closed-book RAG prompt is mostly working ✅

  Why This Score:

  HIGH because:
  - Your strict prompt enforcement works:
  "Answer ONLY using information from the provided medical documents"
  "DO NOT use external knowledge"
  - temperature=0.0 reduces creative hallucination
  - Structured output (MedicalResponse Pydantic model) keeps LLM focused

  Not perfect (17% leakage) because:
  - LLM occasionally adds general medical knowledge not in the docs
  - similarity_top_k=2 might be too low - LLM fills gaps with parametric knowledge

  Action:

  - ✅ Keep current prompt
  - Consider adding: "If information is not in documents, explicitly say 'Not mentioned in available research'"

  ---
  2. Context Recall: 0.6857 🟡 MODERATE

  What This Means:

  - Your retriever is only finding 69% of the relevant information needed to answer fully
  - You're missing 31% of critical medical research in each query

  Why This Score:

  MODERATE because:

  The Problem:
  - similarity_top_k=2 is too conservative - you're only retrieving 2 chunks per query
  - Your default in build_retriever() is 5, but retrieve_node overrides it to 2
  - Medical papers are long and dense - 2 chunks often aren't enough

  Evidence from your code:
  # In agents/nodes.py
  nodes = retrieve_nodes(query=enhanced_query, similarity_top_k=2)  # TOO LOW!

  Action:

  🔧 INCREASE similarity_top_k from 2 → 5 or even 8

  This is your biggest improvement opportunity. More chunks = better recall.

  ---
  3. Context Precision: 0.625 🟡 MODERATE

  What This Means:

  - 62.5% of retrieved chunks are actually relevant to the question
  - 37.5% are noise - irrelevant papers wasting context window space

  Why This Score:

  MODERATE because:

  Hybrid search is helping but not perfect:
  - Your hybrid search (BM25 + vector) should be doing better
  - Medical terms might be matching too broadly (e.g., "discharge" matches many papers)

  Possible causes:
  1. No metadata filtering - You extract 8 medical metadata fields but don't use them to pre-filter:
  # You extract: diagnoses, symptoms, ethnicities, etc.
  # But retrieval doesn't filter by these!
  2. Context-enriched query might be too noisy:
  enhanced_query = f"{user_query}\n\nHealth Context:\n{health_context}"
  # This adds A LOT of text - might dilute the core query

  Action:

  🔧 Add metadata filtering:
  # Pre-filter chunks by user's diagnoses/symptoms before vector search
  # Example: If user has PCOS, only search papers tagged with "PCOS"

  🔧 Add re-ranking step after retrieval (cross-encoder to re-score relevance)

  ---
  4. Answer Relevancy: 0.6159 🟡 MODERATE-LOW

  What This Means:

  - Only 62% of your responses directly address what the user asked
  - 38% of responses are somewhat off-topic or include unnecessary tangents

  Why This Score:

  MODERATE-LOW because:

  Possible causes:

  1. Auto-generated queries might be too vague:
  # When no user_message provided:
  default_query = f"My vaginal pH is {ph_value}. What does this mean?"
  # This is very broad - LLM might ramble
  2. No explicit instruction to answer the specific question:
  Your prompt says "Provide a clear, evidence-based answer" but doesn't emphasize "Answer the specific question asked, nothing more"
  3. Retrieved context might be tangentially related:
  If precision is 62.5%, the LLM sees some irrelevant papers and might discuss them

  Action:

  🔧 Improve prompt focus:
  system_prompt += """
  8. ANSWER THE QUESTION DIRECTLY - do not provide general information unless specifically asked
  9. Keep your response concise and focused on the user's specific concern
  """

  🔧 Better auto-query generation - include symptoms:
  # Instead of generic "What does this mean?"
  # Generate: "I have pH 5.2 with white discharge and itching. What could this indicate?"

  ---
  5. Factual Correctness: 0.2417 🔴 CRITICAL ISSUE

  What This Means:

  - Only 24% of your medical claims are factually correct compared to ground truth
  - 76% of facts are wrong, incomplete, or misinterpreted ⚠️

  Why This Score is SO LOW:

  This is your biggest problem. Here's why:

  Root Cause Analysis:

  1. Low Context Recall (68%) + Low Context Precision (62.5%) = Bad Factual Accuracy
    - You're missing relevant info (recall) AND retrieving wrong info (precision)
    - LLM tries to answer with incomplete/wrong context → wrong facts
  2. Only 2 chunks retrieved:
    - Medical facts often span multiple sections of a paper
    - 2 chunks can't capture: abstract + methods + results + conclusion
    - LLM makes educated guesses → wrong facts
  3. Your ground truth might be too specific:
    - If your test set expects exact percentages/statistics but your retrieved chunks don't have them, F1 score tanks
  4. Chain effect from other metrics:
  Poor Recall (69%)
  → Missing key facts
  → LLM fills gaps with hallucination
  → Wrong facts (24%)

  Why Faithfulness is 83% but Factual Correctness is 24%:

  - Faithfulness only checks: "Did you cite the documents?"
  - Factual Correctness checks: "Are the facts TRUE compared to expert answers?"
  - You CAN cite documents but still get facts wrong if:
    - You retrieved the WRONG documents
    - You only got PARTIAL information
    - LLM misinterpreted the context

  Action:

  🚨 URGENT FIXES:

  1. Increase retrieval to 5-8 chunks (fixes recall)
  2. Add re-ranking (fixes precision)
  3. Verify your test set ground truths - ensure they're achievable with your corpus
  4. Add answer validation node in LangGraph:
  # After generate_node, add verify_node that checks:
  # - Are there specific numbers? Verify them against retrieved context
  # - Are there cause-effect claims? Verify they're in the papers

  ---
  Summary: What's Happening in Your System

  The Story Your Metrics Tell:

  1. ✅ Your LLM follows instructions (Faithfulness 83%) and mostly cites the papers
  2. ❌ Your retriever is too conservative (Recall 69%) - missing 1 in 3 relevant papers
  3. ❌ Your retriever is a bit noisy (Precision 62%) - 1 in 3 retrieved papers is irrelevant
  4. 🟡 Your LLM sometimes rambles (Relevancy 62%) - not always directly answering the question
  5. 🔴 CRITICAL: The combination of #2 + #3 causes catastrophically wrong facts (Correctness 24%)

  Mental Model:

  Imagine asking: "What causes high pH?"

  Current system:
  1. Retrieves 2 chunks (recall 69% = probably missed key info)
  2. 1 of those 2 is noise (precision 62%)
  3. LLM gets 1 relevant chunk + 1 irrelevant chunk
  4. LLM cites them both (faithfulness 83%)
  5. But the answer is factually wrong (correctness 24%) because:
     - Missing half the story (low recall)
     - Has some wrong info (low precision)
     - LLM tries its best but fails

  ---
  Priority Action Plan (In Order):

  🔴 URGENT (Week 1):

  1. Increase similarity_top_k from 2 → 5-8
    - File: agents/nodes.py, line ~50
    - This alone will boost recall + factual correctness
  2. Verify your test set ground truths
    - Ensure reference answers are achievable with your 250-500 papers
    - If ground truth has facts not in your corpus, you'll always score low

  🟡 Important (Week 2):

  3. Add metadata-based pre-filtering
    - File: agents/llamaindex_retrieval.py
    - Filter by user's diagnoses/symptoms before vector search
  4. Add re-ranking step
    - Use a cross-encoder (e.g., cross-encoder/ms-marco-MiniLM-L-6-v2)
    - Re-score the 5-8 chunks, keep top 3

  🟢 Nice-to-Have (Week 3):

  5. Improve prompt for direct answers
  6. Add verification node in LangGraph

  ---
  Expected Impact After Fixes:
  ┌─────────────────────┬─────────┬─────────────┬────────────────────────────────────────┐
  │       Metric        │ Current │ After Fixes │               Reasoning                │
  ├─────────────────────┼─────────┼─────────────┼────────────────────────────────────────┤
  │ Faithfulness        │ 0.83    │ 0.85        │ Slight improvement from better context │
  ├─────────────────────┼─────────┼─────────────┼────────────────────────────────────────┤
  │ Context Recall      │ 0.69    │ 0.85+       │ More chunks = more coverage            │
  ├─────────────────────┼─────────┼─────────────┼────────────────────────────────────────┤
  │ Context Precision   │ 0.62    │ 0.75+       │ Metadata filtering + re-ranking        │
  ├─────────────────────┼─────────┼─────────────┼────────────────────────────────────────┤
  │ Answer Relevancy    │ 0.62    │ 0.75+       │ Better prompt + better context         │
  ├─────────────────────┼─────────┼─────────────┼────────────────────────────────────────┤
  │ Factual Correctness │ 0.24    │ 0.65+       │ Cascading improvement from all above   │
  └─────────────────────┴─────────┴─────────────┴────────────────────────────────────────┘
  Your factual correctness should triple just by increasing retrieval and adding re-ranking.

✻ Churned for 1m 1s

❯ help me understand Factual Correctness properly, reply in short 

⏺ Factual Correctness - Simple Explanation

  What It Actually Checks

  Factual Correctness compares your LLM's response against a reference answer (ground truth) and checks:
  - Did you get the facts right?
  - Are the specific claims accurate?
  - Do numbers, percentages, medical terms match?

  Formula: F1 score between extracted facts from your response vs. ground truth

  ---
  Example to Understand Your 24% Score

  Ground Truth (Reference Answer):

  "Normal vaginal pH is 3.8-4.5. Elevated pH (>4.5) can indicate bacterial vaginosis (BV), which affects 30% of women. BV is caused by an overgrowth of
  anaerobic bacteria like Gardnerella vaginalis."

  Your System's Response:

  "Your pH of 5.2 is elevated. This might suggest an imbalance in vaginal flora [1]. Bacterial vaginosis is a common condition [2]."

  What RAGAS Extracts and Compares:
  ┌─────────────────┬───────────────────────┬────────────────────────┬────────┐
  │    Fact Type    │     Ground Truth      │     Your Response      │ Match? │
  ├─────────────────┼───────────────────────┼────────────────────────┼────────┤
  │ Normal pH range │ 3.8-4.5               │ ❌ Not mentioned       │ ❌     │
  ├─────────────────┼───────────────────────┼────────────────────────┼────────┤
  │ Condition name  │ Bacterial vaginosis   │ ✅ Bacterial vaginosis │ ✅     │
  ├─────────────────┼───────────────────────┼────────────────────────┼────────┤
  │ Prevalence      │ 30% of women          │ ❌ Not mentioned       │ ❌     │
  ├─────────────────┼───────────────────────┼────────────────────────┼────────┤
  │ Cause bacteria  │ Gardnerella vaginalis │ ❌ Not mentioned       │ ❌     │
  ├─────────────────┼───────────────────────┼────────────────────────┼────────┤
  │ General info    │ "common condition"    │ ✅ "common condition"  │ ✅     │
  └─────────────────┴───────────────────────┴────────────────────────┴────────┘
  F1 Score: 2 matched / 5 total facts = 40% for this question

  ---
  Why Your Score is 24% (Not Higher)

  Problem 1: Missing Specific Facts

  Your response is vague compared to ground truth:
  - Ground truth: "3.8-4.5" (specific)
  - Your response: "elevated" (generic)

  Cause: Only retrieving 2 chunks - you're missing the papers with specific numbers

  ---
  Problem 2: Incomplete Medical Details

  Ground truth mentions:
  - Prevalence (30%)
  - Specific bacteria (Gardnerella vaginalis)
  - Mechanism (anaerobic overgrowth)

  Your response just says "imbalance" (generic) and "common" (vague)

  Cause: Low context recall (69%) - the detailed papers weren't retrieved

  ---
  Problem 3: Wrong Facts from Wrong Papers

  If context precision is 62%, 1 in 3 retrieved chunks is irrelevant

  Example bad scenario:
  - User asks about pH 5.2
  - Retriever finds:
    - ✅ Chunk about BV (correct)
    - ❌ Chunk about yeast infections (wrong condition)
  - LLM mixes info from both → wrong facts about the condition

  ---
  The Math Behind F1 Score

  Precision = (Correct facts in response) / (All facts in response)
  Recall = (Correct facts in response) / (All facts in ground truth)
  F1 = 2 × (Precision × Recall) / (Precision + Recall)

  Your 24% means:
  - You're missing most facts (low recall)
  - OR facts you state are wrong (low precision)
  - OR both

  ---
  Why Faithfulness (83%) ≠ Factual Correctness (24%)

  Faithfulness Checks:

  "Did you cite documents?" → Yes (83% of the time) ✅

  Factual Correctness Checks:

  "Are those documents' facts CORRECT and COMPLETE?" → No (only 24%) ❌

  Real Example:

  Faithfulness = High:
  "pH above 4.5 may indicate bacterial vaginosis [1]" ← Cited paper ✅

  Factual Correctness = Low:
  Ground truth says: "BV affects 30% of women and is caused by Gardnerella vaginalis"
  Your response: Missing prevalence, missing bacteria name ❌

  You cited papers (faithfulness) but missed key facts (correctness)

  ---
  TL;DR - Your 24% Score Means:

  1. You're answering questions, but missing critical details
    - Like saying "it might rain" when ground truth says "70% chance of rain at 3pm"
  2. Only 1 in 4 medical facts matches expert answers
    - The other 3 are: wrong, incomplete, or missing
  3. Root cause: Not retrieving enough context
    - similarity_top_k=2 → Missing papers with specific facts
    - Low recall (69%) → LLM guesses to fill gaps
    - Wrong guesses = wrong facts

  ---
  Quick Fix Impact

  Increase retrieval from 2 → 5 chunks:
  - More papers → More specific facts available
  - LLM has actual data (not guessing)
  - Factual correctness: 24% → 60-70%

  That's it. Your low score isn't because the LLM is bad - it's because you're starving it of information.
